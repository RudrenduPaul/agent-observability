"""
Integration tests for the Agno framework integration.

These tests require a real agno installation but do NOT require live LLM API
calls — a minimal ``FakeModel`` subclass overrides Agno's abstract streaming
methods (``invoke_stream``/``ainvoke_stream``), which is the code path Agno's
own Agent/Team run loop actually calls when ``stream=True`` (confirmed
against agno==2.7.1: ``agno/agent/_run.py``'s ``arun_dispatch`` always drives
the model through ``Model.aresponse_stream`` -> ``Model.ainvoke_stream`` when
streaming events, never the non-streaming ``ainvoke``/``response`` path).

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("agno", reason="agno not installed")

from agno.agent.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.team.team import Team

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.agno import (
    AgnoTracer,
    instrument_agent_arun,
)

# ---------------------------------------------------------------------------
# Fake Agno models — no network calls, no API key required
# ---------------------------------------------------------------------------


@dataclass
class ScriptedModel(Model):
    """A Model whose streamed responses are scripted per-call.

    Each element of ``script`` is returned (as a single-chunk stream) on
    successive calls; the last element repeats once the script is exhausted.
    """

    id: str = "scripted-model"
    script: list[ModelResponse] = field(default_factory=list)
    raise_in_process: bool = False
    _call_count: int = field(default=0, init=False, repr=False)

    def _next_response(self) -> ModelResponse:
        if self.raise_in_process:
            # Simulates an in-process crash inside Agno's own model-handling
            # code (like #5298's UnboundLocalError in agno/models/base.py) —
            # never reaches the HTTP layer at all.
            raise UnboundLocalError("simulated in-process crash")
        idx = min(self._call_count, len(self.script) - 1)
        self._call_count += 1
        return self.script[idx]

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next_response()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next_response()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._next_response()

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        yield self._next_response()

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


def _content_model(model_id: str, content: str) -> ScriptedModel:
    return ScriptedModel(id=model_id, script=[ModelResponse(role="assistant", content=content)])


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    return str(eval(expression))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAgnoIntegration:
    async def test_agent_run_produces_agent_and_llm_spans(self, tmp_path: Path) -> None:
        agent = Agent(model=_content_model("fake-1", "hello"), name="solo-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-basic") as trace:
            result = await instrument_agent_arun(agent, "hi", tracer=t, trace=trace)

        assert result is not None
        assert result.content == "hello"

        agent_spans = [s for s in trace.spans if s.name.startswith("agent:")]
        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        assert agent_spans, f"Expected an agent: span. Got: {[s.name for s in trace.spans]}"
        assert llm_spans, f"Expected an llm: span. Got: {[s.name for s in trace.spans]}"
        assert agent_spans[0].attributes["agno.agent.name"] == "solo-agent"

    async def test_all_spans_closed_and_ok_on_clean_run(self, tmp_path: Path) -> None:
        agent = Agent(model=_content_model("fake-1", "hello"), name="solo-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-clean") as trace:
            await instrument_agent_arun(agent, "hi", tracer=t, trace=trace)

        unclosed = [s for s in trace.spans if s.end_time is None]
        non_ok = [s for s in trace.spans if s.status != SpanStatus.OK]
        assert unclosed == [], f"Spans left open: {[s.name for s in unclosed]}"
        assert non_ok == [], f"Non-OK spans on clean run: {[(s.name, s.status) for s in non_ok]}"

    async def test_llm_span_has_token_usage(self, tmp_path: Path) -> None:
        agent = Agent(model=_content_model("fake-1", "hello"), name="solo-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-tokens") as trace:
            await instrument_agent_arun(agent, "hi", tracer=t, trace=trace)

        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        # ScriptedModel reports no real usage, but the attribute keys must
        # exist so a real provider's numbers land in the same place.
        assert "llm.usage.prompt_tokens" in llm_span.attributes
        assert "llm.usage.completion_tokens" in llm_span.attributes
        assert "llm.usage.total_tokens" in llm_span.attributes

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    async def test_tool_call_produces_child_span(self, tmp_path: Path) -> None:
        model = ScriptedModel(
            id="fake-tool",
            script=[
                ModelResponse(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "2+2"}',
                            },
                        }
                    ],
                ),
                ModelResponse(role="assistant", content="the answer is 4"),
            ],
        )
        agent = Agent(model=model, name="tool-agent", tools=[calculator])

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-tool") as trace:
            result = await instrument_agent_arun(agent, "what is 2+2", tracer=t, trace=trace)

        assert result.content == "the answer is 4"

        tool_spans = [s for s in trace.spans if s.name == "tool:calculator"]
        assert tool_spans, f"Expected a tool:calculator span. Got: {[s.name for s in trace.spans]}"
        assert tool_spans[0].status == SpanStatus.OK
        assert tool_spans[0].attributes["tool.result_length"] == len("4")

        # Two model requests: one that emits the tool call, one after the
        # tool result is fed back.
        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        assert len(llm_spans) == 2
        assert all(s.end_time is not None for s in llm_spans)

    # ------------------------------------------------------------------
    # In-process exceptions (never reach the HTTP layer)
    # ------------------------------------------------------------------

    async def test_in_process_exception_produces_error_span(self, tmp_path: Path) -> None:
        """A crash entirely inside Agno's own response-processing code (no
        HTTP call ever made) must still surface as an ERROR span — this is
        exactly the #5298 UnboundLocalError scenario reported upstream.
        """
        model = ScriptedModel(id="fake-crash", raise_in_process=True)
        agent = Agent(model=model, name="crash-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-crash") as trace:
            result = await instrument_agent_arun(agent, "hi", tracer=t, trace=trace)

        assert result is None  # RunErrorEvent path yields no RunOutput

        agent_spans = [s for s in trace.spans if s.name.startswith("agent:")]
        assert agent_spans, f"Expected an agent: span. Got: {[s.name for s in trace.spans]}"
        error_span = agent_spans[0]
        assert error_span.status == SpanStatus.ERROR
        assert error_span.end_time is not None
        messages = [e.attributes.get("exception.message", "") for e in error_span.events]
        assert any("simulated in-process crash" in m for m in messages), messages

    async def test_in_process_exception_leaves_no_open_spans(self, tmp_path: Path) -> None:
        """The in-flight ModelRequestStarted span (no matching Completed event
        ever arrives once the run errors) must be force-closed by the safety
        net, not leaked open forever."""
        model = ScriptedModel(id="fake-crash", raise_in_process=True)
        agent = Agent(model=model, name="crash-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-crash-leak") as trace:
            await instrument_agent_arun(agent, "hi", tracer=t, trace=trace)

        unclosed = [s for s in trace.spans if s.end_time is None]
        assert unclosed == [], f"Spans left open after an in-process crash: {[s.name for s in unclosed]}"

    # ------------------------------------------------------------------
    # Team delegation — per-team-member attribution
    # ------------------------------------------------------------------

    async def test_team_delegation_attributes_member_run_to_member_agent(
        self, tmp_path: Path
    ) -> None:
        """When a Team delegates to a member Agent, the member's own run span
        must be a child of the team's run span and carry the member's own
        agent identity — not be indistinguishable team-only traffic. This is
        the exact gap upstream issue #5326 describes."""

        @dataclass
        class DelegatingModel(ScriptedModel):
            def _next_response(self) -> ModelResponse:  # type: ignore[override]
                self._call_count += 1
                if self._call_count == 1:
                    return ModelResponse(
                        role="assistant",
                        tool_calls=[
                            {
                                "id": "call_delegate",
                                "type": "function",
                                "function": {
                                    "name": "delegate_task_to_member",
                                    "arguments": (
                                        '{"member_id": "member-agent", "task": "say hi"}'
                                    ),
                                },
                            }
                        ],
                    )
                return ModelResponse(role="assistant", content="leader final answer")

        member = Agent(
            model=_content_model("member-model", "member says hi"),
            name="member-agent",
            id="member-agent",
        )
        team = Team(
            model=DelegatingModel(id="leader-model"),
            name="my-team",
            members=[member],
            stream_member_events=True,
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-delegate") as trace:
            result = await instrument_agent_arun(team, "hi team", tracer=t, trace=trace)

        assert result.content == "leader final answer"

        team_span = next(s for s in trace.spans if s.name == "team:my-team")
        member_spans = [s for s in trace.spans if s.name == "agent:member-agent"]
        assert member_spans, (
            f"Expected a member run span parented under the team. "
            f"Got: {[s.name for s in trace.spans]}"
        )
        assert member_spans[0].parent_id == team_span.span_id
        assert member_spans[0].status == SpanStatus.OK

        delegate_tool_span = next(
            s for s in trace.spans if s.name == "tool:delegate_task_to_member"
        )
        assert delegate_tool_span.parent_id == team_span.span_id
        # The delegation tool span carries the member run's id, correlating
        # the delegation call with the member's own span.
        assert delegate_tool_span.attributes.get("agno.child_run_id") == (
            member_spans[0].attributes["agno.run_id"]
        )

    # ------------------------------------------------------------------
    # AgnoTracer used directly (hook-based usage, not the convenience wrapper)
    # ------------------------------------------------------------------

    async def test_hook_based_usage_matches_convenience_wrapper(self, tmp_path: Path) -> None:
        agent = Agent(model=_content_model("fake-1", "hello"), name="solo-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agno-hook") as trace:
            hook = AgnoTracer(tracer=t, trace=trace)
            async for event in agent.arun("hi", stream=True, stream_events=True):
                hook.process_event(event)

        agent_spans = [s for s in trace.spans if s.name.startswith("agent:")]
        assert agent_spans
        assert agent_spans[0].status == SpanStatus.OK
        assert hook._run_spans == {}
