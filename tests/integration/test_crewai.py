"""
Integration tests for the crewAI event-bus integration.

These tests require a real crewAI installation but do NOT require live LLM
API calls — a custom ``crewai.BaseLLM`` subclass returns scripted responses
entirely offline (no network), exercising the real ``Agent``/``Task``/
``Crew`` runtime and the real, threaded ``crewai_event_bus`` dispatch path
instead of hand-constructed events.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("crewai", reason="crewai not installed")

from crewai import Agent, BaseLLM, Crew, Process, Task
from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import llm_call_context
from crewai.tools import tool

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.crewai import CrewAITracer


# crewAI's Agent construction validates an LLM string against known
# providers, but does not make a network call at construction time — a
# placeholder key is enough (this suite never makes a real LLM call; see
# ScriptedLLM below). Applied via a module-scoped, self-undoing fixture
# (not a bare os.environ.setdefault at import time) so a fake key never
# leaks into unrelated tests — e.g. openai-agents integration tests
# further down the same pytest session — that treat a *present*
# OPENAI_API_KEY as "run this live-API test".
@pytest.fixture(autouse=True, scope="module")
def _openai_api_key_placeholder() -> Any:
    mp = pytest.MonkeyPatch()
    if not os.environ.get("OPENAI_API_KEY"):
        mp.setenv("OPENAI_API_KEY", "sk-test-placeholder-not-a-real-key")
    yield
    mp.undo()


class ScriptedLLM(BaseLLM):  # type: ignore[misc]
    """A fully offline ``BaseLLM`` implementation returning canned
    responses in order, one per ``call()`` invocation (clamped to the last
    response once exhausted).

    Emits the exact same ``llm_call_started``/``llm_call_completed`` event
    pair (via the private ``_emit_call_*_event`` helpers) that every real
    provider completion class in the installed crewai package emits from
    inside its own ``call()`` — confirmed by direct inspection of
    ``crewai.llms.providers.*.completion`` — so this exercises agent-trace's
    handlers against the exact same event shapes a live OpenAI/Anthropic/
    Gemini call would produce, without making any network request.
    """

    def __init__(self, responses: list[str], **kwargs: Any) -> None:
        super().__init__(model="fake-model", **kwargs)
        self._responses = list(responses)
        self._call_count = 0

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        with llm_call_context():
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )
            index = min(self._call_count, len(self._responses) - 1)
            text = self._responses[index]
            self._call_count += 1
            self._emit_call_completed_event(
                response=text,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                finish_reason="stop",
            )
            return text

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 4096


@tool("lookup")
def lookup(query: str) -> str:
    """Look something up (scripted — returns a canned string)."""
    return f"result for {query}"


def _single_agent_crew(llm: BaseLLM, with_tool: bool = False) -> Crew:
    agent = Agent(
        role="researcher",
        goal="answer the question",
        backstory="a careful, concise research assistant",
        llm=llm,
        tools=[lookup] if with_tool else [],
        verbose=False,
    )
    task = Task(
        description="What is the answer?",
        expected_output="A short answer.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential)


@pytest.mark.integration
class TestCrewAIIntegration:
    def test_simple_crew_run_produces_full_span_tree(self, tmp_path: Path) -> None:
        """A single-agent, single-task, no-tool crew run must produce a
        fully closed crew -> task -> agent -> llm span tree with no
        network calls."""
        llm = ScriptedLLM(["Final Answer: 42 is the answer"])
        crew = _single_agent_crew(llm)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("crewai-integration-test") as trace:
            with CrewAITracer(tracer=t, trace=trace):
                result = crew.kickoff()

        assert "42" in str(result)

        names = {s.name for s in trace.spans}
        assert any(n.startswith("crew:") for n in names)
        assert any(n.startswith("task:") for n in names)
        assert any(n.startswith("agent:") for n in names)
        assert any(n.startswith("llm:") for n in names)

        # The whole point of CrewAITracer.close() flushing the event bus:
        # every span must be fully closed by the time the `with` block has
        # exited, not just eventually.
        for span in trace.spans:
            assert span.end_time is not None, f"{span.name} was left open"
            assert span.status == SpanStatus.OK

    def test_crew_run_with_tool_call_closes_every_span(self, tmp_path: Path) -> None:
        """Reproduces the exact shape that exposed the close-before-open
        race: two LLM calls (one ReAct action step, one final answer) with a
        real tool invocation in between, run through the real, threaded
        crewai_event_bus dispatch path. Every span (crew/task/agent/llm x2/
        tool) must close — none may be left at SpanStatus.UNSET."""
        llm = ScriptedLLM(
            [
                "Thought: I should look something up.\n"
                "Action: lookup\n"
                'Action Input: {"query": "the answer"}',
                "Final Answer: 42 is the answer",
            ]
        )
        crew = _single_agent_crew(llm, with_tool=True)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("crewai-tool-integration-test") as trace:
            with CrewAITracer(tracer=t, trace=trace):
                result = crew.kickoff()

        assert "42" in str(result)

        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        tool_spans = [s for s in trace.spans if s.name.startswith("tool:")]
        assert len(llm_spans) == 2, (
            f"expected 2 llm: spans (action step + final answer), "
            f"got {[s.name for s in trace.spans]}"
        )
        assert len(tool_spans) == 1

        for span in trace.spans:
            assert span.end_time is not None, (
                f"{span.name} was left open (SpanStatus.UNSET) — the "
                "close-before-open race is not being handled"
            )
            assert span.status == SpanStatus.OK

        # Token usage/finish_reason must have landed on both LLM spans,
        # regardless of which of that call's two handlers (started vs
        # completed) happened to run first on crewAI's thread pool.
        for llm_span in llm_spans:
            assert llm_span.attributes.get("llm.model") == "fake-model"
            assert llm_span.attributes.get("llm.usage.total_tokens") == 15
            assert llm_span.attributes.get("llm.finish_reason") == "stop"

        tool_span = tool_spans[0]
        assert tool_span.attributes.get("tool.name") == "lookup"
        assert tool_span.attributes.get("tool.output_length", 0) > 0

    def test_repeated_runs_never_leave_a_span_open(self, tmp_path: Path) -> None:
        """The close-before-open race (see the module docstring) is timing
        dependent — run the tool-call shape several times so a regression
        that only reintroduces the race intermittently still gets caught."""
        for i in range(8):
            llm = ScriptedLLM(
                [
                    "Thought: look it up.\n"
                    "Action: lookup\n"
                    'Action Input: {"query": "q"}',
                    "Final Answer: done",
                ]
            )
            crew = _single_agent_crew(llm, with_tool=True)

            t = Tracer(trace_dir=tmp_path / f"run-{i}")
            with t.start_trace(f"crewai-repeat-{i}") as trace:
                with CrewAITracer(tracer=t, trace=trace):
                    crew.kickoff()

            still_open = [s.name for s in trace.spans if s.end_time is None]
            assert not still_open, (
                f"run {i}: spans left open (SpanStatus.UNSET): {still_open}"
            )

    def test_failed_llm_call_closes_span_as_error(self, tmp_path: Path) -> None:
        """A raised exception inside a scripted LLM's call() must surface
        as an ERROR-status llm: span, not leave it open or silently drop
        it — mirrors what a real provider 4xx/5xx would produce."""

        class FailingLLM(ScriptedLLM):
            def call(self, messages: Any, **kwargs: Any) -> str:
                with llm_call_context():
                    self._emit_call_started_event(messages=messages)
                    self._emit_call_failed_event(error="simulated provider error")
                    raise RuntimeError("simulated provider error")

        llm = FailingLLM(["unused"])
        crew = _single_agent_crew(llm)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("crewai-failure-test") as trace:
            with CrewAITracer(tracer=t, trace=trace):
                with pytest.raises(Exception):
                    crew.kickoff()

        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        assert llm_spans, "expected at least one llm: span"
        assert llm_spans[0].status == SpanStatus.ERROR
        assert llm_spans[0].end_time is not None
