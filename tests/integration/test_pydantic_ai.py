"""
Integration tests for the pydantic-ai framework integration.

These tests require a real pydantic-ai installation but do NOT require live
LLM API calls — they use pydantic_ai.models.test.TestModel, a fully offline,
deterministic stand-in built into the pydantic-ai package itself. Mirrors
the tests/integration/test_langgraph.py pattern (real framework, no network).

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic_ai", reason="pydantic-ai not installed")

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.test import TestModel

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.pydantic_ai import (
    instrument_agent_run,
    run_traced,
)


@pytest.mark.integration
class TestPydanticAIIntegration:
    async def test_run_traced_creates_agent_and_llm_spans(self, tmp_path: Path) -> None:
        """A plain agent run must produce an agent: root span and llm: span(s)."""
        agent: Agent[None, str] = Agent(TestModel(), name="echo-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("pydantic-ai-basic", record=True) as trace:
            result = await run_traced(agent, "say hello", tracer=t, trace=trace)

        assert result is not None
        agent_spans = [s for s in trace.spans if s.name.startswith("agent:")]
        assert agent_spans, (
            f"Expected agent: span. Got: {[s.name for s in trace.spans]}"
        )
        assert agent_spans[0].attributes["agent.name"] == "echo-agent"
        assert agent_spans[0].status == SpanStatus.OK

        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        assert llm_spans, f"Expected llm: span. Got: {[s.name for s in trace.spans]}"
        assert llm_spans[0].name == "llm:test"
        assert llm_spans[0].attributes["llm.model"] == "test"
        assert llm_spans[0].status == SpanStatus.OK
        assert llm_spans[0].parent_id == agent_spans[0].span_id

    async def test_llm_span_records_token_usage(self, tmp_path: Path) -> None:
        agent: Agent[None, str] = Agent(TestModel(), name="usage-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("pydantic-ai-usage") as trace:
            await run_traced(agent, "say hello", tracer=t, trace=trace)

        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.attributes["llm.usage.prompt_tokens"] > 0
        assert llm_span.attributes["llm.usage.completion_tokens"] > 0

        root_span = next(s for s in trace.spans if s.name.startswith("agent:"))
        assert root_span.attributes["agent.usage.input_tokens"] > 0
        assert root_span.attributes["agent.usage.output_tokens"] > 0
        assert root_span.attributes["agent.usage.requests"] >= 1

    async def test_tool_call_creates_child_tool_span(self, tmp_path: Path) -> None:
        agent: Agent[None, str] = Agent(TestModel(), name="weather-agent")

        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f"sunny in {city}"

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("pydantic-ai-tool") as trace:
            await run_traced(
                agent, "what is the weather in paris?", tracer=t, trace=trace
            )

        tool_spans = [s for s in trace.spans if s.name == "tool:get_weather"]
        assert tool_spans, (
            f"Expected tool:get_weather span. Got: {[s.name for s in trace.spans]}"
        )
        tool_span = tool_spans[0]
        assert tool_span.attributes["tool.name"] == "get_weather"
        assert tool_span.status == SpanStatus.OK
        assert tool_span.attributes["tool.output_length"] > 0

        agent_span = next(s for s in trace.spans if s.name.startswith("agent:"))
        assert tool_span.parent_id == agent_span.span_id

    async def test_output_validator_retry_tagged_on_llm_span(
        self, tmp_path: Path
    ) -> None:
        """A ModelRetry raised from @agent.output_validator must tag the
        *next* llm: span as a retry, distinct from a fresh turn."""
        attempts = {"n": 0}
        agent: Agent[None, str] = Agent(
            TestModel(), name="validator-agent", output_type=str, retries=3
        )

        @agent.output_validator
        def reject_first(output: str) -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ModelRetry("try again please")
            return output

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("pydantic-ai-validator-retry") as trace:
            await run_traced(agent, "hello", tracer=t, trace=trace)

        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        assert len(llm_spans) >= 2, (
            f"Expected at least 2 llm spans (original + retry), got {len(llm_spans)}"
        )
        retried = [s for s in llm_spans if s.attributes.get("llm.is_retry")]
        assert retried, "Expected at least one llm span tagged llm.is_retry"
        assert retried[0].attributes["llm.retry_reason"] == "output_validator"
        assert retried[0].attributes["llm.retry_index"] == 1

        root_span = next(s for s in trace.spans if s.name.startswith("agent:"))
        assert root_span.attributes["agent.retry_count"] == 1

    async def test_tool_retry_tagged_and_closed_ok_not_error(
        self, tmp_path: Path
    ) -> None:
        """A tool raising ModelRetry is pydantic-ai's own soft-retry signal —
        the tool span must close OK (tagged tool.retried), not ERROR."""
        attempts = {"n": 0}
        agent: Agent[None, str] = Agent(TestModel(), name="flaky-tool-agent")

        @agent.tool_plain
        def flaky(query: str) -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ModelRetry("transient failure, try again")
            return "ok"

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("pydantic-ai-tool-retry") as trace:
            await run_traced(agent, "call flaky", tracer=t, trace=trace)

        tool_spans = [s for s in trace.spans if s.name == "tool:flaky"]
        assert len(tool_spans) >= 1
        retried_tool = next(s for s in tool_spans if s.attributes.get("tool.retried"))
        assert retried_tool.status == SpanStatus.OK

        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        retried_llm = [s for s in llm_spans if s.attributes.get("llm.is_retry")]
        assert retried_llm, "Expected the follow-up llm span to be tagged as a retry"
        assert retried_llm[0].attributes["llm.retry_tool_name"] == "flaky"

    async def test_tool_exception_propagates_and_closes_spans_as_error(
        self, tmp_path: Path
    ) -> None:
        """A real exception (not ModelRetry) from a tool must propagate and
        close the agent/tool spans as ERROR, not OK."""
        agent: Agent[None, str] = Agent(TestModel(), name="boom-agent")

        @agent.tool_plain
        def boom() -> str:
            raise RuntimeError("tool exploded")

        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(RuntimeError, match="tool exploded"):
            with t.start_trace("pydantic-ai-error") as trace:
                await run_traced(agent, "call boom", tracer=t, trace=trace)

        agent_span = next(s for s in trace.spans if s.name.startswith("agent:"))
        assert agent_span.status == SpanStatus.ERROR
        assert agent_span.end_time is not None

        tool_span = next(s for s in trace.spans if s.name == "tool:boom")
        assert tool_span.status == SpanStatus.ERROR
        assert tool_span.end_time is not None
        assert any(e.name == "exception" for e in tool_span.events)

    async def test_instrument_agent_run_step_by_step(self, tmp_path: Path) -> None:
        """The lower-level instrument_agent_run() context manager must let a
        caller drive iteration manually and read .result afterwards."""
        agent: Agent[None, str] = Agent(TestModel(), name="step-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("pydantic-ai-step") as trace:
            async with instrument_agent_run(
                agent, "hello", tracer=t, trace=trace
            ) as run:
                node_types = []
                async for node in run:
                    node_types.append(type(node).__name__)
                result = run.result

        assert "UserPromptNode" in node_types
        assert "ModelRequestNode" in node_types
        assert "CallToolsNode" in node_types
        assert result is not None
        assert result.output is not None

    async def test_record_replay_round_trip_against_openai_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The generic httpx interceptor still records/replays pydantic-ai's
        OpenAI-backed traffic underneath the new framework-level spans —
        verified with respx mocking a fake OpenAI-compatible endpoint, no
        live API key required."""
        pytest.importorskip("openai", reason="openai SDK not installed")
        respx = pytest.importorskip("respx", reason="respx not installed")

        import json as _json

        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        chat_completion_body = _json.dumps(
            {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 3,
                    "total_tokens": 12,
                },
            }
        )

        with respx.mock(assert_all_called=False) as respx_mock:
            respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
                200, json=_json.loads(chat_completion_body)
            )

            t = Tracer(trace_dir=tmp_path)
            run_id = "pydantic-ai-openai-record"
            with t.start_trace("record-pass", record=True, run_id=run_id) as trace:
                # The AsyncOpenAI client (and its httpx.AsyncClient) must be
                # constructed *after* start_trace(record=True) is active —
                # agent-trace's transport patch only covers clients built
                # after recording is installed (RecordingTransport only
                # patches at init time, not retroactively).
                model = OpenAIChatModel(
                    "gpt-4o-mini", provider=OpenAIProvider(api_key="sk-test-key")
                )
                agent: Agent[None, str] = Agent(model, name="openai-backed-agent")
                await run_traced(agent, "say hi", tracer=t, trace=trace)

        fixture_path = tmp_path / run_id / "fixture.db"
        assert fixture_path.exists(), "fixture.db must be created during record"

        from agent_trace._replay.fixture import Fixture

        with Fixture(fixture_path) as f:
            exchange_count = f.exchange_count()
        assert exchange_count >= 1, (
            "Expected at least 1 recorded HTTP exchange from the "
            "OpenAI-backed pydantic-ai model call"
        )


@pytest.mark.integration
class TestSystemPromptPartCapture:
    """#3277: a developer had no way to tell, from a captured run, whether
    their system prompt/instructions actually made it into a given
    request without manually diffing two runs by hand — which
    ModelRequestPart types were sent (specifically SystemPromptPart
    presence/absence) and the resolved instructions value are now
    persisted onto the llm: span."""

    async def test_system_prompt_present_recorded_on_span(self, tmp_path: Path) -> None:
        agent: Agent[None, str] = Agent(
            TestModel(),
            name="sys-agent",
            system_prompt="You are a helpful assistant.",
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("sys-prompt-present") as trace:
            await run_traced(agent, "hi", tracer=t, trace=trace)

        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.attributes["llm.has_system_prompt_part"] is True
        assert (
            llm_span.attributes["llm.system_prompt_content"]
            == "You are a helpful assistant."
        )
        part_kinds = llm_span.attributes["llm.request_part_kinds"]
        assert "SystemPromptPart" in part_kinds

    async def test_system_prompt_absent_recorded_on_span(self, tmp_path: Path) -> None:
        agent: Agent[None, str] = Agent(TestModel(), name="no-sys-agent")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("sys-prompt-absent") as trace:
            await run_traced(agent, "hi", tracer=t, trace=trace)

        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.attributes["llm.has_system_prompt_part"] is False
        assert "llm.system_prompt_content" not in llm_span.attributes
        part_kinds = llm_span.attributes.get("llm.request_part_kinds", "")
        assert "SystemPromptPart" not in part_kinds
