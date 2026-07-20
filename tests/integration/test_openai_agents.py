"""
Integration tests for the OpenAI Agents SDK integration.

Run with: uv run pytest tests/integration/ -m integration

Requirements:
  - pip install agent-observability-trace-cli[openai-agents]
  - OPENAI_API_KEY set in the environment

These tests are NOT run in standard CI because they make live API calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("agents", reason="openai-agents not installed")


def _skip_without_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping live API test")


@pytest.mark.integration
class TestOpenAIAgentsIntegration:
    async def test_agent_hook_captures_spans(self, tmp_path: Path) -> None:
        """Run a real openai-agents Agent through Runner.run and assert spans fire."""
        _skip_without_key()

        from agents import Agent, Runner

        from agent_trace import Tracer
        from agent_trace.integrations.openai_agents import AgentTraceHook

        agent = Agent(
            name="echo-agent",
            instructions="Respond with exactly one word: 'done'",
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("openai-agents-test", record=True) as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            result = await Runner.run(agent, "say the word", hooks=hook, max_turns=1)

        assert result is not None
        # AgentTraceHook fires on_agent_start -- at least one span
        assert len(trace.spans) >= 1
        agent_spans = [s for s in trace.spans if s.name.startswith("agent:")]
        assert agent_spans, (
            f"Expected at least one agent: span. Got: {[s.name for s in trace.spans]}"
        )
        assert agent_spans[0].attributes.get("agent.name") == "echo-agent"

    async def test_record_replay_round_trip(self, tmp_path: Path) -> None:
        """Record a live agent run, replay it, assert fixture has exchanges.

        Verifies the core record/replay invariant: HTTP transport interception
        captures the LLM call during record; replay serves those bytes from
        SQLite without hitting the network.
        """
        _skip_without_key()

        from agents import Agent, Runner

        from agent_trace import Tracer, replay
        from agent_trace.integrations.openai_agents import AgentTraceHook

        agent = Agent(
            name="echo-agent",
            instructions="Respond with exactly one word: 'done'",
        )

        run_id = "openai-agents-replay-test"
        t = Tracer(trace_dir=tmp_path)

        # Record pass
        with t.start_trace("record-pass", record=True, run_id=run_id) as record_trace:
            hook = AgentTraceHook(tracer=t, trace=record_trace)
            await Runner.run(agent, "say the word", hooks=hook, max_turns=1)

        record_span_names = [s.name for s in record_trace.spans]
        assert len(record_span_names) >= 1

        fixture_path = tmp_path / run_id / "fixture.db"
        assert fixture_path.exists(), "fixture.db must be created during record"

        from agent_trace._replay.fixture import Fixture

        with Fixture(fixture_path) as f:
            exchange_count = f.exchange_count()
        assert exchange_count >= 1, (
            f"Expected at least 1 recorded HTTP exchange, got {exchange_count}. "
            "Transport interception may not have captured the LLM call."
        )

        # Replay pass -- AGENT_TRACE_NETWORK_GUARD=1 blocks any live HTTP
        with replay(run_id, trace_dir=tmp_path) as ctx:
            assert ctx.fixture.exchange_count() == exchange_count

        trace_json = tmp_path / run_id / "trace.json"
        assert trace_json.exists()
        saved = json.loads(trace_json.read_text())
        assert [s["name"] for s in saved["spans"]] == record_span_names

    async def test_tool_span_captured(self, tmp_path: Path) -> None:
        """An agent with a function tool must emit tool: spans."""
        _skip_without_key()

        from agents import Agent, Runner, function_tool

        from agent_trace import Tracer
        from agent_trace.integrations.openai_agents import AgentTraceHook

        @function_tool
        def get_weather(city: str) -> str:
            """Return the weather for a city."""
            return f"Sunny in {city}"

        agent = Agent(
            name="weather-agent",
            instructions="Use the get_weather tool to answer questions.",
            tools=[get_weather],
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-test", record=True) as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            await Runner.run(
                agent, "What is the weather in Paris?", hooks=hook, max_turns=2
            )

        tool_spans = [s for s in trace.spans if s.name.startswith("tool:")]
        assert tool_spans, (
            f"Expected at least one tool: span. Got: {[s.name for s in trace.spans]}"
        )
        assert tool_spans[0].attributes.get("tool.name") == "get_weather"
        # Persist the actual tool result content, not just its length.
        assert "Sunny in Paris" in tool_spans[0].attributes.get("tool.result", "")
        assert tool_spans[0].attributes.get("tool.result_length") == len(
            "Sunny in Paris"
        )

    async def test_llm_span_records_model_settings_and_tool_call_presence(
        self, tmp_path: Path
    ) -> None:
        """llm: spans must record model_settings + whether a response had tool calls."""
        _skip_without_key()

        from agents import Agent, ModelSettings, Runner, function_tool

        from agent_trace import Tracer
        from agent_trace.integrations.openai_agents import AgentTraceHook

        @function_tool
        def get_weather(city: str) -> str:
            """Return the weather for a city."""
            return f"Sunny in {city}"

        agent = Agent(
            name="weather-agent-settings",
            instructions="Always call get_weather for weather questions.",
            tools=[get_weather],
            model_settings=ModelSettings(verbosity="low"),
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("model-settings-test", record=True) as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            await Runner.run(
                agent, "What is the weather in Tokyo?", hooks=hook, max_turns=2
            )

        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        assert llm_spans, f"Expected llm: spans. Got: {[s.name for s in trace.spans]}"
        assert llm_spans[0].attributes.get("llm.model_settings.verbosity") == "low"
        assert any(
            s.attributes.get("llm.response.has_tool_calls") is True for s in llm_spans
        )

    async def test_streamed_run_captures_events_and_final_output(
        self, tmp_path: Path
    ) -> None:
        """instrument_runner_streamed must yield real events and stay proxy-transparent."""
        _skip_without_key()

        from agents import Agent

        from agent_trace import Tracer
        from agent_trace.integrations.openai_agents import instrument_runner_streamed

        agent = Agent(
            name="echo-agent-streamed",
            instructions="Respond with exactly one word: 'done'",
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("streamed-test", record=True) as trace:
            streamed = await instrument_runner_streamed(
                agent, "say the word", tracer=t, trace=trace, max_turns=1
            )
            events = [event async for event in streamed.stream_events()]

        assert events, "Expected at least one stream event"
        assert streamed.final_output is not None

        root_spans = [s for s in trace.spans if s.name == "agent_run_streamed"]
        assert root_spans, (
            f"Expected agent_run_streamed span. Got: {[s.name for s in trace.spans]}"
        )
        assert root_spans[0].status.value == "OK"
        assert len(root_spans[0].events) == len(events)

    async def test_handoff_span_has_duration(self, tmp_path: Path) -> None:
        """A multi-agent handoff must produce a duration-based handoff:<a>-><b> span."""
        _skip_without_key()

        from agents import Agent, Runner

        from agent_trace import Tracer
        from agent_trace.integrations.openai_agents import AgentTraceHook

        writer = Agent(
            name="writer-agent",
            instructions="Write a one-sentence answer using the handed-off context.",
        )
        researcher = Agent(
            name="researcher-agent",
            instructions=(
                "Always hand off immediately to writer-agent; never answer directly."
            ),
            handoffs=[writer],
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("handoff-test", record=True) as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            await Runner.run(
                researcher, "What does agent-trace do?", hooks=hook, max_turns=3
            )

        handoff_spans = [s for s in trace.spans if s.name.startswith("handoff:")]
        assert handoff_spans, (
            f"Expected a handoff: span. Got: {[s.name for s in trace.spans]}"
        )
        assert handoff_spans[0].end_time is not None
        assert handoff_spans[0].duration_ms is not None
