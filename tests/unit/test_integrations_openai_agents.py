"""
Unit tests for agent_trace.integrations.openai_agents.AgentTraceHook.

Tests do NOT require the openai-agents package — they exercise the hook
methods directly using plain mock objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.openai_agents import AgentTraceHook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hook(tmp_path: Path) -> tuple[Tracer, Any, AgentTraceHook]:
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("hook-unit") as trace:
        hook = AgentTraceHook(tracer=t, trace=trace)
        return t, trace, hook


def _fake_agent(name: str = "test-agent", model: str = "gpt-4o") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.model = model
    return agent


def _fake_context() -> MagicMock:
    return MagicMock()


def _fake_usage(
    input_tokens: int, output_tokens: int, total_tokens: int | None = None
) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = total_tokens
    return usage


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestAgentTraceHookInit:
    def test_spans_dict_empty_at_start(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("init-test") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
        assert hook._spans == {}

    def test_tracer_and_trace_stored(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("init-test2") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
        assert hook._tracer is t
        assert hook._trace is trace


# ---------------------------------------------------------------------------
# on_agent_start / on_agent_end round-trip
# ---------------------------------------------------------------------------


class TestAgentSpanLifecycle:
    def test_on_agent_start_creates_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agent-start") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent("my-agent")
            hook.on_agent_start(ctx, agent)
            assert len(trace.spans) == 1
            assert "my-agent" in trace.spans[0].name

    def test_on_agent_end_closes_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agent-end") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            hook.on_agent_start(ctx, agent)
            hook.on_agent_end(ctx, agent, output="result")
            assert trace.spans[0].end_time is not None
            assert trace.spans[0].status == SpanStatus.OK

    def test_on_agent_end_clears_registry(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agent-clear") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            hook.on_agent_start(ctx, agent)
            hook.on_agent_end(ctx, agent, output="done")
            assert hook._spans == {}


# ---------------------------------------------------------------------------
# on_llm_start / on_llm_end — token usage (B2 bug fix coverage)
# ---------------------------------------------------------------------------


class TestLLMTokenUsage:
    def test_on_llm_end_records_prompt_tokens(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-tokens") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            hook.on_agent_start(ctx, agent)
            hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = _fake_usage(input_tokens=7, output_tokens=13)
            hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.prompt_tokens"] == 7

    def test_on_llm_end_records_completion_tokens(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-compl") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            hook.on_agent_start(ctx, agent)
            hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = _fake_usage(input_tokens=7, output_tokens=13)
            hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.completion_tokens"] == 13

    def test_on_llm_end_records_total_tokens_when_explicit(
        self, tmp_path: Path
    ) -> None:
        """total_tokens must be recorded when the SDK provides it explicitly."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-total-explicit") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            hook.on_agent_start(ctx, agent)
            hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = _fake_usage(
                input_tokens=5, output_tokens=10, total_tokens=15
            )
            hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.total_tokens"] == 15

    def test_on_llm_end_computes_total_tokens_when_absent(self, tmp_path: Path) -> None:
        """total_tokens must be computed as prompt + completion when SDK omits it."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-total-computed") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            hook.on_agent_start(ctx, agent)
            hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            # total_tokens is None — must be computed
            response.usage = _fake_usage(
                input_tokens=6, output_tokens=9, total_tokens=None
            )
            hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.total_tokens"] == 15  # 6 + 9

    def test_on_llm_end_tolerates_no_usage(self, tmp_path: Path) -> None:
        """on_llm_end must not raise if response.usage is None."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-no-usage") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            hook.on_agent_start(ctx, agent)
            hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = None
            hook.on_llm_end(ctx, agent, response)  # must not raise

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.end_time is not None


# ---------------------------------------------------------------------------
# on_tool_start / on_tool_end round-trip
# ---------------------------------------------------------------------------


class TestToolSpanLifecycle:
    def test_tool_span_is_child_of_agent_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-child") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            tool = MagicMock()
            tool.name = "search"

            hook.on_agent_start(ctx, agent)
            agent_span = next(s for s in trace.spans if "agent" in s.name)

            hook.on_tool_start(ctx, agent, tool)
            tool_span = next(s for s in trace.spans if "tool" in s.name)

            assert tool_span.parent_id == agent_span.span_id

    def test_on_tool_end_closes_span_ok(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-end") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            tool = MagicMock()
            tool.name = "calc"

            hook.on_agent_start(ctx, agent)
            hook.on_tool_start(ctx, agent, tool)
            hook.on_tool_end(ctx, agent, tool, result="42")

            tool_span = next(s for s in trace.spans if "tool" in s.name)
            assert tool_span.status == SpanStatus.OK
            assert tool_span.end_time is not None
            assert hook._spans == {} or all(
                k for k in hook._spans if "tool:calc" not in k
            )


# ---------------------------------------------------------------------------
# _enrich_step_span was removed (B1 bug fix — dead code)
# ---------------------------------------------------------------------------


class TestDeadCodeRemoved:
    def test_enrich_step_span_does_not_exist(self) -> None:
        """_enrich_step_span was dead code — verify it's been removed."""
        import agent_trace.integrations.openai_agents as oa_mod

        assert not hasattr(oa_mod, "_enrich_step_span"), (
            "_enrich_step_span is dead code (never called); it should have been removed"
        )
