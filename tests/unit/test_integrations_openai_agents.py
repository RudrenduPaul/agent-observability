"""
Unit tests for agent_trace.integrations.openai_agents.

Tests do NOT require the openai-agents package — they exercise the hook
methods directly using plain mock objects.  Every hook method on
``AgentTraceHook``/``AgentTraceRealtimeHook`` is ``async def`` (matching the
real openai-agents SDK's ``RunHooksBase`` interface, which the SDK itself
``await``s), so every test that drives them is ``async def`` too
(``asyncio_mode = "auto"`` in pyproject.toml runs these without a decorator).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.openai_agents import (
    AgentTraceHook,
    AgentTraceRealtimeHook,
    _response_has_tool_calls,
    instrument_runner,
    instrument_runner_streamed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hook(tmp_path: Path) -> tuple[Tracer, Any, AgentTraceHook]:
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("hook-unit") as trace:
        hook = AgentTraceHook(tracer=t, trace=trace)
        return t, trace, hook


def _fake_agent(
    name: str = "test-agent", model: str = "gpt-4o", model_settings: Any = None
) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.model = model
    agent.model_settings = model_settings
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


def _fake_output_item(item_type: str) -> MagicMock:
    item = MagicMock()
    item.type = item_type
    return item


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

    def test_hook_subclasses_run_hooks_base_when_sdk_installed(self) -> None:
        """AgentTraceHook must satisfy Runner.run()'s isinstance(hooks, RunHooksBase) check."""
        agents_lifecycle = pytest.importorskip("agents.lifecycle")
        assert isinstance(
            AgentTraceHook.__new__(AgentTraceHook), agents_lifecycle.RunHooksBase
        )


# ---------------------------------------------------------------------------
# on_agent_start / on_agent_end round-trip
# ---------------------------------------------------------------------------


class TestAgentSpanLifecycle:
    async def test_on_agent_start_creates_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agent-start") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent("my-agent")
            await hook.on_agent_start(ctx, agent)
            assert len(trace.spans) == 1
            assert "my-agent" in trace.spans[0].name

    async def test_on_agent_end_closes_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agent-end") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_agent_end(ctx, agent, output="result")
            assert trace.spans[0].end_time is not None
            assert trace.spans[0].status == SpanStatus.OK

    async def test_on_agent_end_clears_registry(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agent-clear") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_agent_end(ctx, agent, output="done")
            assert hook._spans == {}


# ---------------------------------------------------------------------------
# on_llm_start / on_llm_end — token usage (B2 bug fix coverage)
# ---------------------------------------------------------------------------


class TestLLMTokenUsage:
    async def test_on_llm_end_records_prompt_tokens(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-tokens") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = _fake_usage(input_tokens=7, output_tokens=13)
            response.output = []
            await hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.prompt_tokens"] == 7

    async def test_on_llm_end_records_completion_tokens(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-compl") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = _fake_usage(input_tokens=7, output_tokens=13)
            response.output = []
            await hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.completion_tokens"] == 13

    async def test_on_llm_end_records_total_tokens_when_explicit(
        self, tmp_path: Path
    ) -> None:
        """total_tokens must be recorded when the SDK provides it explicitly."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-total-explicit") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = _fake_usage(
                input_tokens=5, output_tokens=10, total_tokens=15
            )
            response.output = []
            await hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.total_tokens"] == 15

    async def test_on_llm_end_computes_total_tokens_when_absent(
        self, tmp_path: Path
    ) -> None:
        """total_tokens must be computed as prompt + completion when SDK omits it."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-total-computed") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            # total_tokens is None — must be computed
            response.usage = _fake_usage(
                input_tokens=6, output_tokens=9, total_tokens=None
            )
            response.output = []
            await hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.usage.total_tokens"] == 15  # 6 + 9

    async def test_on_llm_end_tolerates_no_usage(self, tmp_path: Path) -> None:
        """on_llm_end must not raise if response.usage is None."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-no-usage") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = None
            response.output = []
            await hook.on_llm_end(ctx, agent, response)  # must not raise

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.end_time is not None


# ---------------------------------------------------------------------------
# on_llm_start — model_settings (reasoning_effort / verbosity)
# ---------------------------------------------------------------------------


class TestModelSettings:
    async def test_records_reasoning_effort_from_dict(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("model-settings-dict") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            model_settings = MagicMock()
            model_settings.reasoning = {"effort": "high"}
            model_settings.verbosity = "low"
            agent = _fake_agent(model_settings=model_settings)

            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.model_settings.reasoning_effort"] == "high"
            assert llm_span.attributes["llm.model_settings.verbosity"] == "low"

    async def test_records_reasoning_effort_from_attr_style(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("model-settings-attr") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            reasoning = MagicMock()
            reasoning.effort = "minimal"
            model_settings = MagicMock()
            model_settings.reasoning = reasoning
            model_settings.verbosity = None
            agent = _fake_agent(model_settings=model_settings)

            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert (
                llm_span.attributes["llm.model_settings.reasoning_effort"] == "minimal"
            )
            assert "llm.model_settings.verbosity" not in llm_span.attributes

    async def test_tolerates_missing_model_settings(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("model-settings-none") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent(model_settings=None)

            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert "llm.model_settings.reasoning_effort" not in llm_span.attributes


class TestToolCallPresence:
    def test_response_has_tool_calls_true_for_function_call(self) -> None:
        response = MagicMock()
        response.output = [
            _fake_output_item("message"),
            _fake_output_item("function_call"),
        ]
        assert _response_has_tool_calls(response) is True

    def test_response_has_tool_calls_false_for_plain_message(self) -> None:
        response = MagicMock()
        response.output = [_fake_output_item("message")]
        assert _response_has_tool_calls(response) is False

    def test_response_has_tool_calls_false_for_empty_output(self) -> None:
        response = MagicMock()
        response.output = []
        assert _response_has_tool_calls(response) is False

    async def test_on_llm_end_sets_has_tool_calls_attribute(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("llm-tool-calls") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            await hook.on_agent_start(ctx, agent)
            await hook.on_llm_start(ctx, agent, system_prompt="sys", input_items=[])

            response = MagicMock()
            response.usage = None
            response.output = [_fake_output_item("function_call")]
            await hook.on_llm_end(ctx, agent, response)

            llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
            assert llm_span.attributes["llm.response.has_tool_calls"] is True


# ---------------------------------------------------------------------------
# on_tool_start / on_tool_end round-trip
# ---------------------------------------------------------------------------


class TestToolSpanLifecycle:
    async def test_tool_span_is_child_of_agent_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-child") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            tool = MagicMock()
            tool.name = "search"

            await hook.on_agent_start(ctx, agent)
            agent_span = next(s for s in trace.spans if "agent" in s.name)

            await hook.on_tool_start(ctx, agent, tool)
            tool_span = next(s for s in trace.spans if "tool" in s.name)

            assert tool_span.parent_id == agent_span.span_id

    async def test_on_tool_end_closes_span_ok(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-end") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            tool = MagicMock()
            tool.name = "calc"

            await hook.on_agent_start(ctx, agent)
            await hook.on_tool_start(ctx, agent, tool)
            await hook.on_tool_end(ctx, agent, tool, result="42")

            tool_span = next(s for s in trace.spans if "tool" in s.name)
            assert tool_span.status == SpanStatus.OK
            assert tool_span.end_time is not None
            assert not any("tool:" in k and ":calc" in k for k in hook._spans)

    async def test_on_tool_end_persists_result_content(self, tmp_path: Path) -> None:
        """Persist the actual result text, not just its length."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-result-content") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            tool = MagicMock()
            tool.name = "calc"

            await hook.on_agent_start(ctx, agent)
            await hook.on_tool_start(ctx, agent, tool)
            await hook.on_tool_end(ctx, agent, tool, result="the answer is 42")

            tool_span = next(s for s in trace.spans if "tool" in s.name)
            assert tool_span.attributes["tool.result"] == "the answer is 42"
            assert tool_span.attributes["tool.result_length"] == len("the answer is 42")

    async def test_on_tool_end_truncates_long_result(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-result-truncate") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            tool = MagicMock()
            tool.name = "big"
            long_result = "x" * 10_000

            await hook.on_agent_start(ctx, agent)
            await hook.on_tool_start(ctx, agent, tool)
            await hook.on_tool_end(ctx, agent, tool, result=long_result)

            tool_span = next(s for s in trace.spans if "tool" in s.name)
            assert len(tool_span.attributes["tool.result"]) < len(long_result)
            assert tool_span.attributes["tool.result_length"] == 10_000


# ---------------------------------------------------------------------------
# on_handoff — duration-based handoff spans
# ---------------------------------------------------------------------------


class TestHandoffSpans:
    async def test_handoff_event_recorded_on_outgoing_agent_span(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("handoff-event") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent_a = _fake_agent("agent-a")
            agent_b = _fake_agent("agent-b")

            await hook.on_agent_start(ctx, agent_a)
            await hook.on_handoff(ctx, agent_a, agent_b)

            agent_a_span = next(s for s in trace.spans if "agent-a" in s.name)
            assert any(e.name == "handoff" for e in agent_a_span.events)

    async def test_handoff_span_opens_on_agent_end_and_closes_on_next_start(
        self, tmp_path: Path
    ) -> None:
        """A duration-based handoff:<from>-><to> span must bound the transition."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("handoff-span") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent_a = _fake_agent("researcher")
            agent_b = _fake_agent("writer")

            await hook.on_agent_start(ctx, agent_a)
            await hook.on_handoff(ctx, agent_a, agent_b)
            await hook.on_agent_end(ctx, agent_a, output="handing off")

            handoff_span = next(
                s for s in trace.spans if s.name == "handoff:researcher->writer"
            )
            assert handoff_span.end_time is None  # still open, awaiting agent_b start

            await hook.on_agent_start(ctx, agent_b)

            assert handoff_span.end_time is not None
            assert handoff_span.status == SpanStatus.OK
            assert handoff_span.duration_ms is not None
            assert handoff_span.attributes["handoff.from_agent"] == "researcher"
            assert handoff_span.attributes["handoff.to_agent"] == "writer"

    async def test_no_handoff_span_without_a_handoff(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("no-handoff") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent("solo")
            await hook.on_agent_start(ctx, agent)
            await hook.on_agent_end(ctx, agent, output="done")
            assert not any(s.name.startswith("handoff:") for s in trace.spans)


# ---------------------------------------------------------------------------
# Exception-to-span attribution (instrument_runner / instrument_runner_streamed)
# ---------------------------------------------------------------------------


class TestExceptionAttribution:
    async def test_close_open_spans_with_exception_marks_error(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("exc-attribution") as trace:
            hook = AgentTraceHook(tracer=t, trace=trace)
            ctx = _fake_context()
            agent = _fake_agent()
            tool = MagicMock()
            tool.name = "flaky"

            await hook.on_agent_start(ctx, agent)
            await hook.on_tool_start(ctx, agent, tool)
            # Tool never ends (simulating a crash mid-call) — spans stay open.

            boom = RuntimeError("boom")
            hook._close_open_spans_with_exception(boom)

            agent_span = next(s for s in trace.spans if "agent" in s.name)
            tool_span = next(s for s in trace.spans if "tool" in s.name)
            for span in (agent_span, tool_span):
                assert span.status == SpanStatus.ERROR
                assert span.end_time is not None
                assert any(e.name == "exception" for e in span.events)
            assert hook._spans == {}

    async def test_instrument_runner_closes_open_spans_on_exception(
        self, tmp_path: Path
    ) -> None:
        """instrument_runner must close every open hook span when Runner.run raises."""
        t = Tracer(trace_dir=tmp_path)

        class _FakeRunner:
            @staticmethod
            async def run(
                agent: Any, input: Any, *, hooks: AgentTraceHook, **kwargs: Any
            ) -> Any:
                ctx = _fake_context()
                await hooks.on_agent_start(ctx, agent)
                raise RuntimeError("model provider 500")

        fake_sdk = MagicMock()
        fake_sdk.Runner = _FakeRunner

        import agent_trace.integrations.openai_agents as oa_mod

        original = oa_mod._require_openai_agents
        oa_mod._require_openai_agents = lambda: fake_sdk  # type: ignore[assignment]
        try:
            with t.start_trace("instrument-runner-exc") as trace:
                with pytest.raises(RuntimeError, match="model provider 500"):
                    await instrument_runner(_fake_agent(), "hi", tracer=t, trace=trace)

                agent_span = next(s for s in trace.spans if "agent" in s.name)
                assert agent_span.status == SpanStatus.ERROR
                root_span = next(s for s in trace.spans if s.name == "agent_run")
                assert root_span.status == SpanStatus.ERROR
        finally:
            oa_mod._require_openai_agents = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# instrument_runner_streamed — Runner.run_streamed()/stream_events() support
# ---------------------------------------------------------------------------


class TestInstrumentRunnerStreamed:
    async def test_stream_events_are_yielded_and_recorded(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        class _FakeEvent:
            def __init__(self, event_type: str) -> None:
                self.type = event_type

        class _FakeResultStreaming:
            final_output = "done"

            async def stream_events(self) -> Any:
                yield _FakeEvent("run_item_stream_event")
                yield _FakeEvent("agent_updated_stream_event")

        class _FakeRunner:
            @staticmethod
            def run_streamed(
                agent: Any, input: Any, *, hooks: Any, **kwargs: Any
            ) -> Any:
                return _FakeResultStreaming()

        fake_sdk = MagicMock()
        fake_sdk.Runner = _FakeRunner

        import agent_trace.integrations.openai_agents as oa_mod

        original = oa_mod._require_openai_agents
        oa_mod._require_openai_agents = lambda: fake_sdk  # type: ignore[assignment]
        try:
            with t.start_trace("streamed-basic") as trace:
                streamed = await instrument_runner_streamed(
                    _fake_agent(), "hi", tracer=t, trace=trace
                )
                events = [e async for e in streamed.stream_events()]
                assert [e.type for e in events] == [
                    "run_item_stream_event",
                    "agent_updated_stream_event",
                ]
                assert streamed.final_output == "done"  # proxied attribute access

                root_span = next(
                    s for s in trace.spans if s.name == "agent_run_streamed"
                )
                assert root_span.status == SpanStatus.OK
                assert len(root_span.events) == 2
        finally:
            oa_mod._require_openai_agents = original  # type: ignore[assignment]

    async def test_stream_events_exception_closes_open_spans(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)

        class _FakeResultStreaming:
            async def stream_events(self) -> Any:
                raise RuntimeError("max turns exceeded")
                yield  # pragma: no cover - unreachable, makes this a generator

        class _FakeRunner:
            @staticmethod
            def run_streamed(
                agent: Any, input: Any, *, hooks: Any, **kwargs: Any
            ) -> Any:
                return _FakeResultStreaming()

        fake_sdk = MagicMock()
        fake_sdk.Runner = _FakeRunner

        import agent_trace.integrations.openai_agents as oa_mod

        original = oa_mod._require_openai_agents
        oa_mod._require_openai_agents = lambda: fake_sdk  # type: ignore[assignment]
        try:
            with t.start_trace("streamed-exc") as trace:
                streamed = await instrument_runner_streamed(
                    _fake_agent(), "hi", tracer=t, trace=trace
                )
                with pytest.raises(RuntimeError, match="max turns exceeded"):
                    async for _ in streamed.stream_events():
                        pass

                root_span = next(
                    s for s in trace.spans if s.name == "agent_run_streamed"
                )
                assert root_span.status == SpanStatus.ERROR
        finally:
            oa_mod._require_openai_agents = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# AgentTraceRealtimeHook
# ---------------------------------------------------------------------------


class TestAgentTraceRealtimeHook:
    async def test_wrap_opens_agent_and_tool_spans_from_events(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)

        def _rt_event(event_type: str, **fields: Any) -> Any:
            e = MagicMock()
            e.type = event_type
            for k, v in fields.items():
                setattr(e, k, v)
            return e

        rt_agent = MagicMock()
        rt_agent.name = "voice-agent"
        rt_tool = MagicMock()
        rt_tool.name = "lookup"

        events = [
            _rt_event("agent_start", agent=rt_agent),
            _rt_event(
                "tool_start", agent=rt_agent, tool=rt_tool, arguments='{"q":"x"}'
            ),
            _rt_event("tool_end", agent=rt_agent, tool=rt_tool, output="found it"),
            _rt_event("agent_end", agent=rt_agent),
        ]

        class _FakeSession:
            def __aiter__(self) -> Any:
                return self._gen()

            async def _gen(self) -> Any:
                for e in events:
                    yield e

        with t.start_trace("realtime-basic") as trace:
            rt_hook = AgentTraceRealtimeHook(tracer=t, trace=trace)
            seen = [e async for e in rt_hook.wrap(_FakeSession())]
            assert len(seen) == 4

            agent_span = next(s for s in trace.spans if "voice-agent" in s.name)
            assert agent_span.status == SpanStatus.OK
            assert agent_span.end_time is not None

            tool_span = next(s for s in trace.spans if "lookup" in s.name)
            assert tool_span.parent_id == agent_span.span_id
            assert tool_span.attributes["tool.result"] == "found it"

            session_span = next(s for s in trace.spans if s.name == "realtime_session")
            assert session_span.status == SpanStatus.OK

    async def test_wrap_closes_open_spans_on_exception(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        rt_agent = MagicMock()
        rt_agent.name = "voice-agent"

        def _rt_event(event_type: str, **fields: Any) -> Any:
            e = MagicMock()
            e.type = event_type
            for k, v in fields.items():
                setattr(e, k, v)
            return e

        class _FakeSession:
            def __aiter__(self) -> Any:
                return self._gen()

            async def _gen(self) -> Any:
                yield _rt_event("agent_start", agent=rt_agent)
                raise RuntimeError("connection dropped")

        with t.start_trace("realtime-exc") as trace:
            rt_hook = AgentTraceRealtimeHook(tracer=t, trace=trace)
            with pytest.raises(RuntimeError, match="connection dropped"):
                async for _ in rt_hook.wrap(_FakeSession()):
                    pass

            agent_span = next(s for s in trace.spans if "voice-agent" in s.name)
            assert agent_span.status == SpanStatus.ERROR
            session_span = next(s for s in trace.spans if s.name == "realtime_session")
            assert session_span.status == SpanStatus.ERROR

    async def test_wrap_records_error_event_without_closing_session(
        self, tmp_path: Path
    ) -> None:
        """A realtime 'error' event is not necessarily fatal to the session."""
        t = Tracer(trace_dir=tmp_path)

        def _rt_event(event_type: str, **fields: Any) -> Any:
            e = MagicMock()
            e.type = event_type
            for k, v in fields.items():
                setattr(e, k, v)
            return e

        class _FakeSession:
            def __aiter__(self) -> Any:
                return self._gen()

            async def _gen(self) -> Any:
                yield _rt_event("error", error="transient glitch")

        with t.start_trace("realtime-error-event") as trace:
            rt_hook = AgentTraceRealtimeHook(tracer=t, trace=trace)
            seen = [e async for e in rt_hook.wrap(_FakeSession())]
            assert len(seen) == 1

            session_span = next(s for s in trace.spans if s.name == "realtime_session")
            assert session_span.status == SpanStatus.OK  # wrap() completed normally
            assert any(e.name == "exception" for e in session_span.events)

    async def test_tool_not_found_error_attaches_fuzzy_match(
        self, tmp_path: Path
    ) -> None:
        """#1671: an intermittent 'Tool X not found' ModelBehaviorError
        crash inside a Realtime/WebSocket voice session — the offending
        name should be fuzzy-matched against the tools registered on the
        session's most recently started agent (captured via `agent_start`),
        with the nearest match + edit distance attached to the exception
        event."""
        t = Tracer(trace_dir=tmp_path)

        def _rt_event(event_type: str, **fields: Any) -> Any:
            e = MagicMock()
            e.type = event_type
            for k, v in fields.items():
                setattr(e, k, v)
            return e

        weather_tool = MagicMock()
        weather_tool.name = "get_weather"
        search_tool = MagicMock()
        search_tool.name = "search"
        rt_agent = MagicMock()
        rt_agent.name = "voice-agent"
        rt_agent.tools = [weather_tool, search_tool]

        class _FakeSession:
            def __aiter__(self) -> Any:
                return self._gen()

            async def _gen(self) -> Any:
                yield _rt_event("agent_start", agent=rt_agent)
                # Model hallucinated a near-miss of a registered tool name.
                yield _rt_event("error", error="Tool get_wather not found")

        with t.start_trace("realtime-tool-not-found") as trace:
            rt_hook = AgentTraceRealtimeHook(tracer=t, trace=trace)
            seen = [e async for e in rt_hook.wrap(_FakeSession())]
            assert len(seen) == 2

            session_span = next(s for s in trace.spans if s.name == "realtime_session")
            exc_event = next(e for e in session_span.events if e.name == "exception")
            assert (
                exc_event.attributes["exception.nearest_registered_tool"]
                == "get_weather"
            )
            assert exc_event.attributes["exception.edit_distance"] == 1

    async def test_tool_not_found_error_without_registered_tools_no_attributes(
        self, tmp_path: Path
    ) -> None:
        """No agent_start event ever fired (or it carried no tools) — there
        is nothing to fuzzy-match against, so no diagnosis attributes are
        added; the base exception attributes are still recorded."""
        t = Tracer(trace_dir=tmp_path)

        def _rt_event(event_type: str, **fields: Any) -> Any:
            e = MagicMock()
            e.type = event_type
            for k, v in fields.items():
                setattr(e, k, v)
            return e

        class _FakeSession:
            def __aiter__(self) -> Any:
                return self._gen()

            async def _gen(self) -> Any:
                yield _rt_event("error", error="Tool mystery_tool not found")

        with t.start_trace("realtime-tool-not-found-no-agent") as trace:
            rt_hook = AgentTraceRealtimeHook(tracer=t, trace=trace)
            seen = [e async for e in rt_hook.wrap(_FakeSession())]
            assert len(seen) == 1

            session_span = next(s for s in trace.spans if s.name == "realtime_session")
            exc_event = next(e for e in session_span.events if e.name == "exception")
            assert "exception.nearest_registered_tool" not in exc_event.attributes
            assert "exception.edit_distance" not in exc_event.attributes

    async def test_error_not_matching_tool_not_found_pattern_no_attributes(
        self, tmp_path: Path
    ) -> None:
        """An 'error' event whose text doesn't match the 'Tool <name> not
        found' shape (e.g. a transient network glitch) must not add fuzzy-
        match attributes — the diagnosis is specific to that error shape."""
        t = Tracer(trace_dir=tmp_path)

        def _rt_event(event_type: str, **fields: Any) -> Any:
            e = MagicMock()
            e.type = event_type
            for k, v in fields.items():
                setattr(e, k, v)
            return e

        weather_tool = MagicMock()
        weather_tool.name = "get_weather"
        rt_agent = MagicMock()
        rt_agent.name = "voice-agent"
        rt_agent.tools = [weather_tool]

        class _FakeSession:
            def __aiter__(self) -> Any:
                return self._gen()

            async def _gen(self) -> Any:
                yield _rt_event("agent_start", agent=rt_agent)
                yield _rt_event("error", error="connection reset by peer")

        with t.start_trace("realtime-unrelated-error") as trace:
            rt_hook = AgentTraceRealtimeHook(tracer=t, trace=trace)
            seen = [e async for e in rt_hook.wrap(_FakeSession())]
            assert len(seen) == 2

            session_span = next(s for s in trace.spans if s.name == "realtime_session")
            exc_event = next(e for e in session_span.events if e.name == "exception")
            assert "exception.nearest_registered_tool" not in exc_event.attributes


# ---------------------------------------------------------------------------
# Dead code removal — on_agent_error/on_tool_error (unreachable via the SDK)
# ---------------------------------------------------------------------------


class TestDeadCodeRemoved:
    def test_enrich_step_span_does_not_exist(self) -> None:
        """_enrich_step_span was dead code — verify it's been removed."""
        import agent_trace.integrations.openai_agents as oa_mod

        assert not hasattr(oa_mod, "_enrich_step_span"), (
            "_enrich_step_span is dead code (never called); it should have been removed"
        )

    def test_on_agent_error_and_on_tool_error_removed(self) -> None:
        """The current openai-agents SDK has no on_agent_error/on_tool_error hook —
        these methods were unreachable dead code; exception attribution now
        happens via instrument_runner's try/except instead (see
        AgentTraceHook._close_open_spans_with_exception)."""
        assert not hasattr(AgentTraceHook, "on_agent_error")
        assert not hasattr(AgentTraceHook, "on_tool_error")
