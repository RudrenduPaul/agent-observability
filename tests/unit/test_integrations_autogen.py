"""
Unit tests for agent_trace.integrations.autogen.

The autogen module itself never imports autogen-agentchat/autogen-ext (it
works entirely by duck-typed instance-attribute wrapping), so these tests
use lightweight fake agent/team/executor/event classes instead of the real
packages -- they run in CI with zero AutoGen install.  See
tests/integration/test_autogen.py for tests against the real installed
autogen-agentchat/autogen-ext package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_trace import SpanStatus, Tracer
from agent_trace._replay.fixture import Fixture
from agent_trace.integrations.autogen import (
    instrument_agent,
    instrument_code_executor,
    instrument_team,
    recording_http_client,
)

# ---------------------------------------------------------------------------
# Fake AutoGen-shaped objects
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    """Stand-in for BaseChatMessage/BaseAgentEvent — carries models_usage."""

    def __init__(
        self, source: str = "agent", models_usage: _FakeUsage | None = None
    ) -> None:
        self.source = source
        self.models_usage = models_usage


def _named(cls_name: str, **fields: Any) -> Any:
    """Build an instance of a dynamically-named class (matches AutoGen's
    real event type names, which _record_event() dispatches on)."""
    cls = type(cls_name, (), {})
    obj = cls()
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj


class _FakeFunctionCall:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeFunctionExecutionResult:
    def __init__(self, name: str, is_error: bool = False) -> None:
        self.name = name
        self.is_error = is_error


class _FakeAgent:
    """Stand-in for autogen_agentchat.agents.BaseChatAgent."""

    def __init__(self, name: str, events: list[Any] | None = None) -> None:
        self.name = name
        self._events = events if events is not None else []
        self._raise: BaseException | None = None

    def raise_after_events(self, exc: BaseException) -> None:
        self._raise = exc

    async def on_messages_stream(self, messages: Any, cancellation_token: Any) -> Any:
        for event in self._events:
            yield event
        if self._raise is not None:
            raise self._raise


class _FakeTeam:
    """Stand-in for autogen_agentchat.teams.BaseGroupChat."""

    def __init__(
        self,
        participants: list[Any],
        events: list[Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._participants = participants
        self._events = events if events is not None else []
        self.name = name or "RoundRobinGroupChat"

    async def run_stream(self, *args: Any, **kwargs: Any) -> Any:
        for event in self._events:
            yield event


class _FakeCodeBlock:
    def __init__(self, code: str, language: str = "python") -> None:
        self.code = code
        self.language = language


class _FakeCodeResult:
    def __init__(self, exit_code: int, output: str) -> None:
        self.exit_code = exit_code
        self.output = output


class _FakeCodeExecutor:
    """Stand-in for autogen_core.code_executor.CodeExecutor."""

    def __init__(self, work_dir: str, result: _FakeCodeResult | None = None) -> None:
        self.work_dir = work_dir
        self._result = result or _FakeCodeResult(0, "ok")
        self._raise: BaseException | None = None

    def raise_on_execute(self, exc: BaseException) -> None:
        self._raise = exc

    async def execute_code_blocks(
        self, code_blocks: Any, cancellation_token: Any
    ) -> Any:
        if self._raise is not None:
            raise self._raise
        return self._result


# ---------------------------------------------------------------------------
# instrument_agent
# ---------------------------------------------------------------------------


class TestInstrumentAgent:
    def test_creates_agent_span_tagged_with_name(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-agent") as trace:
            agent = _FakeAgent("planner")
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                async for _ in agent.on_messages_stream([], None):
                    pass

            asyncio.run(run())

            assert len(trace.spans) == 1
            span = trace.spans[0]
            assert span.name == "agent:planner"
            assert span.attributes["agent.name"] == "planner"
            assert span.status == SpanStatus.OK
            assert span.end_time is not None

    def test_records_exception_and_reraises(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-agent-error") as trace:
            agent = _FakeAgent("worker")
            agent.raise_after_events(TypeError("Cannot instantiate typing.Union"))
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                async for _ in agent.on_messages_stream([], None):
                    pass

            with pytest.raises(TypeError, match="Cannot instantiate"):
                asyncio.run(run())

            span = trace.spans[0]
            assert span.status == SpanStatus.ERROR
            assert span.end_time is not None
            exc_events = [e for e in span.events if e.name == "exception"]
            assert len(exc_events) == 1
            assert exc_events[0].attributes["exception.type"] == "TypeError"

    def test_accumulates_token_usage_across_turn(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-usage") as trace:
            events = [
                _FakeMessage(models_usage=_FakeUsage(10, 5)),
                _FakeMessage(models_usage=_FakeUsage(15, 6)),
                _FakeMessage(models_usage=None),
            ]
            agent = _FakeAgent("worker", events=events)
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                async for _ in agent.on_messages_stream([], None):
                    pass

            asyncio.run(run())

            span = trace.spans[0]
            assert span.attributes["llm.usage.prompt_tokens"] == 25
            assert span.attributes["llm.usage.completion_tokens"] == 11
            assert span.attributes["llm.usage.total_tokens"] == 36

    def test_records_tool_call_request_and_execution_events(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-tools") as trace:
            events = [
                _named(
                    "ToolCallRequestEvent",
                    source="worker",
                    content=[_FakeFunctionCall("search")],
                ),
                _named(
                    "ToolCallExecutionEvent",
                    source="worker",
                    content=[_FakeFunctionExecutionResult("search")],
                ),
            ]
            agent = _FakeAgent("worker", events=events)
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                async for _ in agent.on_messages_stream([], None):
                    pass

            asyncio.run(run())

            span = trace.spans[0]
            event_names = [e.name for e in span.events]
            assert "tool_call_request" in event_names
            assert "tool_call_execution" in event_names
            req = next(e for e in span.events if e.name == "tool_call_request")
            assert req.attributes["tool.names"] == "search"

    def test_records_usage_nested_inside_response_chat_message(
        self, tmp_path: Path
    ) -> None:
        """Response itself never carries models_usage -- only its nested
        chat_message does.  instrument_agent() must look inside it."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-response-usage") as trace:
            chat_message = _FakeMessage(models_usage=_FakeUsage(20, 4))
            response_event = _named("Response", source=None, chat_message=chat_message)
            agent = _FakeAgent("worker", events=[response_event])
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                async for _ in agent.on_messages_stream([], None):
                    pass

            asyncio.run(run())

            span = trace.spans[0]
            assert span.attributes["llm.usage.prompt_tokens"] == 20
            assert span.attributes["llm.usage.completion_tokens"] == 4

    def test_records_handoff_event(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-handoff") as trace:
            events = [
                _named("HandoffMessage", source="triage", target="billing_agent"),
            ]
            agent = _FakeAgent("triage", events=events)
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                async for _ in agent.on_messages_stream([], None):
                    pass

            asyncio.run(run())

            span = trace.spans[0]
            handoff = next(e for e in span.events if e.name == "handoff")
            assert handoff.attributes["handoff.target"] == "billing_agent"
            assert handoff.attributes["handoff.from_agent"] == "triage"

    def test_idempotent_double_instrumentation(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-idempotent") as trace:
            agent = _FakeAgent("worker")
            instrument_agent(agent, tracer=t, trace=trace)
            wrapped_once = agent.on_messages_stream
            instrument_agent(agent, tracer=t, trace=trace)
            assert agent.on_messages_stream is wrapped_once

    def test_span_closes_ok_when_caller_stops_iterating_after_response(
        self, tmp_path: Path
    ) -> None:
        """Regression test: BaseChatAgent.on_messages() stops calling
        anext() as soon as it sees a Response and never resumes the
        generator, so relying on generator exhaustion (or a GeneratorExit
        thrown at GC time, which asyncio schedules asynchronously rather
        than running synchronously) to close the span would leave it open
        (status UNSET, end_time None) indefinitely.  instrument_agent()
        must close the span synchronously as soon as a Response is yielded,
        without requiring the caller to keep iterating."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-early-stop") as trace:
            response_event = _named("Response", source=None)
            trailing_event = _named(
                "ThoughtEvent", source="worker", content="unreachable"
            )
            agent = _FakeAgent("worker", events=[response_event, trailing_event])
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run_like_on_messages() -> Any:
                async for event in agent.on_messages_stream([], None):
                    if type(event).__name__ == "Response":
                        return event
                raise AssertionError("stream should have yielded a Response")

            asyncio.run(run_like_on_messages())

            span = trace.spans[0]
            assert span.end_time is not None
            assert span.status == SpanStatus.OK

    def test_malformed_event_does_not_crash_the_turn(self, tmp_path: Path) -> None:
        """An event missing expected fields must not abort the agent's turn."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-malformed") as trace:
            events = [_named("ToolCallRequestEvent", source="worker", content=None)]
            agent = _FakeAgent("worker", events=events)
            instrument_agent(agent, tracer=t, trace=trace)

            import asyncio

            async def run() -> list[Any]:
                return [e async for e in agent.on_messages_stream([], None)]

            result = asyncio.run(run())
            assert len(result) == 1
            assert trace.spans[0].status == SpanStatus.OK


# ---------------------------------------------------------------------------
# instrument_team
# ---------------------------------------------------------------------------


class TestInstrumentTeam:
    def test_instruments_every_participant(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-team") as trace:
            a = _FakeAgent("planner")
            b = _FakeAgent("writer")
            team = _FakeTeam([a, b])
            instrument_team(team, tracer=t, trace=trace)

            assert getattr(a, "_agent_trace_instrumented", False) is True
            assert getattr(b, "_agent_trace_instrumented", False) is True

    def test_creates_team_root_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-team-span") as trace:
            a = _FakeAgent("planner")
            team = _FakeTeam([a], name="MyTeam")
            instrument_team(team, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                async for _ in team.run_stream():
                    pass

            asyncio.run(run())

            team_spans = [s for s in trace.spans if s.name == "team:MyTeam"]
            assert len(team_spans) == 1
            assert team_spans[0].status == SpanStatus.OK

    def test_recurses_into_nested_team(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-nested-team") as trace:
            leaf_agent = _FakeAgent("leaf")
            inner_team = _FakeTeam([leaf_agent], name="Inner")
            outer_team = _FakeTeam([inner_team], name="Outer")
            instrument_team(outer_team, tracer=t, trace=trace)

            assert getattr(inner_team, "_agent_trace_instrumented", False) is True
            assert getattr(leaf_agent, "_agent_trace_instrumented", False) is True

    def test_idempotent_double_instrumentation(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-team-idempotent") as trace:
            team = _FakeTeam([_FakeAgent("solo")])
            instrument_team(team, tracer=t, trace=trace)
            wrapped_once = team.run_stream
            instrument_team(team, tracer=t, trace=trace)
            assert team.run_stream is wrapped_once


# ---------------------------------------------------------------------------
# instrument_code_executor
# ---------------------------------------------------------------------------


class TestInstrumentCodeExecutor:
    def test_records_successful_execution(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-code-ok") as trace:
            executor = _FakeCodeExecutor(
                "/tmp/coding", _FakeCodeResult(0, "hello from code\n")
            )
            instrument_code_executor(executor, tracer=t, trace=trace)

            import asyncio

            async def run() -> Any:
                return await executor.execute_code_blocks(
                    [_FakeCodeBlock("print('hello from code')")], None
                )

            result = asyncio.run(run())
            assert result.output == "hello from code\n"

            span = trace.spans[0]
            assert span.name == "code_execution"
            assert span.status == SpanStatus.OK
            assert span.attributes["code_execution.exit_code"] == 0
            assert span.attributes["code_execution.output"] == "hello from code\n"
            assert span.attributes["code_execution.work_dir"] == "/tmp/coding"
            assert "print('hello from code')" in span.attributes["code_execution.code"]

    def test_nonzero_exit_code_marks_span_error(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-code-fail") as trace:
            executor = _FakeCodeExecutor(
                "/tmp/coding", _FakeCodeResult(1, "Traceback...\nZeroDivisionError")
            )
            instrument_code_executor(executor, tracer=t, trace=trace)

            import asyncio

            async def run() -> Any:
                return await executor.execute_code_blocks([_FakeCodeBlock("1/0")], None)

            asyncio.run(run())

            span = trace.spans[0]
            assert span.status == SpanStatus.ERROR
            assert span.attributes["code_execution.exit_code"] == 1

    def test_raises_before_result_records_exception(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-code-raise") as trace:
            executor = _FakeCodeExecutor("/tmp/coding")
            executor.raise_on_execute(RuntimeError("docker not available"))
            instrument_code_executor(executor, tracer=t, trace=trace)

            import asyncio

            async def run() -> None:
                await executor.execute_code_blocks([_FakeCodeBlock("print(1)")], None)

            with pytest.raises(RuntimeError, match="docker not available"):
                asyncio.run(run())

            span = trace.spans[0]
            assert span.status == SpanStatus.ERROR
            exc_events = [e for e in span.events if e.name == "exception"]
            assert len(exc_events) == 1

    def test_idempotent_double_instrumentation(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-code-idempotent") as trace:
            executor = _FakeCodeExecutor("/tmp/coding")
            instrument_code_executor(executor, tracer=t, trace=trace)
            wrapped_once = executor.execute_code_blocks
            instrument_code_executor(executor, tracer=t, trace=trace)
            assert executor.execute_code_blocks is wrapped_once


# ---------------------------------------------------------------------------
# recording_http_client
# ---------------------------------------------------------------------------


class TestRecordingHttpClient:
    def test_async_client_wired_with_async_recording_transport(
        self, tmp_path: Path
    ) -> None:
        import httpx

        from agent_trace.interceptor.httpx_hook import AsyncRecordingTransport

        fixture = Fixture(tmp_path / "fixture.db")
        try:
            client = recording_http_client(fixture, is_async=True)
            assert isinstance(client, httpx.AsyncClient)
            assert isinstance(client._transport, AsyncRecordingTransport)
        finally:
            fixture.close()

    def test_sync_client_wired_with_recording_transport(self, tmp_path: Path) -> None:
        import httpx

        from agent_trace.interceptor.httpx_hook import RecordingTransport

        fixture = Fixture(tmp_path / "fixture.db")
        try:
            client = recording_http_client(fixture, is_async=False)
            assert isinstance(client, httpx.Client)
            assert isinstance(client._transport, RecordingTransport)
        finally:
            fixture.close()
