"""
Unit tests for agent_trace.integrations.agno.AgnoTracer.

The agno package is NOT imported anywhere in these tests — AgnoTracer.process_event
duck-types on plain attribute access, so it is exercised here with lightweight
SimpleNamespace stand-ins that mimic the real event dataclass fields (verified
against the actual installed agno==2.7.1 package in
tests/integration/test_agno.py). This mirrors how test_integrations_langgraph.py
tests LangGraphTracer without requiring a real langchain_core installation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.agno import (
    AgnoTracer,
    _actor,
    _make_exception,
    _normalize_event_name,
    instrument_agent_arun,
    instrument_agent_run,
)

# ---------------------------------------------------------------------------
# Helpers — build fake event objects matching real agno dataclass field names
# ---------------------------------------------------------------------------


def _event(event_name: str, **fields: Any) -> SimpleNamespace:
    """Build a duck-typed stand-in for an Agno RunOutputEvent/TeamRunOutputEvent."""
    return SimpleNamespace(event=event_name, **fields)


def _make_tracer(tmp_path: Path) -> tuple[Tracer, Any, AgnoTracer]:
    """Return (tracer, trace, hook) with the trace left active for the test.

    Tracer.start_span() only appends a span to whichever trace is the
    *active* one on the ContextVar at call time — not necessarily the
    `trace` object a caller happens to be holding a reference to. Returning
    from inside a `with t.start_trace(...):` block would pop that ContextVar
    on the way out, silently discarding every span opened after this helper
    returns, so the context manager is entered manually instead.

    That alone isn't enough, though: `t.start_trace(...)` returns an
    anonymous ``@contextmanager``-wrapped generator object. If nothing keeps
    a reference to *that object* (only to the value its `__enter__()`
    yielded), CPython garbage-collects it immediately — and collecting an
    un-exited generator sends it a ``GeneratorExit``, which runs its
    ``finally: self._active_trace_var.reset(token)`` right then, silently
    deactivating the trace before the caller ever gets to use it. Stashing
    the context-manager object on `t` keeps it alive for as long as `t` is
    referenced (i.e. for the rest of the test).
    """
    t = Tracer(trace_dir=tmp_path)
    cm = t.start_trace("agno-unit")
    trace = cm.__enter__()
    t._test_keepalive_cm = cm  # type: ignore[attr-defined]  # see docstring
    hook = AgnoTracer(tracer=t, trace=trace)
    return t, trace, hook


def _tool_execution(**fields: Any) -> SimpleNamespace:
    defaults = {
        "tool_call_id": "call_1",
        "tool_name": "search",
        "tool_args": {"q": "hi"},
        "result": None,
        "child_run_id": None,
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _normalize_event_name / _actor / _make_exception — pure helpers
# ---------------------------------------------------------------------------


class TestNormalizeEventName:
    def test_agent_event_name_unchanged(self) -> None:
        assert _normalize_event_name("RunStarted") == "RunStarted"

    def test_team_event_name_strips_prefix(self) -> None:
        assert _normalize_event_name("TeamRunStarted") == "RunStarted"

    def test_team_tool_call_started_strips_prefix(self) -> None:
        assert _normalize_event_name("TeamToolCallStarted") == "ToolCallStarted"


class TestActor:
    def test_agent_event_returns_agent_kind(self) -> None:
        event = _event("RunStarted", agent_id="a1", agent_name="my-agent")
        assert _actor(event) == ("agent", "my-agent")

    def test_agent_event_falls_back_to_agent_id(self) -> None:
        event = _event("RunStarted", agent_id="a1", agent_name=None)
        assert _actor(event) == ("agent", "a1")

    def test_team_event_returns_team_kind(self) -> None:
        event = _event("TeamRunStarted", team_id="t1", team_name="my-team")
        assert _actor(event) == ("team", "my-team")


class TestMakeException:
    def test_uses_given_type_name(self) -> None:
        exc = _make_exception("boom", "UnboundLocalError")
        assert type(exc).__name__ == "UnboundLocalError"
        assert str(exc) == "boom"

    def test_falls_back_when_type_name_missing(self) -> None:
        exc = _make_exception("boom", None)
        assert type(exc).__name__ == "AgnoRunError"

    def test_sanitizes_unsafe_characters(self) -> None:
        exc = _make_exception("boom", "Some Weird.Type!")
        assert type(exc).__name__ == "SomeWeirdType"


# ---------------------------------------------------------------------------
# Run lifecycle: RunStarted -> RunCompleted
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_run_started_opens_span(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(
            _event("RunStarted", run_id="r1", agent_id="a1", agent_name="my-agent", model="gpt-4o")
        )
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "agent:my-agent"
        assert trace.spans[0].attributes["agno.run_id"] == "r1"
        assert trace.spans[0].attributes["agno.model"] == "gpt-4o"

    def test_run_completed_closes_span_ok(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="my-agent"))
        hook.process_event(_event("RunCompleted", run_id="r1", agent_id="a1", agent_name="my-agent"))
        assert trace.spans[0].status == SpanStatus.OK
        assert trace.spans[0].end_time is not None
        assert hook._run_spans == {}

    def test_team_run_uses_team_prefix(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(
            _event("TeamRunStarted", run_id="r1", team_id="t1", team_name="my-team")
        )
        assert trace.spans[0].name == "team:my-team"

    def test_nested_run_parents_to_parent_run_id(self, tmp_path: Path) -> None:
        """A member Agent run inside a Team must parent to the team's run span."""
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(
            _event("TeamRunStarted", run_id="team-run", team_id="t1", team_name="my-team")
        )
        hook.process_event(
            _event(
                "RunStarted",
                run_id="member-run",
                parent_run_id="team-run",
                agent_id="a1",
                agent_name="member-agent",
            )
        )
        team_span = next(s for s in trace.spans if s.name == "team:my-team")
        member_span = next(s for s in trace.spans if s.name == "agent:member-agent")
        assert member_span.parent_id == team_span.span_id


class TestRunError:
    def test_run_error_records_exception_and_closes_error(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="crash-agent"))
        hook.process_event(
            _event(
                "RunError",
                run_id="r1",
                agent_id="a1",
                agent_name="crash-agent",
                content="boom: UnboundLocalError",
                error_type=None,
            )
        )
        span = trace.spans[0]
        assert span.status == SpanStatus.ERROR
        assert span.end_time is not None
        assert span.events[0].attributes["exception.message"] == "boom: UnboundLocalError"
        assert hook._run_spans == {}

    def test_run_error_without_matching_start_does_not_raise(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        # No RunStarted fired first -- must not raise.
        hook.process_event(
            _event("RunError", run_id="unknown", agent_id="a1", content="boom")
        )
        assert trace.spans == []


class TestRunCancelled:
    def test_run_cancelled_closes_ok_with_attribute(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(
            _event("RunCancelled", run_id="r1", agent_id="a1", reason="user requested")
        )
        span = trace.spans[0]
        assert span.status == SpanStatus.OK
        assert span.attributes["agno.cancelled"] is True
        assert span.attributes["agno.cancel_reason"] == "user requested"


# ---------------------------------------------------------------------------
# Model-request lifecycle
# ---------------------------------------------------------------------------


class TestModelRequestLifecycle:
    def test_model_request_span_is_child_of_run_span(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(
            _event("ModelRequestStarted", run_id="r1", agent_id="a1", model="gpt-4o", model_provider="openai")
        )
        run_span = next(s for s in trace.spans if s.name == "agent:a")
        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.parent_id == run_span.span_id
        assert llm_span.attributes["llm.provider"] == "openai"

    def test_model_request_completed_records_token_usage(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(_event("ModelRequestStarted", run_id="r1", agent_id="a1", model="gpt-4o"))
        hook.process_event(
            _event(
                "ModelRequestCompleted",
                run_id="r1",
                agent_id="a1",
                model="gpt-4o",
                input_tokens=7,
                output_tokens=13,
                total_tokens=20,
                time_to_first_token=0.5,
            )
        )
        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.attributes["llm.usage.prompt_tokens"] == 7
        assert llm_span.attributes["llm.usage.completion_tokens"] == 13
        assert llm_span.attributes["llm.usage.total_tokens"] == 20
        assert llm_span.attributes["llm.time_to_first_token_s"] == 0.5
        assert llm_span.status == SpanStatus.OK
        assert llm_span.end_time is not None

    def test_sequential_model_requests_use_stack_not_dict_collision(
        self, tmp_path: Path
    ) -> None:
        """Two model calls within one run (a tool-calling loop) must each
        close independently via a stack, not overwrite each other."""
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))

        hook.process_event(_event("ModelRequestStarted", run_id="r1", agent_id="a1", model="gpt-4o"))
        hook.process_event(
            _event("ModelRequestCompleted", run_id="r1", agent_id="a1", model="gpt-4o", input_tokens=1)
        )
        hook.process_event(_event("ModelRequestStarted", run_id="r1", agent_id="a1", model="gpt-4o"))
        hook.process_event(
            _event("ModelRequestCompleted", run_id="r1", agent_id="a1", model="gpt-4o", input_tokens=2)
        )

        llm_spans = [s for s in trace.spans if s.name.startswith("llm:")]
        assert len(llm_spans) == 2
        assert all(s.end_time is not None for s in llm_spans)
        assert {s.attributes["llm.usage.prompt_tokens"] for s in llm_spans} == {1, 2}
        assert hook._llm_stacks.get("r1") in (None, [])


# ---------------------------------------------------------------------------
# Tool call lifecycle
# ---------------------------------------------------------------------------


class TestToolCallLifecycle:
    def test_tool_call_started_creates_child_span(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(
            _event(
                "ToolCallStarted",
                run_id="r1",
                agent_id="a1",
                tool=_tool_execution(tool_call_id="call_1", tool_name="calculator"),
            )
        )
        run_span = next(s for s in trace.spans if s.name == "agent:a")
        tool_span = next(s for s in trace.spans if s.name == "tool:calculator")
        assert tool_span.parent_id == run_span.span_id
        assert tool_span.attributes["tool.name"] == "calculator"

    def test_tool_call_completed_closes_ok(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(
            _event(
                "ToolCallStarted",
                run_id="r1",
                agent_id="a1",
                tool=_tool_execution(tool_call_id="call_1", tool_name="calculator"),
            )
        )
        hook.process_event(
            _event(
                "ToolCallCompleted",
                run_id="r1",
                agent_id="a1",
                tool=_tool_execution(tool_call_id="call_1", tool_name="calculator", result="4"),
            )
        )
        tool_span = next(s for s in trace.spans if s.name == "tool:calculator")
        assert tool_span.status == SpanStatus.OK
        assert tool_span.attributes["tool.result_length"] == 1
        assert hook._tool_spans == {}

    def test_tool_call_completed_records_child_run_id(self, tmp_path: Path) -> None:
        """A delegate-to-member tool call carries child_run_id — the mechanism
        that lets a developer correlate the delegation with the member's own
        run span (per-team-member attribution, matching upstream issue #5326)."""
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("TeamRunStarted", run_id="team-r", team_id="t1", team_name="my-team"))
        hook.process_event(
            _event(
                "ToolCallStarted",
                run_id="team-r",
                team_id="t1",
                tool=_tool_execution(tool_call_id="call_1", tool_name="delegate_task_to_member"),
            )
        )
        hook.process_event(
            _event(
                "ToolCallCompleted",
                run_id="team-r",
                team_id="t1",
                tool=_tool_execution(
                    tool_call_id="call_1",
                    tool_name="delegate_task_to_member",
                    result="member says hi",
                    child_run_id="member-run-id",
                ),
            )
        )
        tool_span = next(s for s in trace.spans if s.name == "tool:delegate_task_to_member")
        assert tool_span.attributes["agno.child_run_id"] == "member-run-id"

    def test_tool_call_error_records_exception(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(
            _event(
                "ToolCallStarted",
                run_id="r1",
                agent_id="a1",
                tool=_tool_execution(tool_call_id="call_1", tool_name="flaky"),
            )
        )
        hook.process_event(
            _event(
                "ToolCallError",
                run_id="r1",
                agent_id="a1",
                tool=_tool_execution(tool_call_id="call_1", tool_name="flaky"),
                error="connection refused",
            )
        )
        tool_span = next(s for s in trace.spans if s.name == "tool:flaky")
        assert tool_span.status == SpanStatus.ERROR
        assert tool_span.events[0].attributes["exception.message"] == "connection refused"
        assert hook._tool_spans == {}


# ---------------------------------------------------------------------------
# process_event must never raise on malformed input
# ---------------------------------------------------------------------------


class TestProcessEventIsResilient:
    def test_unknown_event_name_is_ignored(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("SomeFutureEvent", run_id="r1"))
        assert trace.spans == []

    def test_event_missing_run_id_does_not_raise(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(SimpleNamespace(event="RunCompleted"))  # no run_id at all
        assert trace.spans == []

    def test_completely_malformed_event_does_not_raise(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(object())  # no .event attribute at all
        # Falls back to type(event).__name__ == "object", which matches nothing.
        assert trace.spans == []


# ---------------------------------------------------------------------------
# close_open_spans — safety net
# ---------------------------------------------------------------------------


class TestCloseOpenSpans:
    def test_close_open_spans_force_closes_everything(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(_event("ModelRequestStarted", run_id="r1", agent_id="a1", model="gpt-4o"))
        hook.process_event(
            _event(
                "ToolCallStarted",
                run_id="r1",
                agent_id="a1",
                tool=_tool_execution(tool_call_id="call_1", tool_name="calc"),
            )
        )
        assert any(s.end_time is None for s in trace.spans)

        hook.close_open_spans()

        assert all(s.end_time is not None for s in trace.spans)
        assert all(s.status == SpanStatus.ERROR for s in trace.spans)
        assert hook._run_spans == {}
        assert hook._tool_spans == {}
        assert hook._llm_stacks == {}

    def test_close_open_spans_is_noop_when_nothing_open(self, tmp_path: Path) -> None:
        _, trace, hook = _make_tracer(tmp_path)
        hook.process_event(_event("RunStarted", run_id="r1", agent_id="a1", agent_name="a"))
        hook.process_event(_event("RunCompleted", run_id="r1", agent_id="a1"))
        hook.close_open_spans()  # must not raise, must not touch the closed span
        assert trace.spans[0].status == SpanStatus.OK


# ---------------------------------------------------------------------------
# _require_agno / ImportError surface
# ---------------------------------------------------------------------------


class TestRequireAgno:
    def test_instrument_agent_run_raises_clear_error_without_agno(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "agno", None)  # simulate "not installed"
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("no-agno") as trace, pytest.raises(ImportError, match="pip install agno"):
            instrument_agent_run(object(), "hi", tracer=t, trace=trace)

    async def test_instrument_agent_arun_raises_clear_error_without_agno(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "agno", None)
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("no-agno-async") as trace:
            with pytest.raises(ImportError, match="pip install agno"):
                await instrument_agent_arun(object(), "hi", tracer=t, trace=trace)
