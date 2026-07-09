"""
Agno framework integration — Agent/Team run and model-stream lifecycle.

Agno (``agno-agi/agno``) has no LangChain-style callback-handler interface to
subclass.  Its officially supported observation surface is instead a
structured event stream: ``Agent.run``/``Agent.arun`` (and the identical
``Team.run``/``Team.arun``) accept ``stream=True, stream_events=True`` and
yield a sequence of ``RunOutputEvent``/``TeamRunOutputEvent`` objects —
``RunStarted`` -> ``ModelRequestStarted``/``ModelRequestCompleted`` ->
``ToolCallStarted``/``ToolCallCompleted``/``ToolCallError`` ->
``RunCompleted``/``RunError``/``RunCancelled`` (confirmed against the
installed ``agno==2.7.1`` package: ``agno/run/agent.py``,
``agno/run/team.py``, ``agno/agent/_run.py``).

Two properties of this event stream make it the right integration point
(rather than monkey-patching ``Model.aresponse``/``aresponse_stream``
directly, which is unexported, private implementation surface):

* Exceptions raised entirely inside Agno's own in-process response handling
  (never reaching the wire — e.g. a bug in ``agno/models/base.py`` itself)
  are caught by Agno's own streaming loop and re-surfaced as a
  ``RunErrorEvent`` on this same stream, not merely as an HTTP error the
  framework-agnostic interceptor could catch.
* When a ``Team`` delegates a task to a member ``Agent`` (via the built-in
  ``delegate_task_to_member`` tool), the member's own run events are
  forwarded through the team's event stream with ``parent_run_id`` set to
  the team run's ``run_id`` and the member's own ``agent_id``/``agent_name``
  populated — giving per-team-member attribution for free (confirmed via
  ``agno/team/_default_tools.py``'s ``delegate_task_to_member``).

Usage (hook-based — recommended)::

    from agno.agent import Agent
    from agent_trace import Tracer
    from agent_trace.integrations.agno import AgnoTracer

    t = Tracer()
    with t.start_trace("my_agno_run", record=True) as trace:
        hook = AgnoTracer(tracer=t, trace=trace)
        async for event in agent.arun("hello", stream=True, stream_events=True):
            hook.process_event(event)

Usage (convenience wrapper)::

    from agent_trace.integrations.agno import instrument_agent_arun

    result = await instrument_agent_arun(agent, "hello", tracer=t, trace=trace)

The same ``AgnoTracer``/wrappers work for a ``Team`` instance unchanged —
Agno's ``Team.run``/``Team.arun`` share the identical ``stream_events``
contract and event shape (``agno/team/team.py`` mirrors
``agno/agent/agent.py``'s ``run``/``arun`` overloads).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = [
    "AgnoTracer",
    "instrument_agent_arun",
    "instrument_agent_run",
]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The Agno integration requires the agno package.\n"
    "Install it with:\n\n"
    "    pip install agno\n"
)


def _require_agno() -> Any:
    """Lazy import guard — raises a clear error if agno is absent."""
    try:
        import agno

        return agno
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------
#
# agno.run.agent.RunEvent and agno.run.team.TeamRunEvent enumerate the same
# lifecycle in parallel: every Team event name is exactly "Team" + the
# corresponding Agent event name (e.g. "RunStarted" / "TeamRunStarted",
# "ToolCallStarted" / "TeamToolCallStarted" — confirmed against
# agno/run/agent.py:143-192 and agno/run/team.py:130-187). Stripping a
# leading "Team" prefix lets one classifier handle both without importing
# agno.run.agent / agno.run.team directly, keeping this module import-light.

_RUN_STARTED = "RunStarted"
_RUN_COMPLETED = "RunCompleted"
_RUN_ERROR = "RunError"
_RUN_CANCELLED = "RunCancelled"
_TOOL_CALL_STARTED = "ToolCallStarted"
_TOOL_CALL_COMPLETED = "ToolCallCompleted"
_TOOL_CALL_ERROR = "ToolCallError"
_MODEL_REQUEST_STARTED = "ModelRequestStarted"
_MODEL_REQUEST_COMPLETED = "ModelRequestCompleted"


def _normalize_event_name(name: str) -> str:
    """Strip Team's "Team" prefix so Agent- and Team-level events compare equal."""
    return name[len("Team") :] if name.startswith("Team") else name


def _actor(event: Any) -> tuple[str, str]:
    """Return ``(kind, display_name)`` for the event's originating Agent or Team.

    ``kind`` is ``"agent"`` or ``"team"`` — used as the span-name prefix so a
    team leader's run and a delegated member's run are both visible and
    distinguishable in the span tree (e.g. ``team:my-team`` parenting
    ``agent:researcher``).
    """
    agent_name = getattr(event, "agent_name", None) or getattr(event, "agent_id", None)
    if agent_name:
        return "agent", str(agent_name)
    team_name = getattr(event, "team_name", None) or getattr(event, "team_id", None)
    return "team", str(team_name or "team")


def _make_exception(message: str, type_name: str | None) -> Exception:
    """Synthesize an exception carrying Agno's reported error type/message.

    Agno's ``RunErrorEvent``/``ToolCallErrorEvent`` carry only strings — by
    the time the exception reaches this hook, Agno's own streaming loop has
    already caught it and converted it to event data (see
    ``agno/agent/_run.py``'s ``except Exception as e`` handler around the
    streaming tool-call loop, which calls ``create_run_error_event(...,
    error=str(e))``). There is no real ``BaseException`` object to forward,
    so build a lightweight one so ``Span.record_exception`` still populates
    ``exception.type``/``exception.message`` correctly.
    """
    cleaned = "".join(ch for ch in (type_name or "") if ch.isalnum() or ch == "_")
    safe_name = cleaned or "AgnoRunError"
    exc_cls: type[Exception] = type(safe_name, (RuntimeError,), {})
    return exc_cls(message or "")


# ---------------------------------------------------------------------------
# AgnoTracer
# ---------------------------------------------------------------------------


class AgnoTracer:
    """Consumes an Agno event stream and emits agent-trace spans.

    Feed it every event yielded by ``agent.run``/``agent.arun`` (or
    ``team.run``/``team.arun``) called with ``stream=True,
    stream_events=True`` via :meth:`process_event`. It maintains its own
    span registry keyed by Agno's ``run_id`` (for run spans), ``tool_call_id``
    (for tool spans), and a per-run stack of open model-request spans (model
    calls within one run never overlap, but a tool-calling loop can issue
    several in sequence).

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
    """

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        self._tracer: Tracer = tracer
        self._trace: Trace = trace
        self._run_spans: dict[str, Span] = {}
        self._tool_spans: dict[str, Span] = {}
        self._llm_stacks: dict[str, list[Span]] = {}
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_event(self, event: Any) -> None:
        """Handle a single event from an Agno run's event stream.

        Never raises — a malformed/unrecognized event is logged at debug
        level and skipped rather than aborting the caller's run loop.
        """
        try:
            self._process_event(event)
        except Exception:
            logger.debug(
                "agent-trace: failed to process Agno event %r",
                event,
                exc_info=True,
            )

    def close_open_spans(self) -> None:
        """Force-close any spans left open (e.g. the caller's loop broke early
        or the run was interrupted by an exception the event stream never
        reported). Safety net so a partially-consumed stream doesn't leak
        permanently-open spans into the trace.
        """
        with self._lock:
            run_spans = list(self._run_spans.values())
            self._run_spans.clear()
            tool_spans = list(self._tool_spans.values())
            self._tool_spans.clear()
            llm_spans = [s for stack in self._llm_stacks.values() for s in stack]
            self._llm_stacks.clear()
        for span in (*llm_spans, *tool_spans, *run_spans):
            if span.end_time is None:
                span.end(SpanStatus.ERROR)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_run_span(self, event: Any) -> Span | None:
        run_id = getattr(event, "run_id", None)
        if not run_id:
            return None
        kind, name = _actor(event)
        parent_run_id = getattr(event, "parent_run_id", None)
        parent_span_id: str | None = None
        if parent_run_id:
            with self._lock:
                parent_span = self._run_spans.get(str(parent_run_id))
            if parent_span is not None:
                parent_span_id = parent_span.span_id

        span = self._tracer.start_span(f"{kind}:{name}", parent_id=parent_span_id)
        span.set_attribute(f"agno.{kind}.name", name)
        span.set_attribute("agno.run_id", str(run_id))
        model = getattr(event, "model", None)
        if model:
            span.set_attribute("agno.model", str(model))
        with self._lock:
            self._run_spans[str(run_id)] = span
        return span

    def _pop_tool_span(self, tool_call_id: Any) -> Span | None:
        if not tool_call_id:
            return None
        with self._lock:
            return self._tool_spans.pop(str(tool_call_id), None)

    def _process_event(self, event: Any) -> None:
        raw_name = getattr(event, "event", None) or type(event).__name__
        name = _normalize_event_name(str(raw_name))

        if name == _RUN_STARTED:
            self._open_run_span(event)
            return

        run_id = getattr(event, "run_id", None)
        if run_id is None:
            return
        run_key = str(run_id)

        handler = self._EVENT_HANDLERS.get(name)
        if handler is not None:
            handler(self, event, run_key)

    def _handle_run_completed(self, event: Any, run_key: str) -> None:
        with self._lock:
            span = self._run_spans.pop(run_key, None)
        if span is not None and span.end_time is None:
            span.end(SpanStatus.OK)

    def _handle_run_error(self, event: Any, run_key: str) -> None:
        with self._lock:
            span = self._run_spans.pop(run_key, None)
        if span is not None:
            message = getattr(event, "content", None) or "Agno run error"
            exc = _make_exception(str(message), getattr(event, "error_type", None))
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)

    def _handle_run_cancelled(self, event: Any, run_key: str) -> None:
        with self._lock:
            span = self._run_spans.pop(run_key, None)
        if span is not None:
            span.set_attribute("agno.cancelled", True)
            reason = getattr(event, "reason", None)
            if reason:
                span.set_attribute("agno.cancel_reason", str(reason))
            if span.end_time is None:
                span.end(SpanStatus.OK)

    def _handle_model_request_started(self, event: Any, run_key: str) -> None:
        with self._lock:
            parent_span = self._run_spans.get(run_key)
        parent_span_id = parent_span.span_id if parent_span is not None else None
        model = getattr(event, "model", None) or "unknown"
        span = self._tracer.start_span(f"llm:{model}", parent_id=parent_span_id)
        span.set_attribute("llm.model", str(model))
        provider = getattr(event, "model_provider", None)
        if provider:
            span.set_attribute("llm.provider", str(provider))
        with self._lock:
            self._llm_stacks.setdefault(run_key, []).append(span)

    def _handle_model_request_completed(self, event: Any, run_key: str) -> None:
        with self._lock:
            stack = self._llm_stacks.get(run_key)
            span = stack.pop() if stack else None
        if span is None:
            return
        for attribute, field_name in (
            ("llm.usage.prompt_tokens", "input_tokens"),
            ("llm.usage.completion_tokens", "output_tokens"),
            ("llm.usage.total_tokens", "total_tokens"),
        ):
            value = getattr(event, field_name, None)
            if value is not None:
                span.set_attribute(attribute, int(value))
        ttft = getattr(event, "time_to_first_token", None)
        if ttft is not None:
            span.set_attribute("llm.time_to_first_token_s", float(ttft))
        if span.end_time is None:
            span.end(SpanStatus.OK)

    def _handle_tool_call_started(self, event: Any, run_key: str) -> None:
        tool = getattr(event, "tool", None)
        tool_call_id = getattr(tool, "tool_call_id", None) or f"{run_key}:{id(event)}"
        tool_name = getattr(tool, "tool_name", None) or "tool"
        with self._lock:
            parent_span = self._run_spans.get(run_key)
        parent_span_id = parent_span.span_id if parent_span is not None else None
        span = self._tracer.start_span(f"tool:{tool_name}", parent_id=parent_span_id)
        span.set_attribute("tool.name", str(tool_name))
        tool_args = getattr(tool, "tool_args", None)
        if tool_args:
            span.set_attribute("tool.args", str(tool_args)[:2000])
        with self._lock:
            self._tool_spans[str(tool_call_id)] = span

    def _handle_tool_call_completed(self, event: Any, run_key: str) -> None:
        tool = getattr(event, "tool", None)
        span = self._pop_tool_span(getattr(tool, "tool_call_id", None))
        if span is None:
            return
        result = getattr(tool, "result", None)
        if result is not None:
            span.set_attribute("tool.result_length", len(str(result)))
        child_run_id = getattr(tool, "child_run_id", None)
        if child_run_id:
            # Set when this tool call spawned a nested Agent/Team run (e.g.
            # Team's built-in delegate_task_to_member) — correlates this
            # tool span with the child run span.
            span.set_attribute("agno.child_run_id", str(child_run_id))
        if span.end_time is None:
            span.end(SpanStatus.OK)

    def _handle_tool_call_error(self, event: Any, run_key: str) -> None:
        tool = getattr(event, "tool", None)
        span = self._pop_tool_span(getattr(tool, "tool_call_id", None))
        if span is None:
            return
        message = getattr(event, "error", None) or "Agno tool call error"
        exc = _make_exception(str(message), "AgnoToolCallError")
        span.record_exception(exc)
        if span.end_time is None:
            span.end(SpanStatus.ERROR)

    _EVENT_HANDLERS: ClassVar[dict[str, Any]] = {
        _RUN_COMPLETED: _handle_run_completed,
        _RUN_ERROR: _handle_run_error,
        _RUN_CANCELLED: _handle_run_cancelled,
        _MODEL_REQUEST_STARTED: _handle_model_request_started,
        _MODEL_REQUEST_COMPLETED: _handle_model_request_completed,
        _TOOL_CALL_STARTED: _handle_tool_call_started,
        _TOOL_CALL_COMPLETED: _handle_tool_call_completed,
        _TOOL_CALL_ERROR: _handle_tool_call_error,
    }


# ---------------------------------------------------------------------------
# instrument_agent_run / instrument_agent_arun — convenience wrappers
# ---------------------------------------------------------------------------


def instrument_agent_run(
    runnable: Any,
    input: Any,
    *,
    tracer: Tracer,
    trace: Trace,
    **kwargs: Any,
) -> Any:
    """Run an Agno ``Agent`` or ``Team`` synchronously with span instrumentation.

    Combines :class:`AgnoTracer` with ``runnable.run(input, stream=True,
    stream_events=True, ...)``, draining the event stream into spans and
    returning the final ``RunOutput``/``TeamRunOutput``.

    Parameters
    ----------
    runnable:
        An Agno ``Agent`` or ``Team`` instance.
    input:
        The input passed through to ``runnable.run``.
    tracer:
        The active :class:`~agent_trace.Tracer`.
    trace:
        The current :class:`~agent_trace.Trace`.
    **kwargs:
        Additional keyword arguments forwarded to ``runnable.run`` (e.g.
        ``user_id``, ``session_id``).
    """
    _require_agno()

    hook = AgnoTracer(tracer=tracer, trace=trace)
    root_span = tracer.start_span("agno_run")
    result: Any = None

    try:
        for event in runnable.run(
            input, stream=True, stream_events=True, yield_run_output=True, **kwargs
        ):
            event_kind = type(event).__name__
            if event_kind in ("RunOutput", "TeamRunOutput"):
                result = event
                continue
            hook.process_event(event)
        root_span.end(SpanStatus.OK)
    except Exception as exc:
        root_span.record_exception(exc)
        if root_span.end_time is None:
            root_span.end(SpanStatus.ERROR)
        raise
    finally:
        # Safety net: an event stream that ends mid-run (a RunError event
        # closes the run span but never fires the matching
        # ModelRequestCompleted/ToolCallCompleted for whatever was in
        # flight, or the caller's loop broke early) must not leave spans
        # open forever.
        hook.close_open_spans()

    return result


async def instrument_agent_arun(
    runnable: Any,
    input: Any,
    *,
    tracer: Tracer,
    trace: Trace,
    **kwargs: Any,
) -> Any:
    """Async equivalent of :func:`instrument_agent_run`.

    Combines :class:`AgnoTracer` with ``runnable.arun(input, stream=True,
    stream_events=True, ...)``, draining the async event stream into spans
    and returning the final ``RunOutput``/``TeamRunOutput``.
    """
    _require_agno()

    hook = AgnoTracer(tracer=tracer, trace=trace)
    root_span = tracer.start_span("agno_run")
    result: Any = None

    try:
        async for event in runnable.arun(
            input, stream=True, stream_events=True, yield_run_output=True, **kwargs
        ):
            event_kind = type(event).__name__
            if event_kind in ("RunOutput", "TeamRunOutput"):
                result = event
                continue
            hook.process_event(event)
        root_span.end(SpanStatus.OK)
    except Exception as exc:
        root_span.record_exception(exc)
        if root_span.end_time is None:
            root_span.end(SpanStatus.ERROR)
        raise
    finally:
        # Safety net — see the matching comment in instrument_agent_run.
        hook.close_open_spans()

    return result
