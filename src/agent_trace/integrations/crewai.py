"""
crewAI integration — event-bus listener and span enrichment.

Subscribes to crewAI's process-wide event bus (``crewai.events.crewai_event_bus``)
and emits agent-trace spans for each crew kickoff, agent execution, task, LLM
call, and tool invocation — the crewAI-native equivalent of what
``LangGraphTracer`` does for LangGraph's callback-handler list and
``AgentTraceHook`` does for the OpenAI Agents SDK's hooks interface.

Unlike those two frameworks, crewAI has no per-invocation "pass a callback
object to this call" extension point. Instead it exposes a global event bus
(confirmed via a live install of ``crewai==1.15.2``:
``crewai/events/event_bus.py``) that every ``Crew``/``Agent``/``Task``/``LLM``
emits onto via ``crewai_event_bus.emit(source, event)``. Handlers register with
``crewai_event_bus.register_handler(event_type, handler)`` and receive
``(source, event)`` — ``source`` is the emitting object itself (the ``Crew``,
``Agent``, or ``LLM`` instance), and ``event`` is a typed Pydantic event model
(``crewai.events.types.*``).

Span pairing uses crewAI's own event-scope bookkeeping rather than an ad hoc
run-id: every "started" event type in this integration is one of crewAI's
``SCOPE_STARTING_EVENTS`` (``crewai/events/event_context.py``), and its
matching "completed"/"failed"/"error" event is a ``SCOPE_ENDING_EVENT`` whose
``started_event_id`` field is set (by ``CrewAIEventsBus._prepare_event``) to
the ``event_id`` of the started event it closes. Spans are therefore keyed by
``event.event_id`` on open and looked up by ``event.started_event_id`` on
close — the same mechanism crewAI itself uses internally to validate
start/end pairing. Parent/child nesting likewise reuses crewAI's own
``event.parent_event_id`` (the enclosing open scope's ``event_id`` at
emission time), so a task span opened while a crew-kickoff span is open is
automatically parented under it with no bookkeeping of our own.

Usage:
    from agent_trace import Tracer
    from agent_trace.integrations.crewai import CrewAITracer

    t = Tracer()
    with t.start_trace("my_crew", record=True) as trace:
        with CrewAITracer(tracer=t, trace=trace):
            result = crew.kickoff()

Known limitation — process-wide event bus:
    Because ``crewai_event_bus`` is a process-wide singleton, a
    ``CrewAITracer`` instance receives events from *every* Crew/Agent/Task run
    in the process while its handlers are registered — not just the run
    wrapped by the enclosing ``start_trace()`` block. If your process runs
    multiple concurrent ``crew.kickoff()`` calls, a single ``CrewAITracer``
    will see spans from all of them interleaved into one trace. This mirrors
    the concurrent-recording isolation gap already tracked in this repo for
    ``Tracer._install_recording_transport`` — treat both as "one in-flight
    run per process" until addressed. Always use ``CrewAITracer`` as a context
    manager (or call ``.close()`` explicitly) so handlers are unregistered
    when the trace ends; leaving it open leaks handlers onto the shared bus
    for the remaining lifetime of the process.

Known limitation — parent linkage can be dropped under concurrent handler
dispatch:
    Confirmed via a live end-to-end run (``examples/04-crewai-research-crew``)
    against a real (deliberately invalid) OpenAI key: ``CrewAIEventsBus.emit``
    computes ``event.parent_event_id`` synchronously on the emitting thread
    (correct and stable by the time any handler sees it), but *dispatches*
    each registered sync handler via a 10-worker ``ThreadPoolExecutor``
    (``crewai/events/event_bus.py``) with no ordering guarantee between
    handlers for different events. A child event's handler (e.g.
    ``llm_call_started``) can therefore run — and look up its parent span in
    ``self._spans`` — *before* the parent event's handler (e.g.
    ``agent_execution_started``) has finished registering that span,
    especially under an agent's internal retry loop, which fires several
    started/error pairs in quick succession. When this race is lost the
    child span still opens, closes, and records errors correctly — only its
    ``parent_id`` falls back to ``None`` (a flat span instead of a nested
    one) rather than being silently dropped or mis-attached to the wrong
    parent.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = ["CrewAITracer"]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "CrewAITracer requires crewai.\n"
    "Install it with:\n\n"
    "    pip install crewai\n"
)


def _require_crewai_events() -> Any:
    """Lazy import guard — raises a clear error if crewai is absent."""
    try:
        import crewai.events as crewai_events

        return crewai_events
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


class CrewAITracer:
    """Registers agent-trace span handlers on crewAI's global event bus.

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.

    ``crewai`` is imported lazily — importing this module succeeds even when
    ``crewai`` is not installed; only constructing ``CrewAITracer`` requires
    it.

    Use as a context manager (or call :meth:`close`) so handlers are removed
    from the shared event bus once the traced run finishes::

        with CrewAITracer(tracer=t, trace=trace):
            crew.kickoff()
    """

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        events = _require_crewai_events()

        self._tracer: Tracer = tracer
        self._trace: Trace = trace
        # Span registry keyed by the crewAI event_id of the *_started event
        # that opened the span (see module docstring — this mirrors crewAI's
        # own started_event_id pairing rather than inventing a new run-id).
        self._spans: dict[str, Span] = {}
        self._lock: threading.Lock = threading.Lock()
        self._bus = events.crewai_event_bus
        self._registered: list[tuple[type, Any]] = []
        self._closed = False

        self._register(events.CrewKickoffStartedEvent, self._on_crew_started)
        self._register(events.CrewKickoffCompletedEvent, self._on_crew_completed)
        self._register(events.CrewKickoffFailedEvent, self._on_crew_failed)

        self._register(events.AgentExecutionStartedEvent, self._on_agent_started)
        self._register(events.AgentExecutionCompletedEvent, self._on_agent_completed)
        self._register(events.AgentExecutionErrorEvent, self._on_agent_error)

        self._register(events.TaskStartedEvent, self._on_task_started)
        self._register(events.TaskCompletedEvent, self._on_task_completed)
        self._register(events.TaskFailedEvent, self._on_task_failed)

        self._register(events.LLMCallStartedEvent, self._on_llm_started)
        self._register(events.LLMCallCompletedEvent, self._on_llm_completed)
        self._register(events.LLMCallFailedEvent, self._on_llm_failed)

        self._register(events.ToolUsageStartedEvent, self._on_tool_started)
        self._register(events.ToolUsageFinishedEvent, self._on_tool_finished)
        self._register(events.ToolUsageErrorEvent, self._on_tool_error)

    # ------------------------------------------------------------------
    # Registration / lifecycle
    # ------------------------------------------------------------------

    def _register(self, event_type: type, handler: Any) -> None:
        self._bus.register_handler(event_type, handler)
        self._registered.append((event_type, handler))

    def close(self) -> None:
        """Unregister all handlers from the shared crewAI event bus.

        Safe to call more than once. After ``close()``, this instance no
        longer receives crewAI events and any spans still open are left
        open (they will simply never close — the same behaviour LangGraph's
        ``on_chain_end`` would have if it were never called).
        """
        if self._closed:
            return
        self._closed = True
        for event_type, handler in self._registered:
            self._bus.off(event_type, handler)
        self._registered.clear()

    def __enter__(self) -> CrewAITracer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal span helpers (same pattern as LangGraphTracer / AgentTraceHook)
    # ------------------------------------------------------------------

    def _open_span(
        self,
        key: str,
        name: str,
        parent_key: str | None = None,
    ) -> Span:
        parent_span_id: str | None = None
        if parent_key is not None:
            with self._lock:
                parent_span = self._spans.get(parent_key)
            if parent_span is not None:
                parent_span_id = parent_span.span_id

        span = self._tracer.start_span(name, parent_id=parent_span_id)
        with self._lock:
            self._spans[key] = span
        return span

    def _close_span(self, key: str, status: SpanStatus = SpanStatus.OK) -> Span | None:
        with self._lock:
            span = self._spans.pop(key, None)
        if span is not None and span.end_time is None:
            span.end(status)
        return span

    def _close_span_with_error(self, key: str | None, error_message: str) -> None:
        """Pop the span (if the started_event_id key is known), record the
        error text as an exception event, and close it as ERROR.

        crewAI's failure events carry ``error: str`` rather than a raised
        exception object, so a synthetic ``RuntimeError`` is used to satisfy
        ``Span.record_exception``'s ``BaseException`` signature.
        """
        if key is None:
            return
        with self._lock:
            span = self._spans.pop(key, None)
        if span is not None:
            span.record_exception(RuntimeError(error_message))
            if span.end_time is None:
                span.end(SpanStatus.ERROR)

    # ------------------------------------------------------------------
    # Crew kickoff
    # ------------------------------------------------------------------

    def _on_crew_started(self, source: Any, event: Any) -> None:
        crew_name = event.crew_name or "crew"
        span = self._open_span(
            event.event_id, f"crew:{crew_name}", event.parent_event_id
        )
        span.set_attribute("crew.name", crew_name)

    def _on_crew_completed(self, source: Any, event: Any) -> None:
        span = self._close_span(event.started_event_id, SpanStatus.OK)
        if span is not None:
            try:
                span.set_attribute("crew.total_tokens", int(event.total_tokens or 0))
            except Exception:
                logger.debug(
                    "agent-trace: failed to record crew total_tokens", exc_info=True
                )

    def _on_crew_failed(self, source: Any, event: Any) -> None:
        self._close_span_with_error(event.started_event_id, event.error)

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _on_agent_started(self, source: Any, event: Any) -> None:
        agent_role = getattr(event.agent, "role", None) or "agent"
        span = self._open_span(
            event.event_id, f"agent:{agent_role}", event.parent_event_id
        )
        span.set_attribute("agent.role", agent_role)

    def _on_agent_completed(self, source: Any, event: Any) -> None:
        self._close_span(event.started_event_id, SpanStatus.OK)

    def _on_agent_error(self, source: Any, event: Any) -> None:
        self._close_span_with_error(event.started_event_id, event.error)

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def _on_task_started(self, source: Any, event: Any) -> None:
        task_name = event.task_name or "task"
        span = self._open_span(
            event.event_id, f"task:{task_name}", event.parent_event_id
        )
        span.set_attribute("task.name", task_name)
        if event.task_id:
            span.set_attribute("task.id", event.task_id)

    def _on_task_completed(self, source: Any, event: Any) -> None:
        self._close_span(event.started_event_id, SpanStatus.OK)

    def _on_task_failed(self, source: Any, event: Any) -> None:
        self._close_span_with_error(event.started_event_id, event.error)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _on_llm_started(self, source: Any, event: Any) -> None:
        model = event.model or "unknown"
        span = self._open_span(event.event_id, f"llm:{model}", event.parent_event_id)
        span.set_attribute("llm.model", model)

    def _on_llm_completed(self, source: Any, event: Any) -> None:
        span = self._close_span(event.started_event_id, SpanStatus.OK)
        if span is not None:
            try:
                usage = event.usage or {}
                if usage.get("prompt_tokens") is not None:
                    span.set_attribute(
                        "llm.usage.prompt_tokens", int(usage["prompt_tokens"])
                    )
                if usage.get("completion_tokens") is not None:
                    span.set_attribute(
                        "llm.usage.completion_tokens", int(usage["completion_tokens"])
                    )
                if usage.get("total_tokens") is not None:
                    span.set_attribute(
                        "llm.usage.total_tokens", int(usage["total_tokens"])
                    )
                if event.finish_reason:
                    span.set_attribute("llm.finish_reason", event.finish_reason)
            except Exception:
                logger.debug(
                    "agent-trace: failed to record token usage for crewAI LLM call",
                    exc_info=True,
                )

    def _on_llm_failed(self, source: Any, event: Any) -> None:
        self._close_span_with_error(event.started_event_id, event.error)

    # ------------------------------------------------------------------
    # Tool usage
    # ------------------------------------------------------------------

    def _on_tool_started(self, source: Any, event: Any) -> None:
        tool_name = event.tool_name or "tool"
        span = self._open_span(
            event.event_id, f"tool:{tool_name}", event.parent_event_id
        )
        span.set_attribute("tool.name", tool_name)

    def _on_tool_finished(self, source: Any, event: Any) -> None:
        span = self._close_span(event.started_event_id, SpanStatus.OK)
        if span is not None:
            try:
                span.set_attribute(
                    "tool.output_length",
                    len(str(event.output)) if event.output is not None else 0,
                )
            except Exception:
                logger.debug(
                    "agent-trace: failed to record tool output length", exc_info=True
                )

    def _on_tool_error(self, source: Any, event: Any) -> None:
        self._close_span_with_error(event.started_event_id, str(event.error))
