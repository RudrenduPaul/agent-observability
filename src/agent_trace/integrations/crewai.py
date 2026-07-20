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

Known race (fixed) — a "close" event's handler can run before its own
"open" event's handler:
    The same unordered ``ThreadPoolExecutor`` dispatch above has a second,
    more severe consequence, reproduced live (a scripted, fully-offline
    ``BaseLLM`` making two sequential calls within one ReAct loop):
    ``llm_call_completed``'s handler was observed running — and printing its
    debug line — *before* the matching ``llm_call_started``'s handler for
    the very same call had run at all, on a different pool worker thread.
    Naively, this would mean the "started" handler eventually creates the
    span (via ``_open_span``) but no "completed" handler is left to ever
    close it — a span silently, permanently stuck at ``SpanStatus.UNSET``
    with ``end_time=None``, worse than the parent-linkage degradation above
    since a genuinely open span (not just a flat one) sits in the trace
    forever. ``_open_span``/``_close_span``/``_close_span_with_error`` below
    handle this: a close that finds no span yet is stashed in
    ``self._pending_closes`` instead of being dropped, and ``_open_span``
    checks (and immediately applies) any pending close for its own key the
    moment it does run — so the span still always ends up correctly closed,
    regardless of which of the pair's two handlers happens to run first.

Known limitation (fixed by ``close()``) — ``kickoff()`` returns before its
own last events are processed:
    Confirmed via a live, fully-offline ``crew.kickoff()`` reproduction (a
    custom ``BaseLLM`` subclass returning canned responses, no network
    calls): because ``crewai_event_bus.emit()`` dispatches sync handlers onto
    a background ``ThreadPoolExecutor`` and nothing in crewAI's own call
    chain awaits the returned ``Future``, ``Crew.kickoff()`` routinely
    returns to its caller *before* the handlers for its own final events
    (typically the last ``llm_call_completed``/``agent_execution_completed``/
    ``crew_kickoff_completed`` trio) have finished running — reproduced with
    spans still ``SpanStatus.UNSET`` (open) immediately after ``kickoff()``
    returned, which then closed correctly a moment later on a background
    thread. ``CrewAITracer.close()`` calls ``crewai_event_bus.flush()``
    (crewAI's own public API for exactly this "wait for pending handlers at
    the end of an operation like kickoff" case) before unregistering, so a
    caller using the documented ``with CrewAITracer(...): crew.kickoff()``
    pattern is guaranteed a fully-closed span tree by the time that ``with``
    block exits — do not skip calling ``close()`` (e.g. by holding a
    ``CrewAITracer`` open across multiple ``kickoff()`` calls without ever
    exiting it) if you need the trace to be complete before you read or
    export it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = ["CrewAITracer"]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "CrewAITracer requires crewai.\nInstall it with:\n\n    pip install crewai\n"
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
        # Keyed by the *_started event's event_id, same as self._spans.
        # Holds a (status, finisher) pair for a "close" event that arrived
        # before its matching "open" event's handler had a chance to run —
        # see the module docstring's "close arrives before its own open"
        # known-and-fixed race for why this exists. Consumed (and popped)
        # the moment the matching _open_span() call for that key runs.
        self._pending_closes: dict[str, tuple[SpanStatus, Callable[[Span], None]]] = {}
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
        """Flush pending event-bus work, then unregister all handlers from
        the shared crewAI event bus.

        Confirmed via a live, fully-offline ``crew.kickoff()`` reproduction
        (a custom ``BaseLLM`` subclass, no network): ``crewai_event_bus.emit()``
        dispatches sync handlers onto a background ``ThreadPoolExecutor`` and
        returns a ``Future`` that nothing in crewAI's own call chain awaits —
        ``Crew.kickoff()`` itself returns to its caller *before* the handlers
        for its own last few events (typically the final ``llm_call_completed``/
        ``agent_execution_completed``/``crew_kickoff_completed`` trio) have
        necessarily finished running. Without draining that queue here, a
        caller doing::

            with CrewAITracer(tracer=t, trace=trace):
                result = crew.kickoff()
            # trace.spans / trace.json written right here could still show
            # dangling open llm:/tool: spans that close moments later,
            # invisibly, on a background thread.

        would non-deterministically get an incomplete trace — worse, a
        *silently* incomplete one, since every span that does eventually
        close still ends up correct; only the *timing* relative to
        ``kickoff()`` returning is wrong. ``crewai_event_bus.flush()`` (a
        public method documented for exactly this "at the end of operations
        like kickoff" use case) blocks until every pending handler future
        bus-wide has completed, so calling it here — before unregistering —
        guarantees every event this instance's handlers were going to see for
        the just-completed run has, in fact, been seen by the time ``close()``
        returns.

        Safe to call more than once. After ``close()``, this instance no
        longer receives crewAI events; any span that is still open at that
        point (e.g. because the wrapped run itself was cut off, such as by
        an unhandled exception outside crewAI's own try/except) is simply
        left open, matching what LangGraph's ``on_chain_end`` not firing
        would look like.
        """
        if self._closed:
            return
        self._closed = True
        self._bus.flush()
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
        """Open a span for *key* (a ``*_started`` event's ``event_id``).

        If a ``_close_span``/``_close_span_with_error`` call for this exact
        *key* already ran and found nothing to close (see
        ``self._pending_closes`` and the module docstring's "close arrives
        before its own open" race), that pending close is applied
        immediately here — the span still goes through a real open-then-close
        transition, just compressed into this one call, instead of being
        left open forever because the handler that would have closed it
        already ran and gave up.
        """
        parent_span_id: str | None = None
        if parent_key is not None:
            with self._lock:
                parent_span = self._spans.get(parent_key)
            if parent_span is not None:
                parent_span_id = parent_span.span_id

        span = self._tracer.start_span(name, parent_id=parent_span_id)
        with self._lock:
            pending = self._pending_closes.pop(key, None)
            if pending is None:
                self._spans[key] = span

        if pending is not None:
            status, finisher = pending
            finisher(span)
            if span.end_time is None:
                span.end(status)
        return span

    def _close_span(
        self,
        key: str,
        status: SpanStatus = SpanStatus.OK,
        finisher: Callable[[Span], None] | None = None,
    ) -> Span | None:
        """Close the span opened under *key*, applying *finisher* (if given)
        before ending it — attributes are always set on a still-open span,
        never after ``end()`` (``Span`` documents post-``end()`` mutation as
        undefined for exporters).

        If no span is registered under *key* yet — the ``_open_span`` call
        that would have created it hasn't run yet, confirmed via a live,
        fully-offline reproduction to be a real, reproducible race in
        crewAI's own thread-pool handler dispatch, not a hypothetical one —
        the close is stashed in ``self._pending_closes`` instead of being
        silently dropped, so ``_open_span`` can apply it retroactively the
        moment it does run.
        """
        with self._lock:
            span = self._spans.pop(key, None)
            if span is None:
                self._pending_closes[key] = (status, finisher or (lambda _span: None))
        if span is not None:
            if finisher is not None:
                finisher(span)
            if span.end_time is None:
                span.end(status)
        return span

    def _close_span_with_error(self, key: str | None, error_message: str) -> None:
        """Close the span opened under *key* (if the started_event_id key is
        known) as ERROR, recording *error_message* as an exception event —
        same open-before-close-arrives handling as ``_close_span``.

        crewAI's failure events carry ``error: str`` rather than a raised
        exception object, so a synthetic ``RuntimeError`` is used to satisfy
        ``Span.record_exception``'s ``BaseException`` signature.
        """
        if key is None:
            return

        def finisher(span: Span) -> None:
            span.record_exception(RuntimeError(error_message))

        self._close_span(key, SpanStatus.ERROR, finisher=finisher)

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
        def finisher(span: Span) -> None:
            try:
                span.set_attribute("crew.total_tokens", int(event.total_tokens or 0))
            except Exception:
                logger.debug(
                    "agent-trace: failed to record crew total_tokens", exc_info=True
                )

        self._close_span(event.started_event_id, SpanStatus.OK, finisher=finisher)

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
        def finisher(span: Span) -> None:
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

        self._close_span(event.started_event_id, SpanStatus.OK, finisher=finisher)

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
        def finisher(span: Span) -> None:
            try:
                span.set_attribute(
                    "tool.output_length",
                    len(str(event.output)) if event.output is not None else 0,
                )
            except Exception:
                logger.debug(
                    "agent-trace: failed to record tool output length", exc_info=True
                )

        self._close_span(event.started_event_id, SpanStatus.OK, finisher=finisher)

    def _on_tool_error(self, source: Any, event: Any) -> None:
        self._close_span_with_error(event.started_event_id, str(event.error))
