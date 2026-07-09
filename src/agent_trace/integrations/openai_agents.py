"""
OpenAI Agents SDK integration.

Wraps openai-agents Runner to emit spans for each agent turn, tool call,
and handoff.

Usage (hook-based — recommended)::

    from agents import Agent, Runner
    from agent_trace import Tracer
    from agent_trace.integrations.openai_agents import AgentTraceHook

    t = Tracer()
    with t.start_trace("my_agent_run", record=True) as trace:
        hook = AgentTraceHook(tracer=t, trace=trace)
        result = await Runner.run(agent, "hello", hooks=hook)

Usage (convenience wrapper)::

    from agent_trace.integrations.openai_agents import instrument_runner

    result = await instrument_runner(agent, "hello", tracer=t, trace=trace)

Usage (streamed convenience wrapper)::

    from agent_trace.integrations.openai_agents import instrument_runner_streamed

    streamed = await instrument_runner_streamed(agent, "hello", tracer=t, trace=trace)
    async for event in streamed.stream_events():
        ...
    print(streamed.final_output)

Note on the openai-agents SDK's hook interface (as of 0.17+/0.18.x): the
``RunHooksBase`` class the SDK actually calls only exposes seven ``async def``
methods — ``on_agent_start``, ``on_agent_end``, ``on_handoff``,
``on_llm_start``, ``on_llm_end``, ``on_tool_start``, ``on_tool_end``. There is
no ``on_agent_error``/``on_tool_error`` hook anywhere in the SDK; most
in-process errors (e.g. a failed tool call) are caught internally by the SDK
and turned into a normal-looking tool result, never reaching any hook at all.
Exception-to-span attribution is therefore handled by
``instrument_runner``/``instrument_runner_streamed`` wrapping the ``Runner``
call itself in a try/except, not by a hook method — see
``AgentTraceHook._close_open_spans_with_exception``.
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from agent_trace._inspect import _edit_distance
from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

# ``agents`` (openai-agents) is an optional dependency — see the
# ``openai-agents`` extra in pyproject.toml.  Resolve ``RunHooksBase`` at
# import time so ``AgentTraceHook`` genuinely satisfies the SDK's hook
# interface (``Runner.run(..., hooks=hook)`` calls ``validate_run_hooks()``,
# which rejects anything that isn't a ``RunHooksBase`` instance) when the
# package is installed, while still importing cleanly when it isn't.
try:
    from agents.lifecycle import RunHooksBase as _RunHooksBase
except ImportError:  # pragma: no cover - exercised by unit tests without the extra
    _RunHooksBase = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from agents.realtime import RealtimeSession, RealtimeSessionEvent

__all__ = [
    "AgentTraceHook",
    "AgentTraceRealtimeHook",
    "instrument_runner",
    "instrument_runner_streamed",
]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The OpenAI Agents integration requires the openai-agents package.\n"
    "Install it with:\n\n"
    "    pip install openai-agents\n"
)

# Cap on how much of a tool result / stream-event payload gets persisted onto
# a span attribute.  Attribute values must stay small, primitive strings —
# this is not a place to dump multi-megabyte tool outputs.
_MAX_ATTR_LEN = 4000

# `ModelResponse.output` item `.type` values that represent the model asking
# to invoke a tool (as opposed to a plain message/reasoning item).  Sourced
# from the openai-agents SDK's `openai.types.responses` output-item union.
_TOOL_CALL_OUTPUT_TYPES = frozenset(
    {
        "function_call",
        "custom_tool_call",
        "computer_call",
        "file_search_call",
        "web_search_call",
        "code_interpreter_call",
        "local_shell_call",
        "shell_call",
        "apply_patch_call",
        "mcp_call",
        "image_generation_call",
    }
)


def _require_openai_agents() -> Any:
    """Lazy import guard — raises a clear error if openai-agents is absent."""
    # Try the canonical package name first, then the legacy alias.
    for module_name in ("agents", "openai_agents"):
        try:
            import importlib

            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError(_INSTALL_HINT)


def _get_runner_cls(sdk: Any) -> Any:
    runner_cls = getattr(sdk, "Runner", None)
    if runner_cls is None:
        import importlib

        runner_mod = importlib.import_module("agents.run")
        runner_cls = runner_mod.Runner
    return runner_cls


# Matches the "Tool <name> not found" shape openai-agents' ModelBehaviorError
# raises when the model hallucinates a tool name (#1671), optionally
# single/double-quoted, e.g. `Tool 'lookup_wather' not found in agent
# 'voice-agent'` or `Tool transfer_back_to_supervisor not found`.
_TOOL_NOT_FOUND_RE = re.compile(r"Tool ['\"]?([\w\-.]+)['\"]? not found")


def _truncate(value: Any) -> str:
    text = str(value)
    if len(text) > _MAX_ATTR_LEN:
        return text[:_MAX_ATTR_LEN] + f"...<truncated, {len(text)} chars total>"
    return text


def _extract_reasoning_effort(model_settings: Any) -> str | None:
    """Pull ``model_settings.reasoning["effort"]`` out, dict- or attr-style."""
    reasoning = getattr(model_settings, "reasoning", None)
    if reasoning is None:
        return None
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    else:
        effort = getattr(reasoning, "effort", None)
    return str(effort) if effort is not None else None


def _response_has_tool_calls(response: Any) -> bool:
    """Whether a ``ModelResponse.output`` contains any tool-call-shaped item."""
    output = getattr(response, "output", None)
    if not output:
        return False
    for item in output:
        if getattr(item, "type", None) in _TOOL_CALL_OUTPUT_TYPES:
            return True
    return False


# ---------------------------------------------------------------------------
# AgentTraceHook
# ---------------------------------------------------------------------------


class AgentTraceHook(_RunHooksBase):  # type: ignore[misc]
    """Hook implementation for the openai-agents SDK lifecycle events.

    Pass an instance of this class to a ``Runner`` (via ``hooks=hook``) to
    receive callbacks on agent start/end, tool calls, LLM calls, and
    handoffs.  Subclasses ``agents.lifecycle.RunHooksBase`` so it satisfies
    ``Runner.run()``'s ``validate_run_hooks()`` check, and every method is
    ``async def`` to match the SDK's own hook interface (the SDK
    ``await``s each hook via ``asyncio.gather(...)``).

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
        # Thread-safe span registry keyed by a string context key.
        self._spans: dict[str, Span] = {}
        # id(context) -> (from_agent_name, to_agent_name), set by on_handoff
        # and consumed by on_agent_end to open a duration-based handoff span.
        self._pending_handoffs: dict[int, tuple[str, str]] = {}
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_span(self, key: str, name: str, parent_key: str | None = None) -> Span:
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

    def _close_open_spans_with_exception(self, exc: BaseException) -> None:
        """Close every still-open span this hook is tracking as an error.

        The current openai-agents SDK has no ``on_agent_error``/
        ``on_tool_error`` hook, so a crashed turn otherwise leaves its
        ``agent:``/``tool:``/``llm:`` span open forever with no error status
        and no exception recorded.  Call this from the ``except`` branch
        around ``Runner.run(...)``/``Runner.run_streamed(...)`` so the trace
        shows *which* span was in flight when the run failed, not just that
        it silently stopped.
        """
        with self._lock:
            spans = list(self._spans.values())
            self._spans.clear()
            self._pending_handoffs.clear()
        for span in spans:
            if span.end_time is None:
                span.record_exception(exc)
                span.end(SpanStatus.ERROR)

    # ------------------------------------------------------------------
    # openai-agents hook interface (all async — see module docstring)
    # ------------------------------------------------------------------

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        """Called when an agent turn begins."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        model: str = getattr(agent, "model", None) or "unknown"
        ctx_id = id(context)

        # If a handoff transition into this agent is pending, close its
        # duration-based handoff span now that the incoming agent has
        # actually started.
        self._close_span(f"handoff:{ctx_id}:{agent_name}", SpanStatus.OK)

        key = f"agent:{ctx_id}:{agent_name}"
        span = self._open_span(key, f"agent:{agent_name}")
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.model", model)

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """Called when an agent turn completes successfully."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        ctx_id = id(context)
        key = f"agent:{ctx_id}:{agent_name}"
        closed_span = self._close_span(key, SpanStatus.OK)

        # If on_handoff fired for this agent, open a bounded transition span
        # now, to be closed by the next agent's on_agent_start.  This gives
        # the handoff itself an explicit duration_ms instead of only a
        # point-in-time event on the outgoing agent's span.
        with self._lock:
            pending = self._pending_handoffs.pop(ctx_id, None)
        if pending is not None:
            from_name, to_name = pending
            if from_name == agent_name:
                parent_id = closed_span.parent_id if closed_span is not None else None
                handoff_span = self._tracer.start_span(
                    f"handoff:{from_name}->{to_name}", parent_id=parent_id
                )
                handoff_span.set_attribute("handoff.from_agent", from_name)
                handoff_span.set_attribute("handoff.to_agent", to_name)
                with self._lock:
                    self._spans[f"handoff:{ctx_id}:{to_name}"] = handoff_span

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """Called when a tool invocation begins."""
        tool_name: str = getattr(tool, "name", None) or "tool"
        agent_name: str = getattr(agent, "name", None) or "agent"
        parent_key = f"agent:{id(context)}:{agent_name}"
        key = f"tool:{id(context)}:{tool_name}"
        span = self._open_span(key, f"tool:{tool_name}", parent_key=parent_key)
        span.set_attribute("tool.name", tool_name)

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        """Called when a tool invocation completes."""
        tool_name: str = getattr(tool, "name", None) or "tool"
        key = f"tool:{id(context)}:{tool_name}"
        span = self._close_span(key, SpanStatus.OK)
        if span is not None:
            try:
                span.set_attribute(
                    "tool.result_length",
                    len(str(result)) if result is not None else 0,
                )
                # Persist the actual result content (not just its length) —
                # this is frequently the only diagnostic text available,
                # e.g. when the SDK's own error-handling wraps a tool
                # exception into its "successful" string return value.
                if result is not None:
                    span.set_attribute("tool.result", _truncate(result))
            except Exception:
                logger.debug(
                    "agent-trace: failed to record tool result for %r",
                    tool_name,
                    exc_info=True,
                )

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        """Called when control is handed off from one agent to another."""
        from_name: str = getattr(from_agent, "name", None) or "unknown"
        to_name: str = getattr(to_agent, "name", None) or "unknown"
        ctx_id = id(context)
        parent_key = f"agent:{ctx_id}:{from_name}"

        with self._lock:
            parent_span = self._spans.get(parent_key)

        if parent_span is not None:
            parent_span.add_event(
                "handoff",
                attributes={
                    "handoff.from_agent": from_name,
                    "handoff.to_agent": to_name,
                },
            )

        # Record the pending transition so on_agent_end (for from_agent) can
        # open a duration-based handoff span, closed by on_agent_start (for
        # to_agent).
        with self._lock:
            self._pending_handoffs[ctx_id] = (from_name, to_name)

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        """Called before each LLM request — record model + settings used."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        model: str = getattr(agent, "model", None) or "unknown"
        key = f"llm:{id(context)}:{agent_name}"
        parent_key = f"agent:{id(context)}:{agent_name}"
        span = self._open_span(key, f"llm:{model}", parent_key=parent_key)
        span.set_attribute("llm.model", model)

        model_settings = getattr(agent, "model_settings", None)
        if model_settings is not None:
            try:
                effort = _extract_reasoning_effort(model_settings)
                if effort is not None:
                    span.set_attribute("llm.model_settings.reasoning_effort", effort)

                verbosity = getattr(model_settings, "verbosity", None)
                if verbosity is not None:
                    span.set_attribute("llm.model_settings.verbosity", str(verbosity))
            except Exception:
                logger.debug(
                    "agent-trace: failed to record model_settings",
                    exc_info=True,
                )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        """Called after each LLM response — record token usage + tool-call presence."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        key = f"llm:{id(context)}:{agent_name}"
        with self._lock:
            span = self._spans.get(key)
        if span is not None:
            try:
                usage = getattr(response, "usage", None)
                if usage is not None:
                    pt = getattr(usage, "input_tokens", None)
                    ct = getattr(usage, "output_tokens", None)
                    tt = getattr(usage, "total_tokens", None)
                    if pt is not None:
                        span.set_attribute("llm.usage.prompt_tokens", int(pt))
                    if ct is not None:
                        span.set_attribute("llm.usage.completion_tokens", int(ct))
                    # total_tokens: prefer explicit field, fall back to sum
                    if tt is not None:
                        span.set_attribute("llm.usage.total_tokens", int(tt))
                    elif pt is not None and ct is not None:
                        span.set_attribute("llm.usage.total_tokens", int(pt) + int(ct))

                span.set_attribute(
                    "llm.response.has_tool_calls", _response_has_tool_calls(response)
                )
            except Exception:
                logger.debug(
                    "agent-trace: failed to record LLM response attributes",
                    exc_info=True,
                )
        self._close_span(key, SpanStatus.OK)


# ---------------------------------------------------------------------------
# instrument_runner — high-level async wrapper (Runner.run)
# ---------------------------------------------------------------------------


async def instrument_runner(
    agent: Any,
    input: Any,
    *,
    tracer: Tracer,
    trace: Trace,
    **kwargs: Any,
) -> Any:
    """Run an openai-agents ``Agent`` through ``Runner.run`` with span instrumentation.

    This is a convenience wrapper that combines ``AgentTraceHook`` with
    ``Runner.run``.  Pass an ``Agent`` and the input text; agent-trace handles
    hooking in the span collection automatically.  If the run raises, every
    span still open in the hook's registry is closed as an error (with the
    exception attached) before the exception is re-raised — see
    ``AgentTraceHook._close_open_spans_with_exception``.

    Parameters
    ----------
    agent:
        An openai-agents ``Agent`` instance.
    input:
        The input string (or message list) to pass to ``Runner.run``.
    tracer:
        The active :class:`~agent_trace.Tracer`.
    trace:
        The current :class:`~agent_trace.Trace`.
    **kwargs:
        Additional keyword arguments forwarded to ``Runner.run``
        (e.g. ``max_turns``, ``context``).

    Returns
    -------
    Any
        The ``RunResult`` from ``Runner.run``.
    """
    sdk = _require_openai_agents()
    runner_cls = _get_runner_cls(sdk)

    hook = AgentTraceHook(tracer=tracer, trace=trace)
    root_span = tracer.start_span("agent_run")
    result: Any = None

    try:
        raw = runner_cls.run(agent, input, hooks=hook, **kwargs)
        if inspect.isawaitable(raw):
            result = await raw
        else:
            result = raw

        root_span.end(SpanStatus.OK)
    except Exception as exc:
        hook._close_open_spans_with_exception(exc)
        root_span.record_exception(exc)
        if root_span.end_time is None:
            root_span.end(SpanStatus.ERROR)
        raise

    return result


# ---------------------------------------------------------------------------
# instrument_runner_streamed — high-level wrapper (Runner.run_streamed)
# ---------------------------------------------------------------------------


class _TracedRunResultStreaming:
    """Wraps openai-agents' ``RunResultStreaming``, instrumenting ``stream_events()``.

    Every attribute/method other than ``stream_events`` is proxied straight
    through to the wrapped ``RunResultStreaming`` (``final_output``,
    ``last_agent``, ``to_input_list``, ``cancel``, ...) so this is a drop-in
    replacement for the object ``Runner.run_streamed()`` itself returns.
    """

    def __init__(
        self, result_streaming: Any, root_span: Span, hook: AgentTraceHook
    ) -> None:
        self._result_streaming = result_streaming
        self._root_span = root_span
        self._hook = hook

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result_streaming, name)

    async def stream_events(self) -> AsyncIterator[Any]:
        """Yield each event exactly as ``stream_events()`` would, recording it.

        A ``SpanEvent`` is added to the root span for every event actually
        delivered to this iterator — i.e. this observes the consumer's real
        drain rate, not just whether ``RunHooks`` callbacks fired internally
        (those are two distinct channels inside ``RunResultStreaming``).
        """
        try:
            async for event in self._result_streaming.stream_events():
                event_type = getattr(event, "type", type(event).__name__)
                self._root_span.add_event(
                    "stream_event", attributes={"event.type": str(event_type)}
                )
                yield event
            if self._root_span.end_time is None:
                self._root_span.end(SpanStatus.OK)
        except Exception as exc:
            self._hook._close_open_spans_with_exception(exc)
            self._root_span.record_exception(exc)
            if self._root_span.end_time is None:
                self._root_span.end(SpanStatus.ERROR)
            raise


async def instrument_runner_streamed(
    agent: Any,
    input: Any,
    *,
    tracer: Tracer,
    trace: Trace,
    **kwargs: Any,
) -> _TracedRunResultStreaming:
    """Run an openai-agents ``Agent`` via ``Runner.run_streamed``, span-instrumented.

    ``instrument_runner`` only ever calls ``Runner.run`` — a reporter calling
    ``Runner.run_streamed()``/consuming ``stream_events()`` directly gets no
    exception attribution at all, since ``RunHooks`` firing correctly does
    not imply the corresponding event actually reached the consumer's
    ``stream_events()`` loop.  This wrapper covers that call shape: it
    returns a :class:`_TracedRunResultStreaming` immediately (matching
    ``Runner.run_streamed()``'s own non-blocking contract), whose
    ``stream_events()`` is span-instrumented and whose "agent_run_streamed"
    root span stays open until the caller's consumption of the stream
    actually finishes (or raises).

    Parameters
    ----------
    agent:
        An openai-agents ``Agent`` instance.
    input:
        The input string (or message list) to pass to ``Runner.run_streamed``.
    tracer:
        The active :class:`~agent_trace.Tracer`.
    trace:
        The current :class:`~agent_trace.Trace`.
    **kwargs:
        Additional keyword arguments forwarded to ``Runner.run_streamed``
        (e.g. ``max_turns``, ``context``).

    Returns
    -------
    _TracedRunResultStreaming
        A drop-in wrapper around the SDK's ``RunResultStreaming``.
    """
    sdk = _require_openai_agents()
    runner_cls = _get_runner_cls(sdk)

    hook = AgentTraceHook(tracer=tracer, trace=trace)
    root_span = tracer.start_span("agent_run_streamed")

    raw = runner_cls.run_streamed(agent, input, hooks=hook, **kwargs)
    result_streaming = await raw if inspect.isawaitable(raw) else raw

    return _TracedRunResultStreaming(result_streaming, root_span, hook)


# ---------------------------------------------------------------------------
# AgentTraceRealtimeHook — Realtime API (agents.realtime) instrumentation
# ---------------------------------------------------------------------------


class AgentTraceRealtimeHook:
    """Span instrumentation for the OpenAI Agents SDK's Realtime API.

    The Realtime API (``agents.realtime``) has no ``RunHooks``-style
    callback surface at all — ``RealtimeRunner.run()`` doesn't take a
    ``hooks=`` kwarg, and ``RealtimeSession`` exposes turns, tool calls, and
    handoffs only by *iterating the session itself*
    (``RealtimeSession`` is an ``AsyncIterator[RealtimeSessionEvent]``).
    ``AgentTraceHook``, which is scoped entirely to the turn-based
    ``Runner.run()``/``Runner.run_streamed()`` loop, never fires for a
    ``RealtimeSession`` at all — so today a handoff, tool-call, or error
    happening inside a realtime voice session produces no span whatsoever.

    This class wraps that iteration instead, translating each
    ``RealtimeSessionEvent`` into the same span open/close/event pattern
    ``AgentTraceHook`` uses for the turn-based loop:

    - ``agent_start`` / ``agent_end`` -> open/close an ``agent:<name>`` span
    - ``tool_start`` / ``tool_end`` -> open/close a ``tool:<name>`` span
      (child of the current agent span), persisting arguments/result
    - ``handoff`` -> an event on the outgoing agent's span
    - ``error`` / ``guardrail_tripped`` -> ``record_exception``-style event
      on the session's root span, without ending the session (a realtime
      session's error/guardrail events are not necessarily fatal). When an
      ``error`` event's text matches the "Tool <name> not found" shape
      (#1671's ``ModelBehaviorError`` traceback — an intermittent crash
      inside a Realtime/WebSocket voice session, distinct from the
      httpx-captured chat-completion exchanges
      ``check_tool_call_name_fuzzy_match`` diagnoses), the offending name is
      fuzzy-matched against the tool names registered on the most recently
      started agent for this session (captured from the ``agent_start``
      event's ``event.agent.tools``) and the nearest match + edit distance
      are attached as ``exception.nearest_registered_tool``/
      ``exception.edit_distance`` attributes on the exception event.

    Usage::

        from agents.realtime import RealtimeRunner
        from agent_trace.integrations.openai_agents import AgentTraceRealtimeHook

        session = await RealtimeRunner(realtime_agent).run()
        rt_hook = AgentTraceRealtimeHook(tracer=t, trace=trace)
        async with session:
            async for event in rt_hook.wrap(session):
                ...  # your own event handling; spans are recorded as a side effect
    """

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        self._tracer: Tracer = tracer
        self._trace: Trace = trace
        self._spans: dict[str, Span] = {}
        self._lock: threading.Lock = threading.Lock()
        # session_key -> registered tool names of the most recently started
        # agent for that session (#1671). Populated from `agent_start`'s
        # `event.agent.tools`, the only place a realtime session's tool
        # roster is available — there is no separate "tools registered"
        # event.
        self._registered_tools: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Internal helpers (same pattern as AgentTraceHook)
    # ------------------------------------------------------------------

    def _open_span(self, key: str, name: str, parent_key: str | None = None) -> Span:
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

    def _diagnose_tool_not_found(
        self, session_key: str, error_message: str
    ) -> dict[str, Any]:
        """Fuzzy-match a "Tool <name> not found" error's ``<name>`` against
        this session's registered tool names (#1671 — an intermittent
        ``ModelBehaviorError``-shaped crash inside a Realtime/WebSocket
        voice session), reusing the same edit-distance helper
        ``check_tool_call_name_fuzzy_match`` already applies to
        httpx-captured chat-completion exchanges. Returns an empty dict
        (no attributes added) when the error text doesn't match the
        pattern, or no registered tool names were captured for this
        session."""
        match = _TOOL_NOT_FOUND_RE.search(error_message)
        if match is None:
            return {}
        with self._lock:
            registered = self._registered_tools.get(session_key)
        if not registered:
            return {}
        called_name = match.group(1)
        nearest = min(registered, key=lambda r: _edit_distance(called_name, r))
        return {
            "exception.nearest_registered_tool": nearest,
            "exception.edit_distance": _edit_distance(called_name, nearest),
        }

    def _capture_registered_tools(self, session_key: str, agent: Any) -> None:
        """Track *agent*'s registered tool names for *session_key* (#1671),
        so a later "Tool <name> not found" error event on this session can
        be fuzzy-matched against them (see _diagnose_tool_not_found)."""
        tool_names = {
            name
            for tool in getattr(agent, "tools", None) or ()
            if isinstance((name := getattr(tool, "name", None)), str)
        }
        if tool_names:
            with self._lock:
                self._registered_tools[session_key] = tool_names

    def _handle_event(self, session_key: str, event: Any) -> None:
        event_type = getattr(event, "type", None)

        if event_type == "agent_start":
            agent = getattr(event, "agent", None)
            agent_name = getattr(agent, "name", None) or "agent"
            self._open_span(
                f"agent:{session_key}:{agent_name}",
                f"agent:{agent_name}",
                parent_key=session_key,
            ).set_attribute("agent.name", agent_name)
            self._capture_registered_tools(session_key, agent)

        elif event_type == "agent_end":
            agent_name = getattr(getattr(event, "agent", None), "name", None) or "agent"
            self._close_span(f"agent:{session_key}:{agent_name}", SpanStatus.OK)

        elif event_type == "tool_start":
            agent_name = getattr(getattr(event, "agent", None), "name", None) or "agent"
            tool_name = getattr(getattr(event, "tool", None), "name", None) or "tool"
            span = self._open_span(
                f"tool:{session_key}:{tool_name}",
                f"tool:{tool_name}",
                parent_key=f"agent:{session_key}:{agent_name}",
            )
            span.set_attribute("tool.name", tool_name)
            arguments = getattr(event, "arguments", None)
            if arguments is not None:
                span.set_attribute("tool.arguments", _truncate(arguments))

        elif event_type == "tool_end":
            tool_name = getattr(getattr(event, "tool", None), "name", None) or "tool"
            tool_key = f"tool:{session_key}:{tool_name}"
            closed_span = self._close_span(tool_key, SpanStatus.OK)
            if closed_span is not None:
                output = getattr(event, "output", None)
                if output is not None:
                    closed_span.set_attribute("tool.result", _truncate(output))
                    closed_span.set_attribute("tool.result_length", len(str(output)))

        elif event_type == "handoff":
            from_agent = getattr(event, "from_agent", None)
            to_agent = getattr(event, "to_agent", None)
            from_name = getattr(from_agent, "name", None) or "unknown"
            to_name = getattr(to_agent, "name", None) or "unknown"
            with self._lock:
                parent_span = self._spans.get(f"agent:{session_key}:{from_name}")
            if parent_span is not None:
                parent_span.add_event(
                    "handoff",
                    attributes={
                        "handoff.from_agent": from_name,
                        "handoff.to_agent": to_name,
                    },
                )

        elif event_type in ("error", "guardrail_tripped"):
            with self._lock:
                root_span = self._spans.get(session_key)
            if root_span is not None:
                detail = (
                    getattr(event, "error", None)
                    or getattr(event, "message", None)
                    or event
                )
                message = _truncate(detail)
                attributes = {
                    "exception.type": event_type,
                    "exception.message": message,
                    **(
                        self._diagnose_tool_not_found(session_key, message)
                        if event_type == "error"
                        else {}
                    ),
                }
                root_span.add_event(
                    "exception" if event_type == "error" else "guardrail_tripped",
                    attributes=attributes,
                )

    async def wrap(
        self, session: RealtimeSession
    ) -> AsyncIterator[RealtimeSessionEvent]:
        """Iterate *session*, recording spans as a side effect; yields events as-is."""
        session_key = f"realtime:{id(session)}"
        self._open_span(session_key, "realtime_session")
        try:
            async for event in session:
                self._handle_event(session_key, event)
                yield event
            self._close_span(session_key, SpanStatus.OK)
        except Exception as exc:
            with self._lock:
                spans = list(self._spans.values())
                self._spans.clear()
            for span in spans:
                if span.end_time is None:
                    span.record_exception(exc)
                    span.end(SpanStatus.ERROR)
            raise
