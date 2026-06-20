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
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = [
    "AgentTraceHook",
    "instrument_runner",
]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The OpenAI Agents integration requires the openai-agents package.\n"
    "Install it with:\n\n"
    "    pip install openai-agents\n"
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


# ---------------------------------------------------------------------------
# AgentTraceHook
# ---------------------------------------------------------------------------


class AgentTraceHook:
    """Hook implementation for the openai-agents SDK lifecycle events.

    Pass an instance of this class to a ``Runner`` or ``Agent`` to receive
    callbacks on agent start/end, tool calls, and handoffs.

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
        # Thread-safe span registry keyed by a string context key
        self._spans: dict[str, Span] = {}
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

    # ------------------------------------------------------------------
    # openai-agents hook interface
    # ------------------------------------------------------------------

    def on_agent_start(self, context: Any, agent: Any) -> None:
        """Called when an agent turn begins."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        model: str = getattr(agent, "model", None) or "unknown"
        key = f"agent:{id(context)}:{agent_name}"
        span = self._open_span(key, f"agent:{agent_name}")
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.model", model)

    def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """Called when an agent turn completes successfully."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        key = f"agent:{id(context)}:{agent_name}"
        self._close_span(key, SpanStatus.OK)

    def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """Called when a tool invocation begins."""
        tool_name: str = getattr(tool, "name", None) or "tool"
        agent_name: str = getattr(agent, "name", None) or "agent"
        parent_key = f"agent:{id(context)}:{agent_name}"
        key = f"tool:{id(context)}:{tool_name}"
        span = self._open_span(key, f"tool:{tool_name}", parent_key=parent_key)
        span.set_attribute("tool.name", tool_name)

    def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
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
            except Exception:
                logger.debug(
                    "agent-trace: failed to record tool result length for %r",
                    tool_name,
                    exc_info=True,
                )

    def on_agent_error(self, context: Any, agent: Any, error: BaseException) -> None:
        """Called when an agent turn raises — close the span with ERROR status."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        key = f"agent:{id(context)}:{agent_name}"
        span = self._close_span(key, SpanStatus.ERROR)
        if span is not None:
            span.record_exception(error)

    def on_tool_error(
        self, context: Any, agent: Any, tool: Any, error: BaseException
    ) -> None:
        """Called when a tool invocation raises — close the span with ERROR status."""
        tool_name: str = getattr(tool, "name", None) or "tool"
        key = f"tool:{id(context)}:{tool_name}"
        span = self._close_span(key, SpanStatus.ERROR)
        if span is not None:
            span.record_exception(error)

    def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        """Called when control is handed off from one agent to another."""
        from_name: str = getattr(from_agent, "name", None) or "unknown"
        to_name: str = getattr(to_agent, "name", None) or "unknown"
        parent_key = f"agent:{id(context)}:{from_name}"

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

    def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        """Called before each LLM request — record the model being called."""
        agent_name: str = getattr(agent, "name", None) or "agent"
        model: str = getattr(agent, "model", None) or "unknown"
        key = f"llm:{id(context)}:{agent_name}"
        parent_key = f"agent:{id(context)}:{agent_name}"
        span = self._open_span(key, f"llm:{model}", parent_key=parent_key)
        span.set_attribute("llm.model", model)

    def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        """Called after each LLM response — record token usage if available."""
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
            except Exception:
                logger.debug(
                    "agent-trace: failed to record token usage",
                    exc_info=True,
                )
        self._close_span(key, SpanStatus.OK)


# ---------------------------------------------------------------------------
# instrument_runner — high-level async wrapper
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
    hooking in the span collection automatically.

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

    runner_cls = getattr(sdk, "Runner", None)
    if runner_cls is None:
        import importlib

        runner_mod = importlib.import_module("agents.run")
        runner_cls = runner_mod.Runner

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
        root_span.record_exception(exc)
        if root_span.end_time is None:
            root_span.end(SpanStatus.ERROR)
        raise

    return result
