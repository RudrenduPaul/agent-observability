"""
OpenAI Agents SDK integration.

Wraps openai-agents Runner to emit spans for each agent turn, tool call,
and handoff.

Usage:
    from agent_trace.integrations.openai_agents import instrument_runner
    from agent_trace import tracer

    with tracer.start_trace("my_agent_run", record=True) as trace:
        result = await instrument_runner(runner, input="hello", trace=trace)
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = [
    "AgentTraceHook",
    "instrument_runner",
]

_INSTALL_HINT = (
    "The OpenAI Agents integration requires the openai-agents package.\n"
    "Install it with:\n\n"
    "    pip install openai-agents\n"
)


def _require_openai_agents() -> Any:
    """Lazy import guard — raises a clear error if openai-agents is absent."""
    try:
        import agents  # type: ignore[import-not-found]

        return agents
    except ImportError:
        try:
            import openai_agents

            return openai_agents
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc


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
            except Exception:  # noqa: S110
                pass

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


# ---------------------------------------------------------------------------
# instrument_runner — high-level async wrapper
# ---------------------------------------------------------------------------


async def instrument_runner(
    runner: Any,
    *,
    input: Any,
    tracer: Tracer,
    trace: Trace,
    **kwargs: Any,
) -> Any:
    """Wrap a ``Runner.run()`` call with agent-trace spans.

    Creates a root span named ``"agent_run"`` and, where the SDK exposes a
    streaming interface, creates child spans for each step.

    Parameters
    ----------
    runner:
        An openai-agents ``Runner`` (or compatible object with a ``run``
        method).
    input:
        The input to pass to ``runner.run()``.
    tracer:
        The active :class:`~agent_trace.Tracer`.
    trace:
        The current :class:`~agent_trace.Trace`.
    **kwargs:
        Additional keyword arguments forwarded to ``runner.run()``.

    Returns
    -------
    Any
        The final result from the runner.
    """
    _require_openai_agents()

    root_span = tracer.start_span("agent_run")
    result: Any = None

    try:
        # Attempt to use the streaming interface if available so we can
        # create per-step child spans.
        run_method = getattr(runner, "run_streamed", None) or getattr(
            runner, "run", None
        )
        if run_method is None:
            raise AttributeError(
                f"{type(runner).__name__!r} has no 'run' or 'run_streamed' method."
            )

        step_index = 0
        raw = run_method(input, **kwargs)

        # If the result is an async iterable, iterate and create step spans
        if hasattr(raw, "__aiter__"):
            async for step in raw:
                step_span_name = (
                    getattr(step, "type", None)
                    or getattr(step, "event", None)
                    or f"step_{step_index}"
                )
                step_span = tracer.start_span(
                    str(step_span_name), parent_id=root_span.span_id
                )
                try:
                    _enrich_step_span(step_span, step)
                    step_span.end(SpanStatus.OK)
                except Exception:
                    step_span.end(SpanStatus.OK)
                step_index += 1
                # Track the last step as the result
                result = step
        elif asyncio.iscoroutine(raw):
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


def _enrich_step_span(span: Span, step: Any) -> None:
    """Attach available metadata from a runner step to *span*."""
    try:
        if agent := getattr(step, "agent", None):
            span.set_attribute("agent.name", getattr(agent, "name", str(agent)))
            if model := getattr(agent, "model", None):
                span.set_attribute("agent.model", str(model))

        if tool_calls := getattr(step, "tool_calls", None):
            names = [
                getattr(tc, "name", None)
                or getattr(tc, "function", {}).get("name", "?")
                for tc in tool_calls
            ]
            span.set_attribute("tool_calls", ",".join(str(n) for n in names))

        if usage := getattr(step, "usage", None):
            if pt := getattr(usage, "input_tokens", None):
                span.set_attribute("llm.usage.prompt_tokens", int(pt))
            if ct := getattr(usage, "output_tokens", None):
                span.set_attribute("llm.usage.completion_tokens", int(ct))
    except Exception:  # noqa: S110
        pass
