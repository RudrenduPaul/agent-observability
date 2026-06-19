"""
LangGraph integration — callback handler and graph-aware span enrichment.

Instruments a LangGraph StateGraph to emit spans for each node execution,
edge traversal, and LLM call within the graph.

Usage:
    from agent_trace.integrations.langgraph import LangGraphTracer
    from agent_trace import tracer

    with tracer.start_trace("my_graph", record=True) as trace:
        result = graph.invoke(
            input_state,
            config={"callbacks": [LangGraphTracer(tracer=tracer, trace=trace)]}
        )
"""

from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = ["LangGraphTracer"]

_INSTALL_HINT = (
    "LangGraphTracer requires langchain-core and langgraph.\n"
    "Install them with:\n\n"
    "    pip install langchain-core langgraph\n"
)


def _require_langchain_core() -> Any:
    """Lazy import guard — raises a clear error if langchain_core is absent."""
    try:
        import langchain_core

        return langchain_core
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _base_callback_handler() -> Any:
    """Return langchain_core.callbacks.BaseCallbackHandler (lazy import)."""
    _require_langchain_core()
    from langchain_core.callbacks import BaseCallbackHandler

    return BaseCallbackHandler


class LangGraphTracer:
    """Langchain/LangGraph callback handler that emits agent-trace spans.

    Implements the ``BaseCallbackHandler`` interface from ``langchain_core``
    so it can be passed directly in the ``config["callbacks"]`` list of any
    LangGraph graph invocation.

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
    """

    # Declare the base class lazily — only resolved at instantiation time so
    # that importing this module does NOT fail when langchain_core is absent.
    _base: type | None = None

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        # Trigger the import check at construction time, not at module load.
        base_cls = _base_callback_handler()

        # Dynamically ensure this instance satisfies the ABC.  We do this
        # at runtime rather than at class definition so the import of this
        # module succeeds even when langchain_core is not installed.
        if not isinstance(self, base_cls):
            # Re-register the class with the base to satisfy isinstance checks
            # used internally by LangChain's callback manager.
            self.__class__.__bases__ = (base_cls, *self.__class__.__bases__)

        self._tracer: Tracer = tracer
        self._trace: Trace = trace
        # Thread-safe span registry: run_id (UUID str) -> open Span
        self._spans: dict[str, Span] = {}
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_span(
        self,
        run_id: uuid.UUID | str,
        name: str,
        parent_run_id: uuid.UUID | str | None = None,
    ) -> Span:
        """Create a span and register it in the local registry."""
        run_key = str(run_id)
        parent_span_id: str | None = None
        if parent_run_id is not None:
            with self._lock:
                parent_span = self._spans.get(str(parent_run_id))
            parent_span_id = parent_span.span_id if parent_span is not None else None

        span = self._tracer.start_span(name, parent_id=parent_span_id)
        with self._lock:
            self._spans[run_key] = span
        return span

    def _close_span(
        self,
        run_id: uuid.UUID | str,
        status: SpanStatus = SpanStatus.OK,
    ) -> Span | None:
        """End the span registered for *run_id* and remove it from the registry."""
        run_key = str(run_id)
        with self._lock:
            span = self._spans.pop(run_key, None)
        if span is not None and span.end_time is None:
            span.end(status)
        return span

    # ------------------------------------------------------------------
    # Chain (graph node) callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a span when a graph node (chain) begins execution."""
        node_name: str = (
            serialized.get("name") or serialized.get("id", [None])[-1] or "chain"
        )
        span = self._open_span(run_id, f"node:{node_name}", parent_run_id)
        span.set_attribute("langgraph.node", node_name)
        if tags:
            span.set_attribute("langgraph.tags", ",".join(tags))

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """End the span when a graph node completes successfully."""
        self._close_span(run_id, SpanStatus.OK)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """End the span with ERROR status when a graph node raises."""
        run_key = str(run_id)
        with self._lock:
            span = self._spans.get(run_key)
        if span is not None:
            span.record_exception(error)
        self._close_span(run_id, SpanStatus.ERROR)

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a span for an LLM call, recording the model name."""
        model_name: str = (
            (serialized.get("kwargs") or {}).get("model_name")
            or (serialized.get("kwargs") or {}).get("model")
            or serialized.get("name")
            or "llm"
        )
        span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
        span.set_attribute("llm.model_name", model_name)
        span.set_attribute("llm.prompt_count", len(prompts))

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """End the LLM span, attaching token usage when available."""
        run_key = str(run_id)
        with self._lock:
            span = self._spans.get(run_key)
        if span is not None:
            try:
                usage = getattr(response, "llm_output", {}) or {}
                token_usage = usage.get("token_usage") or usage.get("usage") or {}
                if token_usage:
                    span.set_attribute(
                        "llm.usage.prompt_tokens",
                        int(token_usage.get("prompt_tokens", 0)),
                    )
                    span.set_attribute(
                        "llm.usage.completion_tokens",
                        int(token_usage.get("completion_tokens", 0)),
                    )
                    span.set_attribute(
                        "llm.usage.total_tokens",
                        int(token_usage.get("total_tokens", 0)),
                    )
            except Exception:  # noqa: S110
                pass
        self._close_span(run_id, SpanStatus.OK)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """End the LLM span with ERROR status."""
        run_key = str(run_id)
        with self._lock:
            span = self._spans.get(run_key)
        if span is not None:
            span.record_exception(error)
        self._close_span(run_id, SpanStatus.ERROR)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a span when a tool begins execution."""
        tool_name: str = serialized.get("name") or "tool"
        span = self._open_span(run_id, f"tool:{tool_name}", parent_run_id)
        span.set_attribute("tool.name", tool_name)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """End the tool span with OK status."""
        self._close_span(run_id, SpanStatus.OK)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """End the tool span with ERROR status."""
        run_key = str(run_id)
        with self._lock:
            span = self._spans.get(run_key)
        if span is not None:
            span.record_exception(error)
        self._close_span(run_id, SpanStatus.ERROR)
