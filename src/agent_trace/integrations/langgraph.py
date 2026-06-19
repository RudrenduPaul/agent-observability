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

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = ["LangGraphTracer"]

logger = logging.getLogger(__name__)

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


# Module-level singleton: the concrete tracer class is built once (the first
# time _get_tracer_class() is called) so that BaseCallbackHandler is a real
# base at class-definition time rather than being spliced in at instantiation
# via __bases__ mutation (which is unsafe under concurrency and forbidden in
# Python 3.14+).
_LangGraphTracerClass: type | None = None
_tracer_class_lock: threading.Lock = threading.Lock()


def _get_tracer_class() -> type:
    """Return (and lazily build) the concrete LangGraphTracer implementation.

    The class is created exactly once, with ``BaseCallbackHandler`` as a
    genuine base class at definition time.  Subsequent calls return the cached
    class without re-importing or re-defining anything.
    """
    global _LangGraphTracerClass  # noqa: PLW0603
    if _LangGraphTracerClass is not None:
        return _LangGraphTracerClass

    with _tracer_class_lock:
        # Double-checked locking: re-test inside the lock.
        if _LangGraphTracerClass is not None:
            return _LangGraphTracerClass

        base_cls: type = _base_callback_handler()

        class _LangGraphTracerImpl(base_cls):  # type: ignore[misc]
            """Concrete implementation — see LangGraphTracer for public docs."""

            def __init__(self, tracer: Tracer, trace: Trace) -> None:
                super().__init__()
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
                    parent_span_id = (
                        parent_span.span_id if parent_span is not None else None
                    )

                span = self._tracer.start_span(name, parent_id=parent_span_id)
                with self._lock:
                    self._spans[run_key] = span
                return span

            def _close_span(
                self,
                run_id: uuid.UUID | str,
                status: SpanStatus = SpanStatus.OK,
            ) -> Span | None:
                """End the span for *run_id* and remove it from the registry."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.pop(run_key, None)
                if span is not None and span.end_time is None:
                    span.end(status)
                return span

            def _close_span_with_exception(
                self,
                run_id: uuid.UUID | str,
                error: BaseException,
            ) -> None:
                """Pop the span, record the exception, and end it as ERROR.

                Consolidates the three error callbacks into a single lock
                acquisition + record + end sequence.
                """
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.pop(run_key, None)
                if span is not None:
                    span.record_exception(error)
                    if span.end_time is None:
                        span.end(SpanStatus.ERROR)

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
                    serialized.get("name")
                    or serialized.get("id", [None])[-1]
                    or "chain"
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
                self._close_span_with_exception(run_id, error)

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
                """Start a span for a legacy LLM call, recording the model name."""
                model_name: str = (
                    (serialized.get("kwargs") or {}).get("model_name")
                    or (serialized.get("kwargs") or {}).get("model")
                    or serialized.get("name")
                    or "llm"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model_name", model_name)
                span.set_attribute("llm.prompt_count", len(prompts))

            def on_chat_model_start(
                self,
                serialized: dict[str, Any],
                messages: list[list[Any]],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                """Start a span for a ChatModel call (e.g. ChatOpenAI, ChatAnthropic).

                Modern LangChain chat models fire ``on_chat_model_start`` instead
                of ``on_llm_start``.  Without this handler those spans are silently
                dropped.
                """
                model_name: str = serialized.get("name") or "unknown"
                span = self._open_span(run_id, "llm", parent_run_id)
                span.set_attribute("llm.model", model_name)

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
                        token_usage = (
                            usage.get("token_usage") or usage.get("usage") or {}
                        )
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
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record token usage for run %r",
                            str(run_id),
                            exc_info=True,
                        )
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
                self._close_span_with_exception(run_id, error)

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
                self._close_span_with_exception(run_id, error)

        _LangGraphTracerClass = _LangGraphTracerImpl
        return _LangGraphTracerClass


class LangGraphTracer:
    """Langchain/LangGraph callback handler that emits agent-trace spans.

    Implements the ``BaseCallbackHandler`` interface from ``langchain_core``
    so it can be passed directly in the ``config["callbacks"]`` list of any
    LangGraph graph invocation.

    ``langchain_core`` is imported lazily — importing this module succeeds even
    when ``langchain_core`` is not installed.  The import (and the class
    definition that inherits from ``BaseCallbackHandler``) happens once, the
    first time a ``LangGraphTracer`` instance is created.

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
    """

    def __new__(cls, tracer: Tracer, trace: Trace) -> LangGraphTracer:
        # Construct the concrete impl directly so Python's normal type.__call__
        # runs _LangGraphTracerImpl.__init__ automatically.  We cannot use
        # impl_cls.__new__(impl_cls) + manual __init__ because the returned
        # object would not be an instance of LangGraphTracer, causing Python
        # to skip __init__ entirely — leaving _tracer/_trace/_spans unset.
        impl_cls = _get_tracer_class()
        return impl_cls(tracer, trace)  # type: ignore[no-any-return]

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        # __init__ is called on the instance whose __class__ is already the
        # concrete impl class (set by __new__).  Delegate to its __init__.
        # This path is only reached if someone subclasses LangGraphTracer
        # directly; normal construction goes through the impl class __init__.
        pass  # pragma: no cover
