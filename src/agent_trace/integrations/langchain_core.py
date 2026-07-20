"""
Generic, framework-agnostic LangChain callback integration.

``LangGraphTracer`` (``agent_trace.integrations.langgraph``) only attaches
to a compiled LangGraph ``StateGraph`` — it hard-codes LangGraph-specific
span-naming conventions (``node:<name>``, ``branch:dispatch``, checkpointer
tracking, ...) and is documented as a LangGraph integration. The much
larger population of plain-LangChain users — anyone calling
``Runnable.invoke()``/``.batch()``/``.ainvoke()`` directly (a document
compressor pipeline, a bare `Chain`, a `Retriever`, ...) with no LangGraph
graph anywhere in sight — has had *no* chain-level integration at all: only
the raw HTTP interceptor sees anything, and an application-level exception
raised inside a Runnable's own Python code (not during an HTTP call) is
invisible everywhere, confirmed via reproduction against issue #31192 (a
`DocumentCompressorPipeline`'s `IndexError` in `_parse_ranking` attaches to
nothing — ``Tracer.start_trace``'s exception handler only records onto
spans already present in ``trace.spans``, and with no chain-level
integration wired in, none exist).

``LangChainTracer`` closes that gap: a plain ``BaseCallbackHandler``
implementation that works with *any* LangChain `Runnable`, not just a
LangGraph graph — pass it into ``config={"callbacks": [...]}`` on any
``.invoke()``/``.batch()``/``.ainvoke()`` call. On error, it calls
``Span.record_exception()`` on a span it opens itself for that Runnable
invocation, the same mechanism LangGraphTracer uses, but without any
LangGraph-specific assumptions about span naming or graph structure.

Usage::

    from agent_trace import Tracer
    from agent_trace.integrations.langchain_core import LangChainTracer

    t = Tracer(trace_dir=...)
    with t.start_trace("my-runnable", record=True) as trace:
        cb = LangChainTracer(tracer=t, trace=trace)
        result = my_runnable.invoke(input, config={"callbacks": [cb]})

Requires ``langchain-core`` (``pip install agent-observability-trace-cli[langchain]``) —
notably *not* ``langgraph``, unlike ``LangGraphTracer``.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = ["LangChainTracer"]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LangChainTracer requires langchain-core.\n"
    "Install it with:\n\n"
    "    pip install agent-observability-trace-cli[langchain]\n"
)

_MAX_ATTR_LEN = 8_000


def _to_attr_string(value: Any, *, max_len: int = _MAX_ATTR_LEN) -> str:
    """Best-effort, bounded string form of an arbitrary LangChain value
    (inputs/outputs dicts, message lists, ...) — mirrors
    agent_trace.integrations.langgraph._to_attr_string, duplicated (not
    imported) so this module has zero dependency on langgraph.py and,
    transitively, on the ``langgraph`` package itself."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > max_len:
        text = text[:max_len] + "...<truncated>"
    return text


def _require_langchain_core() -> Any:
    try:
        import langchain_core

        return langchain_core
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _base_callback_handler() -> Any:
    _require_langchain_core()
    from langchain_core.callbacks import BaseCallbackHandler

    return BaseCallbackHandler


_LangChainTracerClass: type | None = None
_tracer_class_lock: threading.Lock = threading.Lock()


def _get_tracer_class() -> type:
    """Return (and lazily build) the concrete LangChainTracer
    implementation, with BaseCallbackHandler as a genuine base class at
    definition time — same pattern as
    agent_trace.integrations.langgraph._get_tracer_class."""
    global _LangChainTracerClass  # noqa: PLW0603
    if _LangChainTracerClass is not None:
        return _LangChainTracerClass

    with _tracer_class_lock:
        if _LangChainTracerClass is not None:
            return _LangChainTracerClass

        BaseCallbackHandler = _base_callback_handler()

        class _LangChainTracerImpl(BaseCallbackHandler):  # type: ignore[misc,valid-type]
            """Generic BaseCallbackHandler — see module docstring."""

            def __init__(self, tracer: Tracer, trace: Trace) -> None:
                self._tracer = tracer
                self._trace = trace
                self._spans: dict[str, Span] = {}
                self._lock: threading.Lock = threading.Lock()

            # -- span lifecycle -------------------------------------------

            def _open_span(
                self,
                run_id: uuid.UUID | str,
                name: str,
                parent_run_id: uuid.UUID | str | None = None,
            ) -> Span:
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
                self, run_id: uuid.UUID | str, status: SpanStatus = SpanStatus.OK
            ) -> Span | None:
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.pop(run_key, None)
                if span is not None and span.end_time is None:
                    span.end(status)
                return span

            def _close_span_with_exception(
                self, run_id: uuid.UUID | str, error: BaseException
            ) -> None:
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.pop(run_key, None)
                if span is None:
                    return
                span.record_exception(error)
                if span.end_time is None:
                    span.end(SpanStatus.ERROR)

            # -- chain callbacks -------------------------------------------

            def on_chain_start(
                self,
                serialized: dict[str, Any] | None,
                inputs: dict[str, Any],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                ser = serialized or {}
                name = (
                    kwargs.get("name")
                    or ser.get("name")
                    or (ser.get("id") or [None])[-1]
                    or "chain"
                )
                span = self._open_span(run_id, f"chain:{name}", parent_run_id)
                if tags:
                    span.set_attribute("chain.tags", ",".join(tags))
                if metadata:
                    span.set_attribute("chain.metadata", _to_attr_string(metadata))
                if inputs:
                    span.set_attribute("chain.inputs", _to_attr_string(inputs))

            def on_chain_end(
                self,
                outputs: Any,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None and outputs:
                    try:
                        span.set_attribute("chain.outputs", _to_attr_string(outputs))
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record chain outputs for "
                            "run %r",
                            run_key,
                            exc_info=True,
                        )
                self._close_span(run_id, SpanStatus.OK)

            def on_chain_error(
                self,
                error: BaseException,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                self._close_span_with_exception(run_id, error)

            # -- LLM callbacks -----------------------------------------------

            def on_llm_start(
                self,
                serialized: dict[str, Any] | None,
                prompts: list[str],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                ser = serialized or {}
                model_name = (
                    (ser.get("kwargs") or {}).get("model_name")
                    or (ser.get("kwargs") or {}).get("model")
                    or ser.get("name")
                    or "llm"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", str(model_name))

            def on_chat_model_start(
                self,
                serialized: dict[str, Any] | None,
                messages: list[list[Any]],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                ser = serialized or {}
                model_name = (
                    (ser.get("kwargs") or {}).get("model_name")
                    or (ser.get("kwargs") or {}).get("model")
                    or ser.get("name")
                    or "llm"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", str(model_name))
                try:
                    span.set_attribute("llm.messages", _to_attr_string(messages))
                except Exception:
                    logger.debug(
                        "agent-trace: failed to record chat model messages "
                        "for run %r",
                        str(run_id),
                        exc_info=True,
                    )

            def on_llm_end(
                self,
                response: Any,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                self._close_span(run_id, SpanStatus.OK)

            def on_llm_error(
                self,
                error: BaseException,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                self._close_span_with_exception(run_id, error)

            # -- tool callbacks -------------------------------------------

            def on_tool_start(
                self,
                serialized: dict[str, Any] | None,
                input_str: str,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                ser = serialized or {}
                tool_name = kwargs.get("name") or ser.get("name") or "tool"
                span = self._open_span(run_id, f"tool:{tool_name}", parent_run_id)
                span.set_attribute("tool.name", str(tool_name))
                if input_str:
                    span.set_attribute("tool.input", _to_attr_string(input_str))

            def on_tool_end(
                self,
                output: Any,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None and output is not None:
                    try:
                        span.set_attribute("tool.output", _to_attr_string(output))
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record tool output for "
                            "run %r",
                            run_key,
                            exc_info=True,
                        )
                self._close_span(run_id, SpanStatus.OK)

            def on_tool_error(
                self,
                error: BaseException,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                **kwargs: Any,
            ) -> None:
                self._close_span_with_exception(run_id, error)

        _LangChainTracerClass = _LangChainTracerImpl
        return _LangChainTracerClass


def LangChainTracer(tracer: Tracer, trace: Trace) -> Any:  # noqa: N802
    """Construct a generic LangChain ``BaseCallbackHandler`` — see module
    docstring for usage. Works with any ``Runnable``, not just a compiled
    LangGraph graph (that's ``LangGraphTracer``, a separate, LangGraph-
    specific integration)."""
    cls = _get_tracer_class()
    return cls(tracer=tracer, trace=trace)
