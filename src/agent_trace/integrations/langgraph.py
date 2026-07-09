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

import asyncio
import contextvars
import json
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

# Span attributes only accept str/int/float/bool (see agent_trace.core.span).
# Anything structured (BaseMessage, dicts full of BaseMessages, dataclasses,
# etc.) has to be flattened to a bounded string before it can be stored.
_MAX_ATTR_LEN = 8_000

# _deep_serialize() bounds — deliberately small. This is span-attribute
# summary data, not a full-fidelity dump, and the bound also protects against
# pathological/self-referential objects (e.g. an unconfigured
# unittest.mock.MagicMock, whose attribute access and method calls always
# yield a brand-new child Mock) that would otherwise recurse indefinitely.
_MAX_SERIALIZE_DEPTH = 6
_MAX_COLLECTION_ITEMS = 200


def _safe_str(value: Any) -> str:
    """str(value), degrading to a placeholder instead of raising."""
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def _serialize_mapping(
    value: dict[Any, Any], *, _depth: int, _seen: frozenset[int]
) -> dict[str, Any]:
    """_deep_serialize() branch for dict-like values, capped at
    _MAX_COLLECTION_ITEMS keys."""
    out: dict[str, Any] = {}
    items = list(value.items())
    for k, v in items[:_MAX_COLLECTION_ITEMS]:
        out[str(k)] = _deep_serialize(v, _depth=_depth + 1, _seen=_seen)
    if len(items) > _MAX_COLLECTION_ITEMS:
        out["..."] = f"<{len(items) - _MAX_COLLECTION_ITEMS} more items truncated>"
    return out


def _serialize_sequence(
    value: list[Any] | tuple[Any, ...], *, _depth: int, _seen: frozenset[int]
) -> list[Any]:
    """_deep_serialize() branch for list/tuple values, capped at
    _MAX_COLLECTION_ITEMS items."""
    items = list(value)
    out_list = [
        _deep_serialize(item, _depth=_depth + 1, _seen=_seen)
        for item in items[:_MAX_COLLECTION_ITEMS]
    ]
    if len(items) > _MAX_COLLECTION_ITEMS:
        out_list.append(f"<{len(items) - _MAX_COLLECTION_ITEMS} more items truncated>")
    return out_list


def _serialize_container(
    value: dict[Any, Any] | list[Any] | tuple[Any, ...],
    *,
    _depth: int,
    _seen: frozenset[int],
) -> Any:
    """Dispatch a dict/list/tuple to its dedicated serializer.

    Split out purely to keep _deep_serialize's own branch/return count low.
    """
    if isinstance(value, dict):
        return _serialize_mapping(value, _depth=_depth, _seen=_seen)
    return _serialize_sequence(value, _depth=_depth, _seen=_seen)


def _dump_via_attr(value: Any, attr_name: str) -> Any:
    """Call value.<attr_name>() if present and callable, else return None.

    Used for both the pydantic-v2 ``model_dump()`` shape and the pydantic-v1
    / dict-like ``dict()`` shape — a failure or absence of the method is not
    an error, just "this strategy doesn't apply to this object".
    """
    method = getattr(value, attr_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _deep_serialize(
    value: Any,
    *,
    _depth: int = 0,
    _seen: frozenset[int] = frozenset(),
) -> Any:
    """Recursively convert *value* into JSON-primitive-only data.

    Tries, in order, per non-primitive object: a pydantic-v2-style
    ``model_dump()`` (what ``langchain_core.messages.BaseMessage`` exposes),
    a ``dict()`` method (pydantic v1 / other dict-likes), and finally
    ``str()``. Bounded on both depth and collection size (rather than
    delegating recursion to ``json.dumps``'s own ``default`` callback) so a
    pathological or infinitely-self-generating object cannot spin this into
    a multi-thousand-frame stack walk — every recursive branch strictly
    increments ``_depth`` and stops at ``_MAX_SERIALIZE_DEPTH`` regardless of
    what the object's own methods return.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    obj_id = id(value)
    if obj_id in _seen:
        return "<circular-reference>"
    if _depth >= _MAX_SERIALIZE_DEPTH:
        return _safe_str(value)

    seen_here = _seen | {obj_id}

    if isinstance(value, (dict, list, tuple)):
        return _serialize_container(value, _depth=_depth, _seen=seen_here)

    for attr_name in ("model_dump", "dict"):
        dumped = _dump_via_attr(value, attr_name)
        if dumped is not None:
            return _deep_serialize(dumped, _depth=_depth + 1, _seen=seen_here)

    return _safe_str(value)


def _to_attr_string(value: Any, *, max_len: int = _MAX_ATTR_LEN) -> str:
    """Defensively serialize *value* into a bounded, span-safe string.

    Used for anything structured (message lists, chain inputs/outputs,
    metadata dicts, response_metadata, ...) that can't be stored as a raw
    span attribute directly.
    """
    try:
        text = json.dumps(_deep_serialize(value), ensure_ascii=False, default=str)
    except Exception:
        try:
            text = str(value)
        except Exception:
            text = "<unserializable>"
    if len(text) > max_len:
        text = text[:max_len] + "...<truncated>"
    return text


def _stringify(value: Any, *, max_len: int = _MAX_ATTR_LEN) -> str:
    """Turn *value* into a span-safe string.

    Plain strings are truncated as-is (so raw tool I/O isn't wrapped in JSON
    quotes); anything else goes through :func:`_to_attr_string`.
    """
    if isinstance(value, str):
        text = value
    else:
        return _to_attr_string(value, max_len=max_len)
    if len(text) > max_len:
        text = text[:max_len] + "...<truncated>"
    return text


# ---------------------------------------------------------------------------
# Runtime/context capture
# ---------------------------------------------------------------------------
#
# LangGraph injects a per-run ``Runtime``/context object into node functions
# via a private config key (``CONFIG_KEY_RUNTIME`` = "__pregel_runtime")
# that is never exposed to the public ``BaseCallbackHandler`` interface —
# on_chain_start only ever receives (serialized, inputs, run_id, ...), with
# no way to see the runtime object at all. The only way to capture it is to
# read it out of the RunnableConfig at the point LangGraph itself resolves
# it, which happens inside ``RunnableCallable.invoke``/``ainvoke`` — a
# private module. This is stashed into a ContextVar immediately before
# LangGraph calls into the callback manager, so LangGraphTracer.on_chain_start
# (which fires synchronously inside that same call) can read it back out.

_current_runtime: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "agent_trace_langgraph_runtime", default=None
)

_runtime_patch_lock = threading.Lock()
_runtime_patch_installed = False


def _install_runtime_capture_patch() -> None:
    """Best-effort monkeypatch that makes the LangGraph Runtime/context object
    observable to LangGraphTracer.

    This touches a private LangGraph module
    (``langgraph._internal._runnable``/``langgraph._internal._constants``)
    that may change shape across versions without notice. Every step here is
    wrapped so a mismatch degrades to "no runtime captured" (the pre-existing
    behavior) rather than breaking tracing or import of this module.
    """
    global _runtime_patch_installed  # noqa: PLW0603
    if _runtime_patch_installed:
        return
    with _runtime_patch_lock:
        if _runtime_patch_installed:
            return
        try:
            from langgraph._internal._constants import (
                CONF,
                CONFIG_KEY_RUNTIME,
            )
            from langgraph._internal._runnable import (
                RunnableCallable,
            )
        except Exception:
            logger.debug(
                "agent-trace: LangGraph Runtime-capture patch unavailable "
                "(internal module shape not as expected on this LangGraph "
                "version); Runtime/context objects will not be captured on "
                "spans.",
                exc_info=True,
            )
            _runtime_patch_installed = True  # don't retry every call
            return

        original_invoke = RunnableCallable.invoke
        original_ainvoke = RunnableCallable.ainvoke

        def _capture_runtime_from_config(config: Any) -> None:
            """Best-effort: stash config's Runtime object into the ContextVar.

            Swallows everything — a shape mismatch here must never break the
            actual LangGraph invocation it's piggybacking on.
            """
            try:
                if config is not None:
                    runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
                    if runtime is not None:
                        _current_runtime.set(runtime)
            except Exception:
                logger.debug(
                    "agent-trace: failed to read Runtime/context off config",
                    exc_info=True,
                )

        def _patched_invoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            _capture_runtime_from_config(config)
            return original_invoke(self, input, config, **kwargs)

        async def _patched_ainvoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            _capture_runtime_from_config(config)
            return await original_ainvoke(self, input, config, **kwargs)

        RunnableCallable.invoke = _patched_invoke  # type: ignore[method-assign]
        RunnableCallable.ainvoke = _patched_ainvoke  # type: ignore[method-assign]
        _runtime_patch_installed = True


# ---------------------------------------------------------------------------
# on_llm_end helpers — pulled out to keep the callback itself flat/scannable
# ---------------------------------------------------------------------------


def _first_generation_and_message(response: Any) -> tuple[Any, Any]:
    """Return (first_generation, first_generation.message) from a
    langchain-core LLMResult/ChatResult, or (None, None) if absent."""
    generations = getattr(response, "generations", None) or []
    if not generations or not generations[0]:
        return None, None
    first_gen = generations[0][0]
    first_message = getattr(first_gen, "message", None)
    return first_gen, first_message


def _extract_token_usage(response: Any, first_message: Any) -> dict[str, Any]:
    """token_usage from llm_output, falling back to the first message's
    usage_metadata — modern langchain-core often attaches usage there
    independently of llm_output, especially under streaming configurations.
    """
    usage = getattr(response, "llm_output", {}) or {}
    token_usage = usage.get("token_usage") or usage.get("usage") or {}
    if token_usage or first_message is None:
        return token_usage
    msg_usage = getattr(first_message, "usage_metadata", None)
    if not msg_usage:
        return {}
    return {
        "prompt_tokens": msg_usage.get("input_tokens", 0),
        "completion_tokens": msg_usage.get("output_tokens", 0),
        "total_tokens": msg_usage.get("total_tokens", 0),
    }


def _extract_finish_reason(
    response_metadata: dict[str, Any] | None,
    generation_info: dict[str, Any] | None,
) -> str | None:
    """finish_reason (or stop_reason) from response_metadata, falling back to
    generation_info — e.g. Gemini's finish_reason=MALFORMED_FUNCTION_CALL,
    OpenAI's finish_reason/tool_calls presence signal."""
    for source in (response_metadata, generation_info):
        if not source:
            continue
        reason = source.get("finish_reason") or source.get("stop_reason")
        if reason:
            return str(reason)
    return None


def _extract_content(first_message: Any, first_gen: Any) -> Any:
    """Actual generated content — not just usage counts.

    Chat models: response.generations[0][0].message.content
    Legacy completions: response.generations[0][0].text
    """
    if first_message is not None:
        return getattr(first_message, "content", None)
    if first_gen is not None:
        return getattr(first_gen, "text", None)
    return None


def _record_llm_end_data(span: Span, response: Any) -> None:
    """Extract and persist everything on_llm_end previously discarded:
    token usage (with the usage_metadata fallback), response_metadata/
    generation_info (finish_reason, tool-call presence), and the actual
    generated content."""
    first_gen, first_message = _first_generation_and_message(response)

    token_usage = _extract_token_usage(response, first_message)
    if token_usage:
        span.set_attribute(
            "llm.usage.prompt_tokens", int(token_usage.get("prompt_tokens", 0))
        )
        span.set_attribute(
            "llm.usage.completion_tokens",
            int(token_usage.get("completion_tokens", 0)),
        )
        span.set_attribute(
            "llm.usage.total_tokens", int(token_usage.get("total_tokens", 0))
        )

    response_metadata = (
        getattr(first_message, "response_metadata", None)
        if first_message is not None
        else None
    )
    generation_info = getattr(first_gen, "generation_info", None)

    finish_reason = _extract_finish_reason(response_metadata, generation_info)
    if finish_reason:
        span.set_attribute("llm.finish_reason", finish_reason)
    if first_message is not None:
        span.set_attribute(
            "llm.has_tool_calls", bool(getattr(first_message, "tool_calls", None))
        )
    if response_metadata:
        span.set_attribute("llm.response_metadata", _to_attr_string(response_metadata))
    if generation_info:
        span.set_attribute("llm.generation_info", _to_attr_string(generation_info))

    content = _extract_content(first_message, first_gen)
    if content:
        span.set_attribute("llm.content", _stringify(content))


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
        # Best-effort — makes the injected Runtime/context object observable
        # to on_chain_start below. No-ops safely if the private LangGraph
        # internals it depends on aren't present.
        _install_runtime_capture_patch()

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
                # LangGraph 1.x passes serialized=None; node name is in kwargs['name'].
                ser = serialized or {}
                node_name: str = (
                    kwargs.get("name")
                    or ser.get("name")
                    or (ser.get("id") or [None])[-1]
                    or "chain"
                )
                span = self._open_span(run_id, f"node:{node_name}", parent_run_id)
                span.set_attribute("langgraph.node", node_name)
                if tags:
                    span.set_attribute("langgraph.tags", ",".join(tags))
                if metadata:
                    span.set_attribute("chain.metadata", _to_attr_string(metadata))
                if inputs:
                    span.set_attribute("chain.inputs", _to_attr_string(inputs))
                # Best-effort Runtime/context capture — see
                # _install_runtime_capture_patch(). Deliberately kept out of
                # the "langgraph." attribute namespace since the Runtime
                # object (store/writer/context) is not guaranteed to
                # serialize identically across separate invocations of the
                # same graph (e.g. record vs. replay).
                runtime = _current_runtime.get()
                if runtime is not None:
                    span.set_attribute("chain.runtime", _to_attr_string(runtime))

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
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None and outputs:
                    try:
                        span.set_attribute("chain.outputs", _to_attr_string(outputs))
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record chain outputs for run %r",
                            str(run_id),
                            exc_info=True,
                        )
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
                ser = serialized or {}
                model_name: str = (
                    kwargs.get("name")
                    or (ser.get("kwargs") or {}).get("model_name")
                    or (ser.get("kwargs") or {}).get("model")
                    or ser.get("name")
                    or "llm"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", model_name)
                span.set_attribute("llm.prompt_count", len(prompts))
                if metadata:
                    span.set_attribute("llm.metadata", _to_attr_string(metadata))

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
                ser = serialized or {}
                model_name: str = (
                    kwargs.get("name")
                    or (ser.get("kwargs") or {}).get("model_name")
                    or (ser.get("kwargs") or {}).get("model")
                    or ser.get("name")
                    or "unknown"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", model_name)
                if messages:
                    span.set_attribute("llm.messages", _to_attr_string(messages))
                if metadata:
                    span.set_attribute("llm.metadata", _to_attr_string(metadata))

            def on_llm_end(
                self,
                response: Any,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the LLM span, attaching token usage and response content
                when available."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None:
                    try:
                        _record_llm_end_data(span, response)
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record LLM response data "
                            "for run %r",
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
                ser = serialized or {}
                tool_name: str = kwargs.get("name") or ser.get("name") or "tool"
                span = self._open_span(run_id, f"tool:{tool_name}", parent_run_id)
                span.set_attribute("tool.name", tool_name)
                if input_str:
                    span.set_attribute("tool.input", _stringify(input_str))
                if metadata:
                    span.set_attribute("tool.metadata", _to_attr_string(metadata))
                # Threading/event-loop context — the exact structured data
                # needed to diagnose a sync tool dispatched into a
                # ThreadPoolExecutor with no event loop (RuntimeError: "There
                # is no current event loop in thread ...").
                span.set_attribute("tool.thread_name", threading.current_thread().name)
                try:
                    asyncio.get_running_loop()
                    span.set_attribute("tool.has_event_loop", True)
                except RuntimeError:
                    span.set_attribute("tool.has_event_loop", False)

            def on_tool_end(
                self,
                output: str,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the tool span with OK status, recording output text."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None and output is not None:
                    try:
                        span.set_attribute("tool.output", _stringify(output))
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record tool output for run %r",
                            str(run_id),
                            exc_info=True,
                        )
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
