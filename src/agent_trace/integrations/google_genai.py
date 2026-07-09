"""
Google GenAI integration — provider-specific span enrichment for
``google.genai.client.Client`` and ``langchain_google_genai.ChatGoogleGenerativeAI``.

The generic ``httpx``/``requests`` interceptor (see
``agent_trace.interceptor.httpx_hook``) already captures the raw wire bytes for
most Google GenAI traffic (the current ``google-genai`` SDK's API-key auth path
constructs ``httpx.Client`` subclasses that go through the patched
``httpx.Client.__init__``).  What it does *not* do is surface the
provider-specific fields that actually explain a lot of Gemini bug reports —
``thinkingConfig``/``includeThoughts``/``thinkingBudget`` — as first-class span
attributes, or distinguish an LCEL-chain-routed call from a bare model
invocation.  This module adds that layer on top, without duplicating the
wire-level capture the interceptor already does.

Two independent entry points are provided:

``GoogleGenAITracer``
    A ``langchain_core`` ``BaseCallbackHandler`` for
    ``langchain_google_genai.ChatGoogleGenerativeAI`` (or any chain built on
    top of it).  Pass it in ``config={"callbacks": [...]}`` exactly like
    ``LangGraphTracer``.

``instrument_client`` / ``uninstrument_client``
    Instance-level patches for the raw ``google.genai.client.Client`` SDK
    (used directly by, e.g., crewAI's Gemini completion path) that wrap
    ``Client.models.generate_content`` / ``generate_content_stream`` to emit a
    span per call with the same provider-specific attributes.

Usage (LangChain / LCEL)::

    from langchain_google_genai import ChatGoogleGenerativeAI
    from agent_trace import tracer
    from agent_trace.integrations.google_genai import GoogleGenAITracer

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", thinking_budget=1024)
    with tracer.start_trace("gemini_run", record=True) as trace:
        cb = GoogleGenAITracer(tracer=tracer, trace=trace)
        llm.invoke("hello", config={"callbacks": [cb]})

Usage (raw SDK)::

    from google import genai
    from agent_trace import tracer
    from agent_trace.integrations.google_genai import instrument_client

    client = genai.Client(api_key="...")
    with tracer.start_trace("gemini_sdk_run", record=True) as trace:
        instrument_client(client, tracer=tracer, trace=trace)
        client.models.generate_content(model="gemini-2.5-flash", contents="hi")
"""

from __future__ import annotations

import logging
import threading
import types
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_trace import Trace, Tracer

__all__ = [
    "GoogleGenAITracer",
    "instrument_client",
    "uninstrument_client",
]

logger = logging.getLogger(__name__)

# Attribute value types accepted by Span.set_attribute.
_AttrValue = str | int | float | bool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_field(obj: Any, key: str) -> Any:
    """Read *key* off *obj*, whether it's a dict or an attribute-bearing object.

    ``google-genai``'s ``types`` module and ``langchain_google_genai`` both use
    pydantic models for config objects, but callers are free to pass plain
    dicts (``GenerateContentConfigDict``) too — this normalises both shapes.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# ---------------------------------------------------------------------------
# langchain_google_genai.ChatGoogleGenerativeAI — callback handler
# ---------------------------------------------------------------------------

_LANGCHAIN_INSTALL_HINT = (
    "GoogleGenAITracer requires langchain-core.\n"
    "Install it with:\n\n"
    "    pip install langchain-core\n"
    "\n"
    "(langchain-google-genai is not strictly required — this tracer works on\n"
    "any langchain_core callback stream and only enriches spans with Google\n"
    "GenAI-specific fields when they're present.)\n"
)


def _require_langchain_core() -> Any:
    """Lazy import guard — raises a clear error if langchain_core is absent."""
    try:
        import langchain_core

        return langchain_core
    except ImportError as exc:
        raise ImportError(_LANGCHAIN_INSTALL_HINT) from exc


def _base_callback_handler() -> Any:
    """Return langchain_core.callbacks.BaseCallbackHandler (lazy import)."""
    _require_langchain_core()
    from langchain_core.callbacks import BaseCallbackHandler

    return BaseCallbackHandler


def _extract_langchain_thinking_fields(
    ser_kwargs: dict[str, Any],
) -> dict[str, _AttrValue]:
    """Pull the Gemini thinking-config fields out of a serialized model's kwargs.

    ``on_chat_model_start``'s ``serialized["kwargs"]`` mirrors
    ``ChatGoogleGenerativeAI``'s constructor kwargs verbatim (confirmed via
    ``ChatGoogleGenerativeAI(...).to_json()`` against langchain-google-genai
    4.2.7) — ``thinking_budget``, ``thinking_level``, and ``include_thoughts``
    are flat top-level fields; ``thinking_config`` (when set instead of the
    flat fields) is a nested dict/``ThinkingConfig`` with the same three keys.
    """
    attrs: dict[str, _AttrValue] = {}

    include_thoughts = ser_kwargs.get("include_thoughts")
    thinking_budget = ser_kwargs.get("thinking_budget")
    thinking_level = ser_kwargs.get("thinking_level")
    thinking_config = ser_kwargs.get("thinking_config")

    if thinking_config:
        include_thoughts = (
            _get_field(thinking_config, "include_thoughts")
            if include_thoughts is None
            else include_thoughts
        )
        thinking_budget = (
            _get_field(thinking_config, "thinking_budget")
            if thinking_budget is None
            else thinking_budget
        )
        thinking_level = (
            _get_field(thinking_config, "thinking_level")
            if thinking_level is None
            else thinking_level
        )

    if include_thoughts is not None:
        attrs["google_genai.include_thoughts"] = bool(include_thoughts)
    if thinking_budget is not None:
        attrs["google_genai.thinking_budget"] = int(thinking_budget)
    if thinking_level is not None:
        attrs["google_genai.thinking_level"] = str(thinking_level)

    return attrs


def _extract_langchain_usage(response: Any) -> dict[str, _AttrValue]:
    """Extract token usage from an ``LLMResult``, preferring per-message
    ``usage_metadata`` (where Gemini's thoughts-token count lives) and
    falling back to the legacy ``llm_output['token_usage']`` shape used by
    other providers.
    """
    attrs: dict[str, _AttrValue] = {}

    usage_meta: Any = None
    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        message = getattr(generations[0][0], "message", None)
        if message is not None:
            usage_meta = getattr(message, "usage_metadata", None)

    if usage_meta:
        input_tokens = _get_field(usage_meta, "input_tokens")
        output_tokens = _get_field(usage_meta, "output_tokens")
        total_tokens = _get_field(usage_meta, "total_tokens")
        if input_tokens is not None:
            attrs["llm.usage.prompt_tokens"] = int(input_tokens)
        if output_tokens is not None:
            attrs["llm.usage.completion_tokens"] = int(output_tokens)
        if total_tokens is not None:
            attrs["llm.usage.total_tokens"] = int(total_tokens)

        output_details = _get_field(usage_meta, "output_token_details") or {}
        reasoning_tokens = _get_field(output_details, "reasoning")
        if reasoning_tokens:
            attrs["google_genai.usage.thoughts_tokens"] = int(reasoning_tokens)
        return attrs

    # Fallback: legacy llm_output.token_usage shape (non-Gemini providers).
    llm_output = getattr(response, "llm_output", {}) or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if token_usage:
        attrs["llm.usage.prompt_tokens"] = int(token_usage.get("prompt_tokens", 0))
        attrs["llm.usage.completion_tokens"] = int(
            token_usage.get("completion_tokens", 0)
        )
        attrs["llm.usage.total_tokens"] = int(token_usage.get("total_tokens", 0))
    return attrs


# Module-level singleton, built once — mirrors the LangGraphTracer pattern:
# BaseCallbackHandler is spliced in as a genuine base class at class-definition
# time (not via __bases__ mutation, which is unsafe under concurrency and
# forbidden in Python 3.14+).
_GoogleGenAITracerClass: type | None = None
_tracer_class_lock: threading.Lock = threading.Lock()


def _get_tracer_class() -> type:
    """Return (and lazily build) the concrete GoogleGenAITracer implementation."""
    global _GoogleGenAITracerClass  # noqa: PLW0603
    if _GoogleGenAITracerClass is not None:
        return _GoogleGenAITracerClass

    with _tracer_class_lock:
        if _GoogleGenAITracerClass is not None:
            return _GoogleGenAITracerClass

        base_cls: type = _base_callback_handler()

        class _GoogleGenAITracerImpl(base_cls):  # type: ignore[misc]
            """Concrete implementation — see GoogleGenAITracer for public docs."""

            def __init__(self, tracer: Tracer, trace: Trace) -> None:
                super().__init__()
                self._tracer: Tracer = tracer
                self._trace: Trace = trace
                self._spans: dict[str, Span] = {}
                self._lock: threading.Lock = threading.Lock()

            # ------------------------------------------------------------
            # Internal helpers
            # ------------------------------------------------------------

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
                self,
                run_id: uuid.UUID | str,
                status: SpanStatus = SpanStatus.OK,
            ) -> Span | None:
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
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.pop(run_key, None)
                if span is not None:
                    span.record_exception(error)
                    if span.end_time is None:
                        span.end(SpanStatus.ERROR)

            # ------------------------------------------------------------
            # Chain callbacks (generic — lets this tracer run standalone,
            # without LangGraphTracer, on a plain LCEL chain)
            # ------------------------------------------------------------

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
                ser = serialized or {}
                chain_name: str = (
                    kwargs.get("name")
                    or ser.get("name")
                    or (ser.get("id") or [None])[-1]
                    or "chain"
                )
                span = self._open_span(run_id, f"chain:{chain_name}", parent_run_id)
                span.set_attribute("chain.name", chain_name)

            def on_chain_end(
                self,
                outputs: dict[str, Any],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
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
                self._close_span_with_exception(run_id, error)

            # ------------------------------------------------------------
            # LLM callbacks
            # ------------------------------------------------------------

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
                """Start a span for a legacy (non-chat) LLM call."""
                ser = serialized or {}
                ser_kwargs = ser.get("kwargs") or {}
                model_name: str = (
                    ser_kwargs.get("model")
                    or kwargs.get("name")
                    or ser_kwargs.get("model_name")
                    or ser.get("name")
                    or "llm"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", model_name)
                span.set_attribute("llm.prompt_count", len(prompts))
                span.set_attribute(
                    "google_genai.invocation_context",
                    "lcel_chain" if parent_run_id is not None else "direct_invocation",
                )
                for key, value in _extract_langchain_thinking_fields(
                    ser_kwargs
                ).items():
                    span.set_attribute(key, value)

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
                """Start a span for a ChatGoogleGenerativeAI call.

                Captures ``thinkingConfig``/``includeThoughts``/``thinkingBudget``
                onto the span, and records whether this call was routed through
                an LCEL chain (``parent_run_id is not None``) or invoked bare
                (``parent_run_id is None`` — confirmed via a direct probe against
                ``langchain-core`` 1.4.9's callback manager: a bare
                ``llm.invoke(...)`` fires ``on_chat_model_start`` with
                ``parent_run_id=None``, while ``(prompt | llm | parser).invoke(...)``
                fires it with ``parent_run_id`` set to the chain's run id).
                """
                ser = serialized or {}
                ser_kwargs = ser.get("kwargs") or {}
                model_name: str = (
                    ser_kwargs.get("model")
                    or kwargs.get("name")
                    or ser_kwargs.get("model_name")
                    or ser.get("name")
                    or "unknown"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", model_name)
                span.set_attribute(
                    "google_genai.invocation_context",
                    "lcel_chain" if parent_run_id is not None else "direct_invocation",
                )
                for key, value in _extract_langchain_thinking_fields(
                    ser_kwargs
                ).items():
                    span.set_attribute(key, value)

            def on_llm_end(
                self,
                response: Any,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the LLM span, attaching token usage (incl. thoughts tokens)."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None:
                    try:
                        for key, value in _extract_langchain_usage(response).items():
                            span.set_attribute(key, value)
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
                self._close_span_with_exception(run_id, error)

            # ------------------------------------------------------------
            # Tool callbacks (generic passthrough — matches LangGraphTracer)
            # ------------------------------------------------------------

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
                ser = serialized or {}
                tool_name: str = kwargs.get("name") or ser.get("name") or "tool"
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
                self._close_span_with_exception(run_id, error)

        _GoogleGenAITracerClass = _GoogleGenAITracerImpl
        return _GoogleGenAITracerClass


class GoogleGenAITracer:
    """Langchain callback handler that emits Google-GenAI-enriched spans.

    Implements the ``BaseCallbackHandler`` interface from ``langchain_core``
    so it can be passed directly in the ``config["callbacks"]`` list of any
    LangChain-based call — bare ``ChatGoogleGenerativeAI.invoke()``, an LCEL
    chain, or a LangGraph node.  Unlike ``LangGraphTracer``, this tracer also
    captures Gemini-specific request fields (``thinking_config`` /
    ``include_thoughts`` / ``thinking_budget``) and marks each LLM span with
    ``google_genai.invocation_context`` so a "why did the chain path behave
    differently from the direct-model path" question (the shape of issue
    #31767) can be answered from span attributes alone.

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
    """

    def __new__(cls, tracer: Tracer, trace: Trace) -> GoogleGenAITracer:
        impl_cls = _get_tracer_class()
        return impl_cls(tracer, trace)  # type: ignore[no-any-return]

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        # Reached only if someone subclasses GoogleGenAITracer directly;
        # normal construction goes through the impl class __init__ (see
        # __new__ and the LangGraphTracer docstring for why).
        pass  # pragma: no cover


# ---------------------------------------------------------------------------
# google.genai.client.Client — raw SDK instrumentation
# ---------------------------------------------------------------------------

_GOOGLE_GENAI_INSTALL_HINT = (
    "instrument_client requires the google-genai package.\n"
    "Install it with:\n\n"
    "    pip install google-genai\n"
)


def _require_google_genai() -> Any:
    """Lazy import guard — raises a clear error if google-genai is absent."""
    try:
        import google.genai

        return google.genai
    except ImportError as exc:
        raise ImportError(_GOOGLE_GENAI_INSTALL_HINT) from exc


def _extract_sdk_thinking_fields(config: Any) -> dict[str, _AttrValue]:
    """Extract thinking-config fields from a ``GenerateContentConfig``.

    Handles both the pydantic ``GenerateContentConfig``/``ThinkingConfig``
    objects and their ``TypedDict`` equivalents (``GenerateContentConfigDict``
    accepts a plain dict for ``config``, confirmed against google-genai 2.10.0).
    """
    attrs: dict[str, _AttrValue] = {}
    thinking = _get_field(config, "thinking_config")
    if thinking is None:
        return attrs

    include_thoughts = _get_field(thinking, "include_thoughts")
    thinking_budget = _get_field(thinking, "thinking_budget")
    thinking_level = _get_field(thinking, "thinking_level")

    if include_thoughts is not None:
        attrs["google_genai.include_thoughts"] = bool(include_thoughts)
    if thinking_budget is not None:
        attrs["google_genai.thinking_budget"] = int(thinking_budget)
    if thinking_level is not None:
        attrs["google_genai.thinking_level"] = str(thinking_level)
    return attrs


def _extract_sdk_usage_fields(response: Any) -> dict[str, _AttrValue]:
    """Extract token usage from a ``GenerateContentResponse.usage_metadata``."""
    attrs: dict[str, _AttrValue] = {}
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return attrs

    prompt_tokens = _get_field(usage, "prompt_token_count")
    candidates_tokens = _get_field(usage, "candidates_token_count")
    total_tokens = _get_field(usage, "total_token_count")
    thoughts_tokens = _get_field(usage, "thoughts_token_count")

    if prompt_tokens is not None:
        attrs["llm.usage.prompt_tokens"] = int(prompt_tokens)
    if candidates_tokens is not None:
        attrs["llm.usage.completion_tokens"] = int(candidates_tokens)
    if total_tokens is not None:
        attrs["llm.usage.total_tokens"] = int(total_tokens)
    if thoughts_tokens is not None:
        attrs["google_genai.usage.thoughts_tokens"] = int(thoughts_tokens)
    return attrs


def _make_generate_content_wrapper(
    original: Callable[..., Any], tracer: Tracer
) -> Callable[..., Any]:
    def _generate_content(
        self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        span = tracer.start_span(f"google_genai.generate_content:{model}")
        span.set_attribute("llm.model", model)
        for key, value in _extract_sdk_thinking_fields(config).items():
            span.set_attribute(key, value)
        try:
            response = original(model=model, contents=contents, config=config, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)
            raise
        for key, value in _extract_sdk_usage_fields(response).items():
            span.set_attribute(key, value)
        if span.end_time is None:
            span.end(SpanStatus.OK)
        return response

    return _generate_content


def _make_generate_content_stream_wrapper(
    original: Callable[..., Any], tracer: Tracer
) -> Callable[..., Any]:
    def _generate_content_stream(
        self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any
    ) -> Iterator[Any]:
        span = tracer.start_span(f"google_genai.generate_content_stream:{model}")
        span.set_attribute("llm.model", model)
        for key, value in _extract_sdk_thinking_fields(config).items():
            span.set_attribute(key, value)

        last_chunk: Any = None
        try:
            stream = original(model=model, contents=contents, config=config, **kwargs)
            for chunk in stream:
                last_chunk = chunk
                yield chunk
        except Exception as exc:
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)
            raise
        else:
            # Gemini's streaming API returns cumulative usage_metadata with
            # each chunk, so the final chunk carries the full-request totals.
            if last_chunk is not None:
                for key, value in _extract_sdk_usage_fields(last_chunk).items():
                    span.set_attribute(key, value)
            if span.end_time is None:
                span.end(SpanStatus.OK)

    return _generate_content_stream


# Registry of instrumented Models instances -> their original bound methods,
# so uninstrument_client() can restore them.  Keyed by the Models instance
# (weakly) rather than the Client, since Client.models is the stable object
# whose methods we actually patch (Client() itself is a thin facade over it —
# confirmed against google-genai 2.10.0's Client.__init__, which builds
# self._models once and exposes it via a `models` property).
_instrumented_methods: WeakKeyDictionary[Any, dict[str, Any]] = WeakKeyDictionary()


def instrument_client(client: Any, *, tracer: Tracer, trace: Trace) -> None:
    """Patch *client*'s ``models.generate_content``/``generate_content_stream``
    to emit a span per call, enriched with thinking-config and usage fields.

    Idempotent: calling this twice on the same client is a no-op the second
    time.  Pair with :func:`uninstrument_client` to restore the originals.

    Parameters
    ----------
    client:
        A ``google.genai.client.Client`` (or ``google.genai.Client``) instance.
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on
        (must already be the active trace — spans are created via
        ``tracer.start_span``, which attaches to whatever trace is active).
    """
    _require_google_genai()

    models_obj = client.models
    if models_obj in _instrumented_methods:
        return

    originals = {
        "generate_content": models_obj.generate_content,
        "generate_content_stream": models_obj.generate_content_stream,
    }
    _instrumented_methods[models_obj] = originals

    models_obj.generate_content = types.MethodType(
        _make_generate_content_wrapper(originals["generate_content"], tracer),
        models_obj,
    )
    models_obj.generate_content_stream = types.MethodType(
        _make_generate_content_stream_wrapper(
            originals["generate_content_stream"], tracer
        ),
        models_obj,
    )


def uninstrument_client(client: Any) -> None:
    """Restore the original (unpatched) methods on *client*'s ``models``."""
    models_obj = client.models
    originals = _instrumented_methods.pop(models_obj, None)
    if originals is None:
        return
    models_obj.generate_content = originals["generate_content"]
    models_obj.generate_content_stream = originals["generate_content_stream"]
