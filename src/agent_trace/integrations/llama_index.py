"""
llama_index integration — Dispatcher-based span/event capture.

llama_index does not accept a per-call ``callbacks=[...]`` list the way
LangGraph or the OpenAI Agents SDK do.  Instead it exposes a global
instrumentation surface, ``llama_index.core.instrumentation``, built around a
tree of ``Dispatcher`` objects (one per module) that fan out to attached
``BaseSpanHandler``/``BaseEventHandler`` instances.  Every method decorated
with ``@dispatcher.span`` — which covers essentially all public entry points
on ``BaseQueryEngine``, ``BaseRetriever``, ``BaseChatEngine``, ``BaseLLM``,
``BaseTool``, agent workers/workflows, etc., via the ``DispatcherSpanMixin`` —
fires ``span_enter``/``span_exit``/``span_drop`` on every handler reachable
from the dispatcher tree's root.  Semantic events (``LLMChatStartEvent``,
``AgentToolCallEvent``, ``ExceptionEvent``, ...) are fired independently and
carry the ``span_id`` of whatever span was active when they were dispatched,
so a ``BaseEventHandler`` can attribute them back to the span that produced
them.

``LlamaIndexTracer`` implements both interfaces and, once installed on a
dispatcher (the root dispatcher by default — every leaf dispatcher's parent
chain terminates there, so this covers the whole library), turns every
llama_index span into an agent-trace ``Span`` with the correct parent/child
nesting, and enriches those spans with chat-history / tool-call / agent-step
data pulled out of the semantic events.

Usage (context manager — recommended, scopes install/uninstall to the block):

    from agent_trace import tracer
    from agent_trace.integrations.llama_index import LlamaIndexTracer

    with tracer.start_trace("my_query_engine", record=True) as trace:
        with LlamaIndexTracer(tracer=tracer, trace=trace):
            response = query_engine.query("What is agent-trace?")

Usage (manual install/uninstall — e.g. a long-lived process):

    li_tracer = LlamaIndexTracer(tracer=tracer, trace=trace)
    li_tracer.install()
    ...
    li_tracer.uninstall()
"""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus
from agent_trace.interceptor.httpx_hook import pop_correlation_id, push_correlation_id

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = ["LlamaIndexTracer"]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LlamaIndexTracer requires llama-index-core.\n"
    "Install it with:\n\n"
    "    pip install llama-index-core\n"
)

# ``Dispatcher.span()`` mints ids as "{ClassName}.{method_name}-{uuid4}" for
# instance methods, or "{func.__qualname__}-{uuid4}" for plain functions
# (see llama_index_instrumentation.dispatcher.Dispatcher.span). Stripping the
# trailing uuid4 gives a stable, human-readable span name.
_UUID_SUFFIX_RE = re.compile(
    r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_TRUNCATE_LEN = 2000


def _span_name(id_: str) -> str:
    """Derive a human-readable span name from a dispatcher span id."""
    stripped = _UUID_SUFFIX_RE.sub("", id_)
    return stripped or id_


def _truncate(value: Any) -> str:
    """Stringify and bound *value* so a single field can't blow up a span."""
    text = str(value)
    if len(text) <= _TRUNCATE_LEN:
        return text
    return text[:_TRUNCATE_LEN] + "...<truncated>"


def _require_llama_index() -> Any:
    """Lazy import guard — raises a clear error if llama-index-core is absent."""
    try:
        import llama_index.core

        return llama_index.core
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _apply_llm_chat_start(span: Span, event: Any) -> None:
    messages = getattr(event, "messages", None) or []
    span.set_attribute("llm.messages_count", len(messages))
    model_dict = getattr(event, "model_dict", None) or {}
    model = model_dict.get("model") or model_dict.get("model_name")
    if model:
        span.set_attribute("llm.model", str(model))
    if messages:
        last = messages[-1]
        span.set_attribute("llm.last_message_role", str(getattr(last, "role", "")))
        span.set_attribute(
            "llm.last_message_content", _truncate(getattr(last, "content", ""))
        )


def _apply_llm_chat_end(span: Span, event: Any) -> None:
    response = getattr(event, "response", None)
    message = getattr(response, "message", None) if response is not None else None
    if message is None:
        return
    span.set_attribute(
        "llm.response_content", _truncate(getattr(message, "content", ""))
    )
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    span.set_attribute("llm.has_tool_calls", bool(additional_kwargs.get("tool_calls")))


def _apply_llm_completion_start(span: Span, event: Any) -> None:
    prompt = getattr(event, "prompt", None)
    if prompt is not None:
        span.set_attribute("llm.prompt", _truncate(prompt))


def _apply_llm_completion_end(span: Span, event: Any) -> None:
    response = getattr(event, "response", None)
    if response is not None:
        span.set_attribute(
            "llm.response_content", _truncate(getattr(response, "text", ""))
        )


def _apply_agent_tool_call(span: Span, event: Any) -> None:
    tool = getattr(event, "tool", None)
    tool_name = getattr(tool, "name", None) if tool is not None else None
    span.add_event(
        "tool_call",
        attributes={
            "tool.name": str(tool_name) if tool_name else "unknown",
            "tool.arguments": _truncate(getattr(event, "arguments", "")),
        },
    )


def _apply_agent_run_step_start(span: Span, event: Any) -> None:
    task_id = getattr(event, "task_id", None)
    step_input = getattr(event, "input", None)
    if task_id is not None:
        span.set_attribute("agent.task_id", str(task_id))
    if step_input is not None:
        span.set_attribute("agent.step_input", _truncate(step_input))


def _apply_agent_run_step_end(span: Span, event: Any) -> None:
    step_output = getattr(event, "step_output", None)
    if step_output is not None:
        span.set_attribute("agent.step_output", _truncate(step_output))


def _apply_exception(span: Span, event: Any) -> None:
    exc = getattr(event, "exception", None)
    if isinstance(exc, BaseException):
        span.record_exception(exc)


# Dispatch on event.class_name() (a plain string every BaseEvent subclass
# implements) rather than importing each concrete event class — this keeps
# the handler resilient to llama_index moving event classes between modules
# across versions, since only the shape (attribute names), not the import
# path, is relied upon.
_EVENT_HANDLERS: dict[str, Any] = {
    "LLMChatStartEvent": _apply_llm_chat_start,
    "LLMChatEndEvent": _apply_llm_chat_end,
    "LLMCompletionStartEvent": _apply_llm_completion_start,
    "LLMCompletionEndEvent": _apply_llm_completion_end,
    "AgentToolCallEvent": _apply_agent_tool_call,
    "AgentRunStepStartEvent": _apply_agent_run_step_start,
    "AgentRunStepEndEvent": _apply_agent_run_step_end,
    "ExceptionEvent": _apply_exception,
}


def _apply_event(span: Span, event: Any) -> None:
    """Enrich *span* with data pulled out of a dispatched llama_index event."""
    handler = _EVENT_HANDLERS.get(event.class_name())
    if handler is not None:
        handler(span, event)


# ---------------------------------------------------------------------------
# Lazily-built concrete handler classes
# ---------------------------------------------------------------------------
#
# BaseSpanHandler / BaseEventHandler / BaseSpan are pydantic BaseModel
# subclasses defined by llama_index_instrumentation.  Subclassing them at
# module import time would require llama-index-core to be installed just to
# `import agent_trace.integrations.llama_index`.  Instead — same pattern as
# integrations/langgraph.py's _get_tracer_class() — the concrete impl classes
# are built exactly once, the first time a LlamaIndexTracer is constructed,
# with the real base classes as genuine bases.

_SpanHandlerClass: type | None = None
_EventHandlerClass: type | None = None
_classes_lock: threading.Lock = threading.Lock()


def _get_handler_classes() -> tuple[type, type]:
    """Return (and lazily build) the concrete span/event handler classes."""
    global _SpanHandlerClass, _EventHandlerClass  # noqa: PLW0603
    if _SpanHandlerClass is not None and _EventHandlerClass is not None:
        return _SpanHandlerClass, _EventHandlerClass

    with _classes_lock:
        if _SpanHandlerClass is not None and _EventHandlerClass is not None:
            return _SpanHandlerClass, _EventHandlerClass

        _require_llama_index()
        from llama_index.core.instrumentation.event_handlers import BaseEventHandler
        from llama_index.core.instrumentation.span.base import BaseSpan
        from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
        from pydantic import PrivateAttr

        class _AgentTraceSpan(BaseSpan):  # type: ignore[misc]
            """Minimal BaseSpan — the real agent-trace Span lives in the
            handler's own registry (keyed by the same dispatcher span id),
            not on this pydantic model."""

        class _AgentTraceSpanHandlerImpl(BaseSpanHandler[_AgentTraceSpan]):  # type: ignore[misc]
            """Turns dispatcher span_enter/span_exit/span_drop into agent-trace
            Spans, using the dispatcher-supplied parent_span_id for nesting."""

            _tracer: Any = PrivateAttr(default=None)
            _registry: dict[str, Span] = PrivateAttr(default_factory=dict)
            _registry_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
            # Per-span correlation-id contextvar tokens (#13449): every
            # span new_span() creates pushes its own span_id as the active
            # httpx_hook correlation id for the duration it's open, so any
            # HTTP exchange made anywhere inside it is recoverable
            # afterwards via Fixture.exchanges_for_correlation_id(span_id)
            # — the same mechanism LangGraphTracer uses for #6037. Same
            # documented scope/limitation: reliable for synchronous
            # dispatch, not guaranteed to propagate across llama_index's
            # own async/concurrent execution paths.
            _correlation_tokens: dict[str, Any] = PrivateAttr(default_factory=dict)

            @classmethod
            def class_name(cls) -> str:
                return "AgentTraceSpanHandler"

            def new_span(
                self,
                id_: str,
                bound_args: Any,
                instance: Any | None = None,
                parent_span_id: str | None = None,
                tags: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> _AgentTraceSpan:
                parent_id: str | None = None
                if parent_span_id is not None:
                    with self._registry_lock:
                        parent_span = self._registry.get(parent_span_id)
                    if parent_span is not None:
                        parent_id = parent_span.span_id

                span = self._tracer.start_span(_span_name(id_), parent_id=parent_id)
                span.set_attribute("llama_index.span_id", id_)
                if instance is not None:
                    span.set_attribute("llama_index.class", type(instance).__name__)
                with self._registry_lock:
                    self._registry[id_] = span
                    try:
                        self._correlation_tokens[id_] = push_correlation_id(
                            span.span_id
                        )
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to push correlation id for "
                            "llama_index span %r",
                            id_,
                            exc_info=True,
                        )
                return _AgentTraceSpan(
                    id_=id_, parent_id=parent_span_id, tags=tags or {}
                )

            def _pop_correlation_token(self, id_: str) -> None:
                with self._registry_lock:
                    token = self._correlation_tokens.pop(id_, None)
                if token is None:
                    return
                try:
                    pop_correlation_id(token)
                except Exception:
                    logger.debug(
                        "agent-trace: failed to pop correlation id for "
                        "llama_index span %r",
                        id_,
                        exc_info=True,
                    )

            def prepare_to_exit_span(
                self,
                id_: str,
                bound_args: Any,
                instance: Any | None = None,
                result: Any | None = None,
                **kwargs: Any,
            ) -> _AgentTraceSpan | None:
                with self._registry_lock:
                    span = self._registry.pop(id_, None)
                self._pop_correlation_token(id_)
                if span is not None and span.end_time is None:
                    span.end(SpanStatus.OK)
                return self.open_spans.get(id_)  # type: ignore[no-any-return]

            def prepare_to_drop_span(
                self,
                id_: str,
                bound_args: Any,
                instance: Any | None = None,
                err: BaseException | None = None,
                **kwargs: Any,
            ) -> _AgentTraceSpan | None:
                if id_ not in self.open_spans:
                    return None
                with self._registry_lock:
                    span = self._registry.pop(id_, None)
                self._pop_correlation_token(id_)
                if span is not None:
                    try:
                        if err is not None:
                            span.record_exception(err)
                        if span.end_time is None:
                            span.end(
                                SpanStatus.ERROR if err is not None else SpanStatus.OK
                            )
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to close dropped llama_index span %r",
                            id_,
                            exc_info=True,
                        )
                return self.open_spans.get(id_)  # type: ignore[no-any-return]

        class _AgentTraceEventHandlerImpl(BaseEventHandler):  # type: ignore[misc]
            """Routes semantic llama_index events onto the currently-open
            agent-trace span they belong to (matched via event.span_id)."""

            _registry: dict[str, Span] = PrivateAttr(default_factory=dict)
            _registry_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

            @classmethod
            def class_name(cls) -> str:
                return "AgentTraceEventHandler"

            def handle(self, event: Any, **kwargs: Any) -> None:
                span_id = getattr(event, "span_id", None)
                if not span_id:
                    return
                with self._registry_lock:
                    span = self._registry.get(span_id)
                if span is None:
                    return
                try:
                    _apply_event(span, event)
                except Exception:
                    logger.debug(
                        "agent-trace: failed to apply llama_index event %r "
                        "onto span %r",
                        getattr(event, "class_name", lambda: "?")(),
                        span_id,
                        exc_info=True,
                    )

        _SpanHandlerClass = _AgentTraceSpanHandlerImpl
        _EventHandlerClass = _AgentTraceEventHandlerImpl
        return _SpanHandlerClass, _EventHandlerClass


class LlamaIndexTracer:
    """Installs agent-trace span/event capture onto a llama_index Dispatcher.

    Unlike ``LangGraphTracer``/``AgentTraceHook``, this is not passed into a
    single call — llama_index's instrumentation is dispatcher-global, so this
    object is *installed* onto a dispatcher (the root dispatcher, by default)
    for the duration of the work you want captured, then *uninstalled*.

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
    """

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        span_handler_cls, event_handler_cls = _get_handler_classes()

        self._tracer: Tracer = tracer
        self._trace: Trace = trace
        # Shared registry: llama_index dispatcher span id -> open agent-trace
        # Span. The span handler populates/pops it; the event handler only
        # reads it, so events can be attributed to the span that produced
        # them (event.span_id matches the dispatcher span id verbatim).
        self._registry: dict[str, Span] = {}
        self._registry_lock: threading.Lock = threading.Lock()

        self._span_handler: Any = span_handler_cls()
        self._span_handler._tracer = tracer
        self._span_handler._registry = self._registry
        self._span_handler._registry_lock = self._registry_lock

        self._event_handler: Any = event_handler_cls()
        self._event_handler._registry = self._registry
        self._event_handler._registry_lock = self._registry_lock

        self._installed_on: Any = None

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------

    def install(self, dispatcher: Any = None) -> None:
        """Attach this tracer's handlers to *dispatcher* (root by default).

        The root dispatcher's parent chain is the terminus for every other
        dispatcher in the process (``Dispatcher.span_enter``/``event`` walk
        up ``c.parent`` until a non-propagating dispatcher is hit, and the
        root dispatcher is created with ``propagate=False``), so handlers
        attached to root see every span/event fired anywhere in llama_index.
        """
        if dispatcher is None:
            _require_llama_index()
            from llama_index.core.instrumentation import get_dispatcher

            dispatcher = get_dispatcher()

        dispatcher.add_span_handler(self._span_handler)
        dispatcher.add_event_handler(self._event_handler)
        self._installed_on = dispatcher

    def uninstall(self) -> None:
        """Detach this tracer's handlers from whatever dispatcher they were
        installed on. Safe to call even if never installed."""
        dispatcher = self._installed_on
        if dispatcher is None:
            return
        dispatcher.span_handlers = [
            h for h in dispatcher.span_handlers if h is not self._span_handler
        ]
        dispatcher.event_handlers = [
            h for h in dispatcher.event_handlers if h is not self._event_handler
        ]
        self._installed_on = None

    def __enter__(self) -> LlamaIndexTracer:
        self.install()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.uninstall()
