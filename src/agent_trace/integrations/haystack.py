"""
Haystack integration — implements Haystack's native ``tracing.Tracer`` /
``tracing.Span`` interface so ``Pipeline.run()`` and ``Component.run()``
execution is captured as agent-trace spans.

Haystack 2.x ships its own instrumentation surface
(``haystack.tracing.Tracer`` / ``haystack.tracing.Span``, see
``haystack/tracing/tracer.py``) that ``Pipeline.run()``
(``haystack/core/pipeline/pipeline.py``) and ``PipelineBase._run_component``
(``haystack/core/pipeline/base.py``) call directly — there is no callback
list to pass in, unlike LangGraph.  A tracer implementation is registered
globally via ``haystack.tracing.enable_tracing(...)`` and every pipeline/
component run after that point is traced through it until
``haystack.tracing.disable_tracing()`` is called.

Two spans are produced per pipeline run:

- ``haystack.pipeline.run`` — one span for the whole ``Pipeline.run()`` call,
  tagged with the pipeline's input/output data and metadata.
- ``haystack.component.run`` — one span per component invocation (nested
  under the pipeline span), tagged with the component's name, type, and I/O
  socket spec.  When Haystack's own content tracing is enabled (see below)
  the component's *actual received arguments* and *actual returned output*
  are attached too — this is the exact capability gap issue #4574 exposes:
  a caller-to-callee argument-propagation bug that occurs entirely
  in-process, before any HTTP request, so agent-trace's httpx/requests
  interceptor is structurally the wrong layer to catch it.

Haystack gates raw input/output content behind its own
``HAYSTACK_CONTENT_TRACING_ENABLED`` environment variable (default
``false``) because pipeline inputs/outputs can contain arbitrary user
content — set it to ``"true"`` (or call
``haystack.tracing.tracer.tracer.is_content_tracing_enabled = True``) to
capture that content in agent-trace spans as well.

Usage::

    import haystack.tracing
    from agent_trace import Tracer
    from agent_trace.integrations.haystack import HaystackTracer

    t = Tracer()
    with t.start_trace("my_pipeline", record=True) as trace:
        haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
        try:
            result = pipeline.run({"component_name": {"value": "hello"}})
        finally:
            haystack.tracing.disable_tracing()
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agent_trace import Trace, Tracer

__all__ = ["HaystackTracer"]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "HaystackTracer requires haystack-ai.\n"
    "Install it with:\n\n"
    "    pip install haystack-ai\n"
)

# agent_trace.core.span.Span attributes only accept str | int | float | bool;
# Haystack tags/content-tags are frequently dicts, lists, dataclasses, or
# Haystack domain objects (Document, ChatMessage, ...).  Truncate the
# stringified form so one component's raw output can't blow up trace.json.
_MAX_TAG_LEN = 4000


def _require_haystack_tracing() -> Any:
    """Lazy import guard — raises a clear error if haystack-ai is absent."""
    try:
        import haystack.tracing

        return haystack.tracing
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _coerce_tag_value(value: Any) -> str | int | float | bool:
    """Coerce an arbitrary Haystack tag value to a Span-attribute-safe primitive."""
    if isinstance(value, str | int | float | bool):
        return value
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepr-able {type(value).__name__}>"
    if len(text) > _MAX_TAG_LEN:
        text = text[:_MAX_TAG_LEN] + "...<truncated>"
    return text


# Module-level singletons: the concrete Tracer/Span implementations are built
# once (the first time _get_classes() is called) so that Haystack's
# tracing.Tracer / tracing.Span are real bases at class-definition time
# rather than spliced in later.
_HaystackTracerImplClass: type | None = None
_HaystackSpanImplClass: type | None = None
_classes_lock: threading.Lock = threading.Lock()


def _get_classes() -> tuple[type, type]:
    """Return (and lazily build) the concrete Tracer/Span implementations."""
    global _HaystackTracerImplClass, _HaystackSpanImplClass  # noqa: PLW0603
    if _HaystackTracerImplClass is not None and _HaystackSpanImplClass is not None:
        return _HaystackTracerImplClass, _HaystackSpanImplClass

    with _classes_lock:
        if _HaystackTracerImplClass is not None and _HaystackSpanImplClass is not None:
            return _HaystackTracerImplClass, _HaystackSpanImplClass

        htracing = _require_haystack_tracing()
        base_span_cls: type = htracing.Span
        base_tracer_cls: type = htracing.Tracer

        class _AgentTraceHaystackSpan(base_span_cls):  # type: ignore[misc]
            """Wraps an agent_trace Span behind Haystack's Span interface."""

            def __init__(self, agent_span: Span) -> None:
                self._agent_span = agent_span

            def set_tag(self, key: str, value: Any) -> None:
                self._agent_span.set_attribute(key, _coerce_tag_value(value))

            def raw_span(self) -> Any:
                """Expose the underlying agent_trace Span for direct access."""
                return self._agent_span

        class _AgentTraceHaystackTracerImpl(base_tracer_cls):  # type: ignore[misc]
            """Concrete implementation — see HaystackTracer for public docs."""

            def __init__(self, tracer: Tracer, trace: Trace) -> None:
                self._tracer: Tracer = tracer
                self._trace: Trace = trace
                # LIFO stack of open agent_trace spans, used to infer the
                # parent span when Haystack doesn't pass one explicitly.
                self._span_stack: list[Span] = []
                self._lock: threading.Lock = threading.Lock()

            @contextlib.contextmanager
            def trace(
                self,
                operation_name: str,
                tags: dict[str, Any] | None = None,
                parent_span: Any | None = None,
            ) -> Iterator[Any]:
                """Open a span for *operation_name*, yield it, then close it.

                Honors an explicit *parent_span* (Haystack passes the
                pipeline-level span in when tracing a component run — see
                ``PipelineBase._create_component_span``); falls back to the
                innermost currently-open span on this tracer otherwise.
                """
                parent_id: str | None = None
                if parent_span is not None:
                    raw = getattr(parent_span, "raw_span", lambda: None)()
                    if isinstance(raw, Span):
                        parent_id = raw.span_id
                else:
                    with self._lock:
                        if self._span_stack:
                            parent_id = self._span_stack[-1].span_id

                agent_span = self._tracer.start_span(
                    operation_name, parent_id=parent_id
                )
                wrapped = _AgentTraceHaystackSpan(agent_span)
                if tags:
                    wrapped.set_tags(tags)

                with self._lock:
                    self._span_stack.append(agent_span)
                try:
                    yield wrapped
                except Exception as exc:
                    agent_span.record_exception(exc)
                    if agent_span.end_time is None:
                        agent_span.end(SpanStatus.ERROR)
                    raise
                else:
                    if agent_span.end_time is None:
                        agent_span.end(SpanStatus.OK)
                finally:
                    with self._lock:
                        if self._span_stack and self._span_stack[-1] is agent_span:
                            self._span_stack.pop()
                        else:
                            # Defensive: out-of-order close (shouldn't happen
                            # with well-behaved context managers) — drop it
                            # wherever it is rather than corrupting the stack.
                            with contextlib.suppress(ValueError):
                                self._span_stack.remove(agent_span)

            def current_span(self) -> Any | None:
                """Return the innermost currently-open span, or None."""
                with self._lock:
                    if not self._span_stack:
                        return None
                    return _AgentTraceHaystackSpan(self._span_stack[-1])

        _HaystackTracerImplClass = _AgentTraceHaystackTracerImpl
        _HaystackSpanImplClass = _AgentTraceHaystackSpan
        return _HaystackTracerImplClass, _HaystackSpanImplClass


class HaystackTracer:
    """Haystack ``tracing.Tracer`` implementation that emits agent-trace spans.

    Register an instance globally with ``haystack.tracing.enable_tracing(...)``
    before calling ``pipeline.run()`` / ``pipeline.run_async()``; every
    ``haystack.pipeline.run`` and ``haystack.component.run`` span Haystack
    creates internally is forwarded to agent-trace for the lifetime of the
    registration.  Call ``haystack.tracing.disable_tracing()`` (or register a
    fresh ``NullTracer``) when done so later pipeline runs outside the
    ``with tracer.start_trace(...)`` block aren't captured into a closed
    trace.

    ``haystack-ai`` is imported lazily — importing this module succeeds even
    when it is not installed.

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
    """

    def __new__(cls, tracer: Tracer, trace: Trace) -> HaystackTracer:
        # Construct the concrete impl directly so Python's normal
        # type.__call__ runs _AgentTraceHaystackTracerImpl.__init__
        # automatically.  See LangGraphTracer.__new__ for why this two-step
        # __new__/__init__ split is required rather than mutating __bases__.
        impl_cls, _ = _get_classes()
        return impl_cls(tracer, trace)  # type: ignore[no-any-return]

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        # __init__ is called on the instance whose __class__ is already the
        # concrete impl class (set by __new__).  This path is only reached if
        # someone subclasses HaystackTracer directly; normal construction
        # goes through the impl class __init__.
        pass  # pragma: no cover
