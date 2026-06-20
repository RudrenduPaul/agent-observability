"""
Plugin protocols and base class for the agent-trace Plugin SDK.

Two structural protocols define the plugin surface:

``SpanPlugin``
    Observes individual spans.  Implement ``on_span_start`` and/or
    ``on_span_end``.

``TracePlugin``
    Observes full traces.  Implement ``on_trace_start`` and/or
    ``on_trace_end``.

``Plugin``
    Union type — any object that satisfies SpanPlugin, TracePlugin, or both.

``PluginBase``
    Optional ABC with no-op defaults for all four hooks.  Inherit from it
    when you only want to override one or two hooks.

All hooks are synchronous and run on the calling thread/task.  Exceptions
raised inside a hook are caught and logged at WARNING level so a buggy plugin
never silences the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_trace.core.span import Span
    from agent_trace.core.trace import Trace

__all__ = [
    "Plugin",
    "PluginBase",
    "SpanPlugin",
    "TracePlugin",
]


@runtime_checkable
class SpanPlugin(Protocol):
    """Protocol for plugins that observe individual spans.

    Implement one or both methods.  Both default to a no-op if the
    implementing class does not define them.
    """

    def on_span_start(self, span: Span) -> None:
        """Called immediately after a span is created via ``Tracer.start_span``.

        The span's ``start_time`` is set; ``end_time`` is None.
        """
        ...  # pragma: no cover

    def on_span_end(self, span: Span) -> None:
        """Called immediately after ``span.end()`` is called.

        Both ``start_time`` and ``end_time`` are set; ``status`` is final.
        """
        ...  # pragma: no cover


@runtime_checkable
class TracePlugin(Protocol):
    """Protocol for plugins that observe full traces.

    Implement one or both methods.
    """

    def on_trace_start(self, trace: Trace) -> None:
        """Called when ``Tracer.start_trace`` context is entered.

        The trace has no spans yet; ``metadata["name"]`` is populated.
        """
        ...  # pragma: no cover

    def on_trace_end(self, trace: Trace) -> None:
        """Called after all spans are complete and ``trace.json`` is written.

        All spans are accessible; the trace is immutable from this point.
        """
        ...  # pragma: no cover


Plugin = SpanPlugin | TracePlugin
"""Type alias: any object that satisfies SpanPlugin, TracePlugin, or both."""


class PluginBase:
    """Convenience base class with no-op defaults for all plugin hooks.

    Inherit from this when you only need to override a subset of hooks::

        class MyPlugin(PluginBase):
            def on_span_end(self, span: Span) -> None:
                metrics.record(span.name, span.duration_ms)
    """

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        pass

    def on_trace_start(self, trace: Any) -> None:
        pass

    def on_trace_end(self, trace: Any) -> None:
        pass
