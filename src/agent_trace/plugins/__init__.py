"""
agent-trace Plugin SDK.

Third-party packages implement SpanPlugin and/or TracePlugin to observe
agent runs without modifying agent code.  Register plugins on the tracer
instance; they are called synchronously on every span and trace event.

Quick start::

    from agent_trace import tracer
    from agent_trace.plugins import SpanPlugin, TracePlugin
    from agent_trace.core.span import Span
    from agent_trace.core.trace import Trace

    class MyPlugin(SpanPlugin, TracePlugin):
        def on_span_end(self, span: Span) -> None:
            print(f"span {span.name} took {span.duration_ms:.1f}ms")

        def on_trace_end(self, trace: Trace) -> None:
            print(f"trace {trace.trace_id}: {len(trace.spans)} spans")

    tracer.add_plugin(MyPlugin())

See ``PluginBase`` for a convenience base class that provides no-op defaults
for every hook, so you only override what you need.
"""

from agent_trace.plugins.base import Plugin, PluginBase, SpanPlugin, TracePlugin

__all__ = [
    "Plugin",
    "PluginBase",
    "SpanPlugin",
    "TracePlugin",
]
