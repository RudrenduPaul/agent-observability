"""
Unit tests for the agent-trace Plugin SDK.

Verifies:
  - SpanPlugin / TracePlugin protocols (structural duck typing)
  - PluginBase convenience class
  - Tracer.add_plugin / remove_plugin
  - on_span_start / on_span_end / on_trace_start / on_trace_end hook delivery
  - Exception isolation — a buggy plugin must not raise to the caller
  - Multiple plugins called in registration order
  - No duplicate registration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_trace import PluginBase, SpanPlugin, TracePlugin, Tracer
from agent_trace.core.trace import Trace
from agent_trace.plugins import SpanPlugin as PluginSpanPlugin  # re-export parity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingPlugin(PluginBase):
    """Captures every hook call for assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def on_span_start(self, span: Any) -> None:
        self.calls.append(("on_span_start", span))

    def on_span_end(self, span: Any) -> None:
        self.calls.append(("on_span_end", span))

    def on_trace_start(self, trace: Any) -> None:
        self.calls.append(("on_trace_start", trace))

    def on_trace_end(self, trace: Any) -> None:
        self.calls.append(("on_trace_end", trace))


class BuggyPlugin(PluginBase):
    """Always raises in every hook — must not propagate."""

    def on_span_start(self, span: Any) -> None:
        raise RuntimeError("buggy on_span_start")

    def on_span_end(self, span: Any) -> None:
        raise RuntimeError("buggy on_span_end")

    def on_trace_start(self, trace: Any) -> None:
        raise RuntimeError("buggy on_trace_start")

    def on_trace_end(self, trace: Any) -> None:
        raise RuntimeError("buggy on_trace_end")


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_pluginbase_satisfies_span_plugin(self) -> None:
        assert isinstance(PluginBase(), SpanPlugin)

    def test_pluginbase_satisfies_trace_plugin(self) -> None:
        assert isinstance(PluginBase(), TracePlugin)

    def test_recording_plugin_satisfies_span_plugin(self) -> None:
        assert isinstance(RecordingPlugin(), SpanPlugin)

    def test_span_plugin_re_export_from_plugins_package(self) -> None:
        assert PluginSpanPlugin is SpanPlugin

    def test_duck_typed_object_satisfies_span_plugin(self) -> None:
        class Duck:
            def on_span_start(self, span: Any) -> None:
                pass

            def on_span_end(self, span: Any) -> None:
                pass

        assert isinstance(Duck(), SpanPlugin)


# ---------------------------------------------------------------------------
# add_plugin / remove_plugin
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def test_add_plugin_stores_plugin(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        assert p in t._plugins

    def test_add_plugin_no_duplicate(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        t.add_plugin(p)
        assert t._plugins.count(p) == 1

    def test_remove_plugin(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        t.remove_plugin(p)
        assert p not in t._plugins

    def test_remove_plugin_not_registered_is_noop(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.remove_plugin(p)  # must not raise


# ---------------------------------------------------------------------------
# Hook delivery — trace lifecycle
# ---------------------------------------------------------------------------


class TestTraceHooks:
    def test_on_trace_start_called(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("hook-test"):
            pass
        events = [e for e, _ in p.calls]
        assert "on_trace_start" in events

    def test_on_trace_end_called_after_context_exit(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("hook-test"):
            pass
        events = [e for e, _ in p.calls]
        assert "on_trace_end" in events

    def test_on_trace_start_before_on_trace_end(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("order-test"):
            pass
        events = [e for e, _ in p.calls]
        assert events.index("on_trace_start") < events.index("on_trace_end")

    def test_on_trace_start_receives_trace_object(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("trace-obj-test") as trace:
            pass
        start_arg = next(arg for ev, arg in p.calls if ev == "on_trace_start")
        assert start_arg is trace

    def test_on_trace_end_receives_trace_with_spans(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("end-spans-test"):
            s = t.start_span("my-span")
            s.end()
        end_arg = next(arg for ev, arg in p.calls if ev == "on_trace_end")
        assert isinstance(end_arg, Trace)
        assert any(sp.name == "my-span" for sp in end_arg.spans)


# ---------------------------------------------------------------------------
# Hook delivery — span lifecycle
# ---------------------------------------------------------------------------


class TestSpanHooks:
    def test_on_span_start_called(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("span-start-test"):
            t.start_span("s")
        assert any(ev == "on_span_start" for ev, _ in p.calls)

    def test_on_span_end_called_after_span_end(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("span-end-test"):
            s = t.start_span("s")
            s.end()
        assert any(ev == "on_span_end" for ev, _ in p.calls)

    def test_on_span_start_receives_span_object(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("span-arg-test"):
            span = t.start_span("target")
        start_args = [arg for ev, arg in p.calls if ev == "on_span_start"]
        assert any(a.name == "target" for a in start_args)

    def test_on_span_end_called_once_per_span(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        p = RecordingPlugin()
        t.add_plugin(p)
        with t.start_trace("multi-span"):
            for i in range(3):
                s = t.start_span(f"s{i}")
                s.end()
        end_events = [ev for ev, _ in p.calls if ev == "on_span_end"]
        assert len(end_events) == 3

    def test_span_end_time_set_before_on_span_end(self, tmp_path: Path) -> None:
        end_times: list[float | None] = []

        class CheckPlugin(PluginBase):
            def on_span_end(self, span: Any) -> None:
                end_times.append(span.end_time)

        t = Tracer(trace_dir=tmp_path)
        t.add_plugin(CheckPlugin())
        with t.start_trace("end-time-test"):
            s = t.start_span("s")
            s.end()

        assert len(end_times) == 1
        assert end_times[0] is not None


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


class TestPluginExceptionIsolation:
    def test_buggy_on_trace_start_does_not_propagate(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        t.add_plugin(BuggyPlugin())
        with t.start_trace("buggy"):  # must not raise
            pass

    def test_buggy_on_span_end_does_not_propagate(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        t.add_plugin(BuggyPlugin())
        with t.start_trace("buggy"):
            s = t.start_span("s")
            s.end()  # must not raise

    def test_later_plugins_still_called_after_buggy_plugin(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)
        t.add_plugin(BuggyPlugin())
        p2 = RecordingPlugin()
        t.add_plugin(p2)
        with t.start_trace("buggy-then-good"):
            pass
        assert any(ev == "on_trace_start" for ev, _ in p2.calls)


# ---------------------------------------------------------------------------
# Multiple plugins — registration order
# ---------------------------------------------------------------------------


class TestMultiplePlugins:
    def test_plugins_called_in_registration_order(self, tmp_path: Path) -> None:
        order: list[str] = []

        class P1(PluginBase):
            def on_trace_start(self, trace: Any) -> None:
                order.append("P1")

        class P2(PluginBase):
            def on_trace_start(self, trace: Any) -> None:
                order.append("P2")

        t = Tracer(trace_dir=tmp_path)
        t.add_plugin(P1())
        t.add_plugin(P2())
        with t.start_trace("order"):
            pass
        assert order == ["P1", "P2"]
