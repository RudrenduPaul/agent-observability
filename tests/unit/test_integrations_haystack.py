"""
Unit tests for agent_trace.integrations.haystack.HaystackTracer.

haystack-ai is NOT an installed test dependency in CI — these tests inject a
fake haystack.tracing module (mirroring the real Tracer/Span ABC interface
in haystack/tracing/tracer.py) so they can run without the real package.
Real-package coverage lives in tests/integration/test_haystack.py.
"""

from __future__ import annotations

import abc
import contextlib
import sys
import threading
import types
from pathlib import Path
from types import ModuleType

import pytest

from agent_trace import SpanStatus, Tracer

# ---------------------------------------------------------------------------
# Fake haystack.tracing fixture (module-level injection)
# ---------------------------------------------------------------------------


def _make_fake_haystack_tracing() -> dict[str, ModuleType]:
    """Return a sys.modules patch mirroring haystack/tracing/tracer.py's ABCs."""

    class FakeSpan(abc.ABC):
        @abc.abstractmethod
        def set_tag(self, key, value):
            pass

        def set_tags(self, tags):
            for key, value in tags.items():
                self.set_tag(key, value)

        def raw_span(self):
            return self

    class FakeTracer(abc.ABC):
        @abc.abstractmethod
        @contextlib.contextmanager
        def trace(self, operation_name, tags=None, parent_span=None):
            pass

        @abc.abstractmethod
        def current_span(self):
            pass

    fake_tracer_mod = types.ModuleType("haystack.tracing.tracer")
    fake_tracer_mod.Span = FakeSpan  # type: ignore[attr-defined]
    fake_tracer_mod.Tracer = FakeTracer  # type: ignore[attr-defined]

    fake_tracing = types.ModuleType("haystack.tracing")
    fake_tracing.Span = FakeSpan  # type: ignore[attr-defined]
    fake_tracing.Tracer = FakeTracer  # type: ignore[attr-defined]
    fake_tracing.tracer = fake_tracer_mod  # type: ignore[attr-defined]

    fake_haystack = types.ModuleType("haystack")
    fake_haystack.tracing = fake_tracing  # type: ignore[attr-defined]

    return {
        "haystack": fake_haystack,
        "haystack.tracing": fake_tracing,
        "haystack.tracing.tracer": fake_tracer_mod,
    }


@pytest.fixture()
def patched_haystack(monkeypatch):
    """Inject a fake haystack.tracing into sys.modules for the duration of a test."""
    fakes = _make_fake_haystack_tracing()
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Force _get_classes to rebuild (it caches the concrete classes once).
    import agent_trace.integrations.haystack as hs_module

    original_tracer_cls = hs_module._HaystackTracerImplClass
    original_span_cls = hs_module._HaystackSpanImplClass
    hs_module._HaystackTracerImplClass = None
    hs_module._HaystackSpanImplClass = None

    yield fakes

    # Restore cached classes so other tests (real haystack, integration) are
    # not affected by the reset.
    hs_module._HaystackTracerImplClass = original_tracer_cls
    hs_module._HaystackSpanImplClass = original_span_cls


@pytest.fixture()
def tracer_and_trace(tmp_path: Path, patched_haystack):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("haystack-unit-test") as trace:
        yield t, trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracer(t, trace):
    from agent_trace.integrations.haystack import HaystackTracer

    return HaystackTracer(tracer=t, trace=trace)


# ---------------------------------------------------------------------------
# Initialisation — mirrors the __new__ / __init__ wiring pattern from
# LangGraphTracer's own test suite.
# ---------------------------------------------------------------------------


class TestHaystackTracerInit:
    def test_tracer_attribute_is_set(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        assert h._tracer is t

    def test_trace_attribute_is_set(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        assert h._trace is trace

    def test_span_stack_starts_empty(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        assert h._span_stack == []

    def test_lock_is_a_lock(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        assert isinstance(h._lock, threading.Lock)

    def test_two_instances_have_independent_stacks(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h1 = _make_tracer(t, trace)
        h2 = _make_tracer(t, trace)
        assert h1._span_stack is not h2._span_stack

    def test_raises_clear_error_without_haystack(self, tmp_path, monkeypatch):
        """Importing haystack must succeed even when haystack-ai is absent.

        A ``None`` entry in sys.modules forces the next ``import haystack``
        to raise ImportError (per the import system's documented behavior),
        which simulates "package not installed" even when the real
        haystack-ai package is present on this machine's venv.
        """
        monkeypatch.setitem(sys.modules, "haystack", None)
        monkeypatch.setitem(sys.modules, "haystack.tracing", None)

        import agent_trace.integrations.haystack as hs_module

        original = hs_module._HaystackTracerImplClass
        hs_module._HaystackTracerImplClass = None
        hs_module._HaystackSpanImplClass = None
        try:
            t = Tracer(trace_dir=tmp_path)
            with t.start_trace("no-haystack") as trace, pytest.raises(ImportError, match="pip install"):
                hs_module.HaystackTracer(tracer=t, trace=trace)
        finally:
            hs_module._HaystackTracerImplClass = original


# ---------------------------------------------------------------------------
# trace() context manager — span lifecycle
# ---------------------------------------------------------------------------


class TestHaystackTracerTrace:
    def test_trace_opens_and_closes_span_ok(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("haystack.pipeline.run"):
            pass
        assert len(trace.spans) == 1
        assert trace.spans[0].status == SpanStatus.OK
        assert trace.spans[0].end_time is not None

    def test_trace_sets_tags_as_span_attributes(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("haystack.component.run", tags={"haystack.component.name": "retriever"}):
            pass
        assert trace.spans[0].attributes.get("haystack.component.name") == "retriever"

    def test_trace_coerces_non_primitive_tag_values(self, tracer_and_trace):
        """Dicts/lists must be stringified — agent_trace Span only allows primitives."""
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("haystack.pipeline.run", tags={"haystack.pipeline.input_data": {"q": "hi"}}):
            pass
        value = trace.spans[0].attributes.get("haystack.pipeline.input_data")
        assert isinstance(value, str)
        assert "hi" in value

    def test_trace_truncates_huge_tag_values(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        huge = "x" * 10_000
        with h.trace("haystack.component.run", tags={"haystack.component.output": huge}):
            pass
        # Strings are passed through untouched (already a primitive) —
        # truncation only applies to non-primitive (repr'd) values. Confirm
        # a non-primitive huge payload IS truncated.
        with h.trace("haystack.component.run", tags={"payload": {"data": "y" * 10_000}}):
            pass
        value = trace.spans[-1].attributes.get("payload")
        assert len(value) <= 4100
        assert "truncated" in value

    def test_trace_marks_error_status_on_exception(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with pytest.raises(ValueError, match="boom"), h.trace("haystack.component.run"):
            raise ValueError("boom")
        assert trace.spans[0].status == SpanStatus.ERROR

    def test_trace_records_exception_event(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with pytest.raises(RuntimeError), h.trace("haystack.component.run"):
            raise RuntimeError("component crashed")
        events = trace.spans[0].events
        assert any(e.name == "exception" for e in events)

    def test_nested_trace_wires_parent_child_via_stack(self, tracer_and_trace):
        """No explicit parent_span passed — falls back to the open-span stack."""
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("haystack.pipeline.run"):
            with h.trace("haystack.component.run"):
                pass
        pipeline_span = next(s for s in trace.spans if s.name == "haystack.pipeline.run")
        component_span = next(s for s in trace.spans if s.name == "haystack.component.run")
        assert component_span.parent_id == pipeline_span.span_id

    def test_explicit_parent_span_takes_precedence(self, tracer_and_trace):
        """Haystack passes parent_span explicitly for component spans; honor it."""
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("root") as root_span:
            with h.trace("unrelated"):
                # Explicitly parent "child" under root_span, even though
                # "unrelated" is the innermost open span on the stack.
                with h.trace("child", parent_span=root_span) as child_span:
                    assert child_span.raw_span().parent_id == root_span.raw_span().span_id

    def test_span_stack_is_empty_after_all_spans_close(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("a"):
            with h.trace("b"):
                pass
        assert h._span_stack == []

    def test_span_stack_unwinds_correctly_on_exception(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with pytest.raises(ValueError):
            with h.trace("a"):
                with h.trace("b"):
                    raise ValueError("boom")
        assert h._span_stack == []


# ---------------------------------------------------------------------------
# current_span()
# ---------------------------------------------------------------------------


class TestHaystackTracerCurrentSpan:
    def test_current_span_none_outside_trace(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        assert h.current_span() is None

    def test_current_span_returns_open_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("outer") as outer_span:
            current = h.current_span()
            assert current.raw_span() is outer_span.raw_span()

    def test_current_span_tracks_innermost_nested_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("outer"):
            with h.trace("inner") as inner_span:
                assert h.current_span().raw_span() is inner_span.raw_span()

    def test_current_span_none_after_all_close(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("a"):
            pass
        assert h.current_span() is None


# ---------------------------------------------------------------------------
# Span.set_tag / set_tags
# ---------------------------------------------------------------------------


class TestHaystackSpan:
    def test_set_tag_writes_through_to_agent_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("op") as span:
            span.set_tag("k", "v")
        assert trace.spans[0].attributes.get("k") == "v"

    def test_set_tags_writes_multiple(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("op") as span:
            span.set_tags({"a": 1, "b": 2})
        assert trace.spans[0].attributes.get("a") == 1
        assert trace.spans[0].attributes.get("b") == 2

    def test_raw_span_returns_underlying_agent_span(self, tracer_and_trace):
        from agent_trace.core.span import Span as AgentSpan

        t, trace = tracer_and_trace
        h = _make_tracer(t, trace)
        with h.trace("op") as span:
            assert isinstance(span.raw_span(), AgentSpan)
