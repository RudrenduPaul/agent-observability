"""
Unit tests for agent_trace.interceptor.warnings_hook.capture_warnings.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from agent_trace import SpanStatus, Tracer
from agent_trace.interceptor.warnings_hook import capture_warnings


@pytest.fixture()
def tracer_and_trace(tmp_path: Path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("warnings-hook-unit-test") as trace:
        yield t, trace


class TestCaptureWarnings:
    def test_captures_runtime_warning(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_warnings(t) as span:
            warnings.warn(
                "Failed to trim messages to fit within max_tokens limit",
                RuntimeWarning,
                stacklevel=2,
            )
        assert span.status == SpanStatus.OK
        assert span.attributes.get("warnings.captured_count") == 1
        events = [e for e in span.events if e.name == "runtime_warning"]
        assert len(events) == 1
        assert events[0].attributes["warning.category"] == "RuntimeWarning"
        assert "max_tokens" in events[0].attributes["warning.message"]

    def test_captures_user_warning_by_default(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_warnings(t) as span:
            warnings.warn("some user-facing warning", UserWarning, stacklevel=2)
        assert span.attributes.get("warnings.captured_count") == 1

    def test_ignores_category_not_in_filter(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_warnings(t, categories=(RuntimeWarning,)) as span:
            warnings.warn("not tracked", DeprecationWarning, stacklevel=2)
        assert span.attributes.get("warnings.captured_count") == 0

    def test_captures_every_occurrence_not_just_once(self, tracer_and_trace):
        """simplefilter('always') means repeated identical warnings from the
        same call site are all captured, unlike Python's default 'once per
        location' behavior."""
        t, trace = tracer_and_trace
        with capture_warnings(t) as span:
            for _ in range(3):
                warnings.warn("repeated warning", RuntimeWarning, stacklevel=2)
        assert span.attributes.get("warnings.captured_count") == 3

    def test_showwarning_restored_after_context_exits(self, tracer_and_trace):
        t, trace = tracer_and_trace
        original = warnings.showwarning
        with capture_warnings(t):
            assert warnings.showwarning is not original
        assert warnings.showwarning is original

    def test_showwarning_restored_even_when_block_raises(self, tracer_and_trace):
        t, trace = tracer_and_trace
        original = warnings.showwarning
        with pytest.raises(ValueError):
            with capture_warnings(t) as span:
                raise ValueError("boom")
        assert warnings.showwarning is original
        assert span.status == SpanStatus.ERROR
        exception_events = [e for e in span.events if e.name == "exception"]
        assert exception_events
        assert exception_events[0].attributes["exception.type"] == "ValueError"

    def test_filters_outside_the_block_are_unaffected(self, tracer_and_trace):
        """catch_warnings() saves/restores warnings.filters — a filter state
        change made outside this context manager's own window must survive
        it untouched."""
        t, trace = tracer_and_trace
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with capture_warnings(t):
                # 'always' inside the block overrides the outer 'error'
                # filter for the duration of the with-block only.
                warnings.warn("captured, not raised", RuntimeWarning, stacklevel=2)
            # Immediately after exiting capture_warnings(), the outer
            # 'error' filter is back in effect.
            with pytest.raises(RuntimeWarning):
                warnings.warn("this should now raise", RuntimeWarning, stacklevel=2)

    def test_capped_event_count_still_counts_past_cap(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_warnings(t) as span:
            for i in range(210):
                warnings.warn(f"spam {i}", RuntimeWarning, stacklevel=2)
        assert span.attributes.get("warnings.captured_count") == 210
        events = [e for e in span.events if e.name == "runtime_warning"]
        assert len(events) == 200

    def test_span_registered_on_trace(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_warnings(t) as span:
            pass
        assert span in trace.spans
        assert span.name == "warnings:capture"
