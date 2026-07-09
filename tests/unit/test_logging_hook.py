"""
Unit tests for agent_trace.interceptor.logging_hook.capture_logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent_trace import SpanStatus, Tracer
from agent_trace.interceptor.logging_hook import capture_logging


@pytest.fixture()
def tracer_and_trace(tmp_path: Path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("logging-hook-unit-test") as trace:
        yield t, trace


class TestCaptureLogging:
    def test_captures_warning_from_named_logger(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_logging(t, logger_names=["mylib"]) as span:
            logging.getLogger("mylib").warning("wrote to unknown channel X")
        assert span.status == SpanStatus.OK
        assert span.attributes.get("logging.captured_count") == 1
        events = [e for e in span.events if e.name == "runtime_log"]
        assert len(events) == 1
        assert events[0].attributes["log.level"] == "WARNING"
        assert events[0].attributes["log.logger"] == "mylib"
        assert "wrote to unknown channel X" in events[0].attributes["log.message"]

    def test_ignores_records_below_level(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_logging(t, logger_names=["mylib"], level=logging.WARNING) as span:
            logging.getLogger("mylib").info("routine info, not a warning")
        assert span.attributes.get("logging.captured_count") == 0
        assert span.events == []

    def test_ignores_unnamed_loggers(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_logging(t, logger_names=["mylib"]) as span:
            logging.getLogger("some_other_lib").warning("irrelevant warning")
        assert span.attributes.get("logging.captured_count") == 0

    def test_handler_removed_after_context_exits(self, tracer_and_trace):
        t, trace = tracer_and_trace
        target = logging.getLogger("mylib")
        handlers_before = list(target.handlers)
        with capture_logging(t, logger_names=["mylib"]):
            assert len(target.handlers) == len(handlers_before) + 1
        assert target.handlers == handlers_before

    def test_handler_removed_even_when_block_raises(self, tracer_and_trace):
        t, trace = tracer_and_trace
        target = logging.getLogger("mylib")
        handlers_before = list(target.handlers)
        with pytest.raises(ValueError):
            with capture_logging(t, logger_names=["mylib"]) as span:
                raise ValueError("boom")
        assert target.handlers == handlers_before
        assert span.status == SpanStatus.ERROR
        exception_events = [e for e in span.events if e.name == "exception"]
        assert exception_events
        assert exception_events[0].attributes["exception.type"] == "ValueError"

    def test_captures_multiple_named_loggers(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_logging(t, logger_names=["lib_a", "lib_b"]) as span:
            logging.getLogger("lib_a").warning("from a")
            logging.getLogger("lib_b").warning("from b")
        assert span.attributes.get("logging.captured_count") == 2

    def test_capped_event_count_still_counts_past_cap(self, tracer_and_trace):
        t, trace = tracer_and_trace
        target = logging.getLogger("hotloop")
        with capture_logging(t, logger_names=["hotloop"]) as span:
            for i in range(210):
                target.warning("spam %d", i)
        # Every record is counted...
        assert span.attributes.get("logging.captured_count") == 210
        # ...but SpanEvents are capped so the span doesn't grow unboundedly.
        events = [e for e in span.events if e.name == "runtime_log"]
        assert len(events) == 200

    def test_span_registered_on_trace(self, tracer_and_trace):
        t, trace = tracer_and_trace
        with capture_logging(t, logger_names=["mylib"]) as span:
            pass
        assert span in trace.spans
        assert span.name == "logging:capture"
