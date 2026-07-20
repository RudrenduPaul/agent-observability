"""
Unit tests for agent_trace.core.span — Span, SpanEvent, SpanStatus.

All time-related tests use FixtureClock so they never depend on wall time.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from agent_trace.core.clock import FixtureClock, restore_clock, set_clock
from agent_trace.core.span import Span, SpanEvent, SpanStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fixture_clock(ts: float = 1_000_000.0) -> tuple[FixtureClock, Any]:
    clock = FixtureClock()
    clock.advance(ts)
    token = set_clock(clock)
    return clock, token


# ---------------------------------------------------------------------------
# SpanStatus
# ---------------------------------------------------------------------------


class TestSpanStatus:
    def test_values_are_strings(self) -> None:
        assert SpanStatus.UNSET.value == "UNSET"
        assert SpanStatus.OK.value == "OK"
        assert SpanStatus.ERROR.value == "ERROR"
        assert SpanStatus.CANCELLED.value == "CANCELLED"

    def test_is_str_subclass(self) -> None:
        # SpanStatus(str, Enum) — each member IS a str
        assert isinstance(SpanStatus.OK, str)
        assert isinstance(SpanStatus.ERROR, str)
        assert isinstance(SpanStatus.CANCELLED, str)

    def test_four_values_only(self) -> None:
        assert len(SpanStatus) == 4


# ---------------------------------------------------------------------------
# SpanEvent
# ---------------------------------------------------------------------------


class TestSpanEvent:
    def test_to_dict_round_trip(self) -> None:
        event = SpanEvent(
            name="my-event",
            timestamp=1234.56,
            attributes={"key": "value", "count": 7},
        )
        d = event.to_dict()
        restored = SpanEvent.from_dict(d)
        assert restored.name == event.name
        assert restored.timestamp == event.timestamp
        assert restored.attributes == event.attributes

    def test_to_dict_keys(self) -> None:
        event = SpanEvent(name="e", timestamp=0.0)
        d = event.to_dict()
        assert set(d.keys()) == {"name", "timestamp", "attributes"}

    def test_from_dict_missing_attributes_defaults_to_empty(self) -> None:
        d = {"name": "e", "timestamp": 1.0}
        event = SpanEvent.from_dict(d)
        assert event.attributes == {}

    def test_to_dict_no_datetime_objects(self) -> None:
        """All values must be JSON-primitive types."""
        import datetime

        event = SpanEvent(name="e", timestamp=42.0, attributes={"x": 1})
        d = event.to_dict()
        for v in d.values():
            assert not isinstance(v, datetime.datetime)


# ---------------------------------------------------------------------------
# Span defaults
# ---------------------------------------------------------------------------


class TestSpanDefaults:
    def test_span_id_is_valid_uuid(self) -> None:
        span = Span()
        # Should not raise
        uuid.UUID(span.span_id)

    def test_trace_id_is_valid_uuid(self) -> None:
        span = Span()
        uuid.UUID(span.trace_id)

    def test_start_time_is_float(self) -> None:
        span = Span()
        assert isinstance(span.start_time, float)

    def test_default_status_is_unset(self) -> None:
        span = Span()
        assert span.status == SpanStatus.UNSET

    def test_default_parent_id_is_none(self) -> None:
        span = Span()
        assert span.parent_id is None

    def test_default_end_time_is_none(self) -> None:
        span = Span()
        assert span.end_time is None

    def test_default_attributes_is_empty_dict(self) -> None:
        span = Span()
        assert span.attributes == {}

    def test_default_events_is_empty_list(self) -> None:
        span = Span()
        assert span.events == []

    def test_start_time_uses_fixture_clock(self) -> None:
        """Span.start_time must come from get_time(), not time.time()."""
        clock, token = _install_fixture_clock(5000.0)
        try:
            span = Span()
            assert span.start_time == 5000.0
        finally:
            restore_clock(token)


# ---------------------------------------------------------------------------
# Span.end()
# ---------------------------------------------------------------------------


class TestSpanEnd:
    def test_end_sets_end_time(self) -> None:
        clock, token = _install_fixture_clock(1_000_000.0)
        try:
            span = Span()
            clock.advance(1_000_001.0)
            span.end()
            assert span.end_time == 1_000_001.0
        finally:
            restore_clock(token)

    def test_end_default_status_ok(self) -> None:
        span = Span()
        span.end()
        assert span.status == SpanStatus.OK

    def test_end_with_error_status(self) -> None:
        span = Span()
        span.end(SpanStatus.ERROR)
        assert span.status == SpanStatus.ERROR

    def test_end_with_unset_status(self) -> None:
        span = Span()
        span.end(SpanStatus.UNSET)
        assert span.status == SpanStatus.UNSET


# ---------------------------------------------------------------------------
# Span.duration_ms
# ---------------------------------------------------------------------------


class TestDurationMs:
    def test_duration_none_before_end(self) -> None:
        span = Span()
        assert span.duration_ms is None

    def test_duration_positive_after_end(self) -> None:
        clock, token = _install_fixture_clock(1_000_000.0)
        try:
            span = Span()
            clock.advance(1_000_001.0)  # +1 second = 1000 ms
            span.end()
            assert span.duration_ms is not None
            assert span.duration_ms == pytest.approx(1000.0, rel=1e-6)
        finally:
            restore_clock(token)

    def test_duration_is_float(self) -> None:
        clock, token = _install_fixture_clock(0.0)
        try:
            span = Span()
            clock.advance(0.001)
            span.end()
            assert isinstance(span.duration_ms, float)
        finally:
            restore_clock(token)


# ---------------------------------------------------------------------------
# Span.add_event()
# ---------------------------------------------------------------------------


class TestAddEvent:
    def test_add_event_appends(self) -> None:
        clock, token = _install_fixture_clock(100.0)
        try:
            span = Span()
            span.add_event("my-event", {"key": "val"})
            assert len(span.events) == 1
            assert span.events[0].name == "my-event"
        finally:
            restore_clock(token)

    def test_add_event_timestamp_from_clock(self) -> None:
        clock, token = _install_fixture_clock(200.0)
        try:
            span = Span()
            clock.advance(300.0)
            span.add_event("evt")
            assert span.events[0].timestamp == 300.0
        finally:
            restore_clock(token)

    def test_add_event_stores_attributes(self) -> None:
        span = Span()
        span.add_event("e", {"x": 1, "y": "hello"})
        assert span.events[0].attributes == {"x": 1, "y": "hello"}

    def test_add_event_no_attributes_defaults_empty(self) -> None:
        span = Span()
        span.add_event("e")
        assert span.events[0].attributes == {}

    def test_multiple_events_ordered(self) -> None:
        clock, token = _install_fixture_clock(1.0)
        try:
            span = Span()
            span.add_event("first")
            clock.advance(2.0)
            span.add_event("second")
            assert span.events[0].name == "first"
            assert span.events[1].name == "second"
            assert span.events[1].timestamp == 2.0
        finally:
            restore_clock(token)


# ---------------------------------------------------------------------------
# Span.set_attribute()
# ---------------------------------------------------------------------------


class TestSetAttribute:
    def test_set_attribute_stores_value(self) -> None:
        span = Span()
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_set_attribute_overwrites(self) -> None:
        span = Span()
        span.set_attribute("key", "first")
        span.set_attribute("key", "second")
        assert span.attributes["key"] == "second"

    def test_set_attribute_various_types(self) -> None:
        span = Span()
        span.set_attribute("str_val", "hello")
        span.set_attribute("int_val", 42)
        span.set_attribute("float_val", 3.14)
        span.set_attribute("bool_val", True)
        assert span.attributes["str_val"] == "hello"
        assert span.attributes["int_val"] == 42
        assert span.attributes["float_val"] == pytest.approx(3.14)
        assert span.attributes["bool_val"] is True


# ---------------------------------------------------------------------------
# Span.record_exception()
# ---------------------------------------------------------------------------


class TestRecordException:
    def test_record_exception_adds_event(self) -> None:
        span = Span()
        exc = ValueError("test error")
        span.record_exception(exc)
        assert len(span.events) == 1
        assert span.events[0].name == "exception"

    def test_record_exception_sets_status_error(self) -> None:
        span = Span()
        exc = RuntimeError("boom")
        span.record_exception(exc)
        assert span.status == SpanStatus.ERROR

    def test_record_exception_captures_type(self) -> None:
        span = Span()
        exc = ValueError("bad value")
        span.record_exception(exc)
        attrs = span.events[0].attributes
        assert "exception.type" in attrs
        assert "ValueError" in attrs["exception.type"]

    def test_record_exception_captures_message(self) -> None:
        span = Span()
        exc = ValueError("bad value")
        span.record_exception(exc)
        attrs = span.events[0].attributes
        assert attrs["exception.message"] == "bad value"

    def test_record_exception_captures_stacktrace(self) -> None:
        span = Span()
        try:
            raise RuntimeError("stack test")
        except RuntimeError as exc:
            span.record_exception(exc)
        attrs = span.events[0].attributes
        assert "exception.stacktrace" in attrs
        assert len(attrs["exception.stacktrace"]) > 0

    def test_record_exception_status_defaults_to_error(self) -> None:
        span = Span()
        span.record_exception(RuntimeError("boom"))
        assert span.status == SpanStatus.ERROR

    def test_record_exception_accepts_cancelled_status(self) -> None:
        span = Span()
        exc = RuntimeError("run was cancelled")
        span.record_exception(exc, status=SpanStatus.CANCELLED)
        assert span.status == SpanStatus.CANCELLED
        # Event is still recorded the same way regardless of status.
        assert span.events[0].attributes["exception.type"] == "RuntimeError"

    def test_record_exception_cancelled_distinct_from_error(self) -> None:
        span_error = Span()
        span_error.record_exception(RuntimeError("real failure"))
        span_cancelled = Span()
        span_cancelled.record_exception(
            RuntimeError("cut off"), status=SpanStatus.CANCELLED
        )
        assert span_error.status != span_cancelled.status

    def test_record_exception_without_response_omits_http_body_attrs(self) -> None:
        span = Span()
        span.record_exception(ValueError("plain error, no .response attribute"))
        attrs = span.events[0].attributes
        assert "exception.http_response_body" not in attrs
        assert "exception.http_status_code" not in attrs

    def test_record_exception_attaches_requests_style_http_error_body(self) -> None:
        """requests.exceptions.HTTPError.response has .status_code and
        .text — mirrors the actual shape #4940's Bedrock validation error
        arrives in via botocore/requests."""

        class _FakeResponse:
            status_code = 400
            text = '{"message": "Malformed input request: extra fields not permitted"}'

        class _FakeHTTPError(Exception):
            def __init__(self) -> None:
                super().__init__(
                    "400 Client Error: None for url: https://bedrock.example.com"
                )
                self.response = _FakeResponse()

        span = Span()
        span.record_exception(_FakeHTTPError())
        attrs = span.events[0].attributes
        assert attrs["exception.http_status_code"] == 400
        assert "Malformed input request" in attrs["exception.http_response_body"]
        # The generic str(exc) message is still captured too, unchanged.
        assert "400 Client Error" in attrs["exception.message"]

    def test_record_exception_attaches_httpx_style_http_error_body(self) -> None:
        """httpx.HTTPStatusError.response has .status_code and .text as
        well (httpx.Response's actual public API)."""

        class _FakeHttpxResponse:
            status_code = 429
            text = "rate limit exceeded, retry after 30s"

        class _FakeHTTPStatusError(Exception):
            def __init__(self) -> None:
                super().__init__("Client error '429 Too Many Requests'")
                self.response = _FakeHttpxResponse()

        span = Span()
        span.record_exception(_FakeHTTPStatusError())
        attrs = span.events[0].attributes
        assert attrs["exception.http_status_code"] == 429
        assert (
            attrs["exception.http_response_body"]
            == "rate limit exceeded, retry after 30s"
        )

    def test_record_exception_falls_back_to_response_content_bytes(self) -> None:
        class _FakeResponse:
            status_code = 500
            text = None
            content = b"internal proxy error body"

        class _FakeHTTPError(Exception):
            def __init__(self) -> None:
                super().__init__("500 Server Error")
                self.response = _FakeResponse()

        span = Span()
        span.record_exception(_FakeHTTPError())
        attrs = span.events[0].attributes
        assert attrs["exception.http_response_body"] == "internal proxy error body"

    def test_record_exception_truncates_very_large_response_body(self) -> None:
        class _FakeResponse:
            status_code = 400
            text = "x" * 20_000

        class _FakeHTTPError(Exception):
            def __init__(self) -> None:
                super().__init__("400 Client Error")
                self.response = _FakeResponse()

        span = Span()
        span.record_exception(_FakeHTTPError())
        body = span.events[0].attributes["exception.http_response_body"]
        assert len(body) < 20_000
        assert body.endswith("...<truncated>")

    def test_record_exception_malformed_response_object_degrades_gracefully(
        self,
    ) -> None:
        """A .response attribute that raises when accessed must never break
        record_exception itself — best-effort only."""

        class _ExplodingResponse:
            @property
            def text(self) -> str:
                raise RuntimeError("response body already consumed")

            @property
            def status_code(self) -> int:
                raise RuntimeError("no status code available")

        class _WeirdHTTPError(Exception):
            def __init__(self) -> None:
                super().__init__("weird error")
                self.response = _ExplodingResponse()

        span = Span()
        # Must not raise.
        span.record_exception(_WeirdHTTPError())
        assert span.status == SpanStatus.ERROR
        attrs = span.events[0].attributes
        assert "exception.http_response_body" not in attrs
        assert "exception.http_status_code" not in attrs


# ---------------------------------------------------------------------------
# Span.to_dict() / from_dict()
# ---------------------------------------------------------------------------


class TestSpanSerialization:
    def test_to_dict_no_datetime_objects(self) -> None:
        import datetime

        span = Span(name="test")
        span.end()
        d = span.to_dict()

        def _check_no_datetime(obj: Any) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _check_no_datetime(v)
            elif isinstance(obj, list):
                for item in obj:
                    _check_no_datetime(item)
            else:
                assert not isinstance(obj, datetime.datetime), (
                    f"Found datetime object: {obj}"
                )

        _check_no_datetime(d)

    def test_to_dict_primitive_values(self) -> None:
        """All top-level values in to_dict() must be str/int/float/bool/None/list/dict."""
        span = Span(name="test")
        span.end()
        d = span.to_dict()
        allowed = (str, int, float, bool, type(None), list, dict)
        for key, val in d.items():
            assert isinstance(val, allowed), (
                f"Key {key!r} has non-primitive value {val!r}"
            )

    def test_to_dict_expected_keys(self) -> None:
        span = Span(name="test")
        d = span.to_dict()
        expected_keys = {
            "span_id",
            "trace_id",
            "parent_id",
            "name",
            "start_time",
            "end_time",
            "status",
            "attributes",
            "events",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_status_is_string(self) -> None:
        span = Span()
        span.end(SpanStatus.ERROR)
        d = span.to_dict()
        assert d["status"] == "ERROR"
        assert isinstance(d["status"], str)

    def test_round_trip_basic(self) -> None:
        span = Span(name="round-trip", parent_id="parent-001")
        span.set_attribute("env", "test")
        span.add_event("checkpoint", {"step": 1})
        span.end()

        d = span.to_dict()
        restored = Span.from_dict(d)

        assert restored.span_id == span.span_id
        assert restored.trace_id == span.trace_id
        assert restored.name == span.name
        assert restored.parent_id == span.parent_id
        assert restored.start_time == span.start_time
        assert restored.end_time == span.end_time
        assert restored.status == span.status
        assert restored.attributes == span.attributes
        assert len(restored.events) == 1
        assert restored.events[0].name == "checkpoint"

    def test_round_trip_no_end(self) -> None:
        span = Span(name="open-span")
        d = span.to_dict()
        restored = Span.from_dict(d)
        assert restored.end_time is None
        assert restored.duration_ms is None

    def test_from_dict_handles_missing_optional_fields(self) -> None:
        minimal = {
            "span_id": str(uuid.uuid4()),
            "trace_id": str(uuid.uuid4()),
            "name": "minimal",
            "start_time": 0.0,
        }
        span = Span.from_dict(minimal)
        assert span.parent_id is None
        assert span.end_time is None
        assert span.status == SpanStatus.UNSET
        assert span.attributes == {}
        assert span.events == []
