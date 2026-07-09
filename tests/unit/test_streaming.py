"""
Unit tests for agent_trace.integrations.streaming — outbound SSE/websocket
delivery instrumentation (traced_sse_stream / traced_sse_astream /
traced_websocket_send).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.streaming import (
    traced_sse_astream,
    traced_sse_stream,
    traced_websocket_send,
)


@pytest.fixture()
def tracer_and_trace(tmp_path: Path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("streaming-unit-test") as trace:
        yield t, trace


class TestTracedSseStream:
    def test_yields_every_chunk_unchanged(self, tracer_and_trace):
        t, trace = tracer_and_trace
        chunks = list(traced_sse_stream(t, [b"data: 1\n\n", b"data: 2\n\n"]))
        assert chunks == [b"data: 1\n\n", b"data: 2\n\n"]

    def test_closes_ok_with_chunk_count(self, tracer_and_trace):
        t, trace = tracer_and_trace
        list(traced_sse_stream(t, ["a", "b", "c"]))
        spans = [s for s in trace.spans if s.name == "transport:sse"]
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.OK
        assert spans[0].attributes["transport.chunk_count"] == 3

    def test_records_chunk_preview_events(self, tracer_and_trace):
        t, trace = tracer_and_trace
        list(traced_sse_stream(t, ["hello", "world"]))
        span = next(s for s in trace.spans if s.name == "transport:sse")
        events = [e for e in span.events if e.name == "transport_chunk_sent"]
        assert len(events) == 2
        assert events[0].attributes["transport.chunk_preview"] == "hello"
        assert events[0].attributes["transport.index"] == 0
        assert events[1].attributes["transport.chunk_preview"] == "world"
        assert events[1].attributes["transport.index"] == 1

    def test_source_exception_closes_error_and_reraises(self, tracer_and_trace):
        t, trace = tracer_and_trace

        def bad_source():
            yield "ok"
            raise RuntimeError("broken pipe")

        with pytest.raises(RuntimeError, match="broken pipe"):
            list(traced_sse_stream(t, bad_source()))

        span = next(s for s in trace.spans if s.name == "transport:sse")
        assert span.status == SpanStatus.ERROR
        exception_events = [e for e in span.events if e.name == "exception"]
        assert exception_events
        assert exception_events[0].attributes["exception.type"] == "RuntimeError"

    def test_early_break_closes_cancelled(self, tracer_and_trace):
        t, trace = tracer_and_trace

        def slow_source():
            yield 1
            yield 2
            yield 3

        gen = traced_sse_stream(t, slow_source())
        next(gen)
        gen.close()

        span = next(s for s in trace.spans if s.name == "transport:sse")
        assert span.status == SpanStatus.CANCELLED
        assert span.attributes["transport.chunk_count"] == 1

    def test_custom_span_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        list(traced_sse_stream(t, ["x"], span_name="transport:custom"))
        assert any(s.name == "transport:custom" for s in trace.spans)

    def test_bytes_chunk_decoded_for_preview(self, tracer_and_trace):
        t, trace = tracer_and_trace
        list(traced_sse_stream(t, [b"hello bytes"]))
        span = next(s for s in trace.spans if s.name == "transport:sse")
        event = span.events[0]
        assert event.attributes["transport.chunk_preview"] == "hello bytes"

    def test_long_chunk_preview_truncated(self, tracer_and_trace):
        t, trace = tracer_and_trace
        long_chunk = "x" * 10_000
        list(traced_sse_stream(t, [long_chunk]))
        span = next(s for s in trace.spans if s.name == "transport:sse")
        preview = span.events[0].attributes["transport.chunk_preview"]
        assert len(preview) < len(long_chunk)
        assert preview.endswith("...<truncated>")


class TestTracedSseAstream:
    async def test_yields_every_chunk_unchanged(self, tracer_and_trace):
        t, trace = tracer_and_trace

        async def gen():
            for i in range(3):
                yield f"chunk-{i}"

        out = [c async for c in traced_sse_astream(t, gen())]
        assert out == ["chunk-0", "chunk-1", "chunk-2"]

    async def test_closes_ok_with_chunk_count(self, tracer_and_trace):
        t, trace = tracer_and_trace

        async def gen():
            yield "a"
            yield "b"

        async for _ in traced_sse_astream(t, gen()):
            pass

        span = next(s for s in trace.spans if s.name == "transport:sse")
        assert span.status == SpanStatus.OK
        assert span.attributes["transport.chunk_count"] == 2

    async def test_source_exception_closes_error_and_reraises(self, tracer_and_trace):
        t, trace = tracer_and_trace

        async def bad_gen():
            yield "ok"
            raise RuntimeError("dropped connection")

        with pytest.raises(RuntimeError, match="dropped connection"):
            async for _ in traced_sse_astream(t, bad_gen()):
                pass

        span = next(s for s in trace.spans if s.name == "transport:sse")
        assert span.status == SpanStatus.ERROR


class TestTracedWebsocketSend:
    async def test_successful_send_closes_ok(self, tracer_and_trace):
        t, trace = tracer_and_trace
        sent = []

        async def real_send(item):
            sent.append(item)
            return "sent"

        wrapped = traced_websocket_send(t, real_send)
        result = await wrapped("hello")

        assert result == "sent"
        assert sent == ["hello"]
        spans = [s for s in trace.spans if s.name == "transport:websocket"]
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.OK
        assert spans[0].attributes["transport.chunk_preview"] == "hello"

    async def test_failed_send_closes_error_and_reraises(self, tracer_and_trace):
        t, trace = tracer_and_trace

        async def raising_send(item):
            raise ConnectionError("client disconnected")

        wrapped = traced_websocket_send(t, raising_send)
        with pytest.raises(ConnectionError, match="client disconnected"):
            await wrapped("bye")

        spans = [s for s in trace.spans if s.name == "transport:websocket"]
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.ERROR
        exception_events = [e for e in spans[0].events if e.name == "exception"]
        assert exception_events
        assert exception_events[0].attributes["exception.type"] == "ConnectionError"

    async def test_one_span_per_call(self, tracer_and_trace):
        t, trace = tracer_and_trace

        async def real_send(item):
            return None

        wrapped = traced_websocket_send(t, real_send)
        await wrapped("first")
        await wrapped("second")

        spans = [s for s in trace.spans if s.name == "transport:websocket"]
        assert len(spans) == 2

    async def test_custom_span_name(self, tracer_and_trace):
        t, trace = tracer_and_trace

        async def real_send(item):
            return None

        wrapped = traced_websocket_send(t, real_send, span_name="transport:ws_custom")
        await wrapped("x")
        assert any(s.name == "transport:ws_custom" for s in trace.spans)
