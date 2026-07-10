"""
Outbound event-stream/SSE (or websocket) delivery instrumentation.

agent-trace's LangGraph integration only hooks ``langchain_core`` callbacks
that fire *in-process* during graph execution — the same layer LangSmith
already instruments. Failures where the backend graph completes
successfully but the frontend never receives the terminal event (e.g.
dropped/stalled SSE connections, a silent exception raised inside a
streaming response writer, a websocket send that raises after the client
disconnected) happen entirely *downstream* of that layer — langgraph#6202's
failure class. There was previously zero code anywhere in agent-trace that
touched this transport boundary (confirmed via repo-wide grep for
``stream|sse|websocket|asgi|fastapi`` prior to this module).

This module instruments the layer that actually carries chunks out to a
browser client, independent of the in-process LangGraph/LangChain callback
layer — a thin, framework-agnostic wrapper around whatever iterable/
async-iterable/send-callable a server framework hands its transport (a
Starlette/FastAPI ``StreamingResponse`` body iterator, the generator behind
LangGraph's own ``astream_events``, or a websocket's ``.send()`` method).
No hard dependency on any specific web framework — it only needs a
generator/async-generator or an async callable.

Usage (SSE via a Starlette/FastAPI ``StreamingResponse``)::

    from agent_trace.integrations.streaming import traced_sse_astream

    async def event_source():
        async for chunk in graph.astream_events(state, version="v2"):
            yield f"data: {json.dumps(chunk)}\\n\\n"

    return StreamingResponse(
        traced_sse_astream(tracer, event_source()),
        media_type="text/event-stream",
    )

Usage (websocket)::

    from agent_trace.integrations.streaming import traced_websocket_send

    send = traced_websocket_send(tracer, websocket.send_text)
    await send(json.dumps(chunk))
"""

from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Generator,
    Iterable,
)
from typing import TYPE_CHECKING, Any, TypeVar

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Tracer

__all__ = [
    "traced_sse_astream",
    "traced_sse_stream",
    "traced_websocket_send",
]

# Bounds — this records enough of each outbound chunk to diagnose a
# dropped/stalled transport, not a full-fidelity mirror of every byte sent.
_MAX_CHUNK_PREVIEW_LEN = 500
_MAX_TRANSPORT_EVENTS = 500

T = TypeVar("T")


def _stringify_chunk(item: Any) -> str:
    """Best-effort, bounded preview of an outbound chunk for a SpanEvent
    attribute. Never raises — falls back to a placeholder."""
    try:
        if isinstance(item, bytes):
            text = item.decode("utf-8", errors="replace")
        else:
            text = str(item)
    except Exception:
        return "<unrepresentable-chunk>"
    if len(text) > _MAX_CHUNK_PREVIEW_LEN:
        text = text[:_MAX_CHUNK_PREVIEW_LEN] + "...<truncated>"
    return text


def _record_transport_chunk_sent(span: Span, index: int, item: Any) -> None:
    """Append one bounded transport_chunk_sent SpanEvent, capped at
    _MAX_TRANSPORT_EVENTS so a very long-lived stream can't grow a single
    span's event list without limit. transport.chunk_count (set at span
    close) still reflects every chunk that was actually sent."""
    if index >= _MAX_TRANSPORT_EVENTS:
        return
    span.add_event(
        "transport_chunk_sent",
        attributes={
            "transport.index": index,
            "transport.chunk_preview": _stringify_chunk(item),
        },
    )


def traced_sse_stream(
    tracer: Tracer,
    source: Iterable[T],
    *,
    span_name: str = "transport:sse",
) -> Generator[T, None, None]:
    """Wrap a synchronous outbound SSE/streaming-response iterable (e.g. the
    body iterator handed to a WSGI/Starlette ``StreamingResponse``) so each
    chunk's actual send moment — and a bounded preview of its content — lands
    on the trace timeline, independent of the in-process LangGraph callback
    layer.

    Opens one ``transport:sse`` span (customizable via *span_name*) that
    stays open for the lifetime of iteration, closing ``OK`` once the
    source is exhausted (every chunk successfully handed off to the
    transport), ``ERROR`` if the source itself raises (e.g. a broken pipe
    surfaced back into application code) with the exception recorded onto
    the span then re-raised unchanged, or ``CANCELLED`` if the *caller*
    (the ASGI/WSGI server) stops iterating early — the exact shape of a
    dropped/stalled client connection — detected via ``GeneratorExit`` when
    this generator is closed/garbage-collected before exhaustion.
    """
    span = tracer.start_span(span_name)
    index = 0
    status = SpanStatus.OK
    try:
        for item in source:
            _record_transport_chunk_sent(span, index, item)
            index += 1
            yield item
    except GeneratorExit:
        status = SpanStatus.CANCELLED
        raise
    except Exception as exc:
        status = SpanStatus.ERROR
        span.record_exception(exc)
        raise
    finally:
        span.set_attribute("transport.chunk_count", index)
        if span.end_time is None:
            span.end(status)


async def traced_sse_astream(
    tracer: Tracer,
    source: AsyncIterable[T],
    *,
    span_name: str = "transport:sse",
) -> AsyncGenerator[T, None]:
    """Async equivalent of :func:`traced_sse_stream` — wraps an async
    outbound SSE generator (e.g. the body passed to FastAPI's
    ``StreamingResponse``, or the generator behind ``graph.astream_events``)
    the same way."""
    span = tracer.start_span(span_name)
    index = 0
    status = SpanStatus.OK
    try:
        async for item in source:
            _record_transport_chunk_sent(span, index, item)
            index += 1
            yield item
    except GeneratorExit:
        status = SpanStatus.CANCELLED
        raise
    except Exception as exc:
        status = SpanStatus.ERROR
        span.record_exception(exc)
        raise
    finally:
        span.set_attribute("transport.chunk_count", index)
        if span.end_time is None:
            span.end(status)


def traced_websocket_send(
    tracer: Tracer,
    send_fn: Callable[[T], Awaitable[Any]],
    *,
    span_name: str = "transport:websocket",
) -> Callable[[T], Awaitable[Any]]:
    """Wrap an async websocket ``send``-shaped callable (e.g.
    ``websocket.send_text``/``send_json``/``send_bytes``) so every outbound
    send is recorded with a timestamp and success/failure status,
    independent of the in-process LangGraph callback layer.

    Returns a new async callable with the same signature as *send_fn*.
    Opens one dedicated span per call (rather than one long-lived span for
    the whole connection, since a websocket connection's lifetime is not
    naturally scoped the way a single HTTP streaming response's is) —
    closing ``OK`` immediately after a successful send, or ``ERROR`` (with
    the exception recorded, then re-raised unchanged) if the send itself
    raises — e.g. because the client already disconnected.
    """

    async def _wrapped(item: T) -> Any:
        span = tracer.start_span(span_name)
        # _stringify_chunk() is itself defensive (never raises) and
        # set_attribute() on a plain string cannot raise either.
        span.set_attribute("transport.chunk_preview", _stringify_chunk(item))
        try:
            result = await send_fn(item)
        except Exception as exc:
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)
            raise
        else:
            if span.end_time is None:
                span.end(SpanStatus.OK)
            return result

    return _wrapped
