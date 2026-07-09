"""
httpx transports for recording and replaying HTTP exchanges.

RecordingTransport / AsyncRecordingTransport wrap a real transport: they let
the request go through, read the full response body, and save the exchange to
the fixture before returning a reconstructed Response to the caller.

ReplayTransport / AsyncReplayTransport never touch the network.  They look up
the next recorded response for the requested (method, url) from the fixture.
If no fixture entry is found and AGENT_TRACE_NETWORK_GUARD=1, they raise
NetworkGuardError so that test suites catch accidental live calls.

Both sync and async variants are provided because:
- httpx.Client (sync) requires httpx.BaseTransport (handle_request).
- httpx.AsyncClient (async) requires httpx.AsyncBaseTransport (handle_async_request).
Passing a sync transport to an async client fails silently at init and raises
AttributeError on the first request.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from agent_trace._replay.fixture import Fixture

from agent_trace.core.exceptions import NetworkGuardError, guard_active

__all__ = [
    "AsyncRecordingTransport",
    "AsyncReplayTransport",
    "NetworkGuardError",
    "RecordingTransport",
    "ReplayTransport",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass-through streaming tee — used by RecordingTransport/
# AsyncRecordingTransport when constructed with stream=True.
# ---------------------------------------------------------------------------
#
# The default (stream=False) recording path calls response.read()/aread()
# before returning anything to the caller: it fully drains the response body
# first, then reconstructs a fully-buffered httpx.Response. That destroys
# real incremental delivery for every request made while recording is
# active, not just the one that eventually reproduces a bug — a real cost
# for continuous/production recording of a streaming endpoint.
#
# The classes below wrap the real response's own byte iterator so the
# wrapped caller receives each chunk as soon as it arrives off the wire (the
# returned httpx.Response is constructed with stream=<tee>, never
# content=<fully-buffered bytes>), while the exact same bytes and their
# arrival timestamps are tee'd off to an on_complete callback once the
# stream is fully consumed (or explicitly closed) — that's the point the
# exchange actually gets written to the fixture.
#
# Best-effort by design: if a caller partially iterates the response and
# then simply drops the reference (never calling .read()/.iter_bytes() to
# exhaustion and never calling .close()), on_complete never fires and that
# exchange is not recorded — the same class of best-effort tradeoff the rest
# of this module already makes (e.g. ReplayTransport's network-guard
# fallback). httpx.Response.close() (auto-invoked by iter_raw() once its own
# source stream is exhausted; see httpx._models.Response.iter_raw) always
# reaches our close() in the normal full-read case.


class _TeeSyncByteStream(httpx.SyncByteStream):
    """Sync pass-through tee — see module docstring above."""

    def __init__(
        self,
        inner_response: httpx.Response,
        on_complete: Callable[[bytes, list[float]], None],
    ) -> None:
        self._inner_response = inner_response
        self._on_complete = on_complete
        self._chunks: list[bytes] = []
        self._chunk_offsets_s: list[float] = []
        self._start = time.monotonic()
        self._finished = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._inner_response.iter_bytes():
                self._chunk_offsets_s.append(time.monotonic() - self._start)
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._finish()

    def close(self) -> None:
        self._inner_response.close()
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._on_complete(b"".join(self._chunks), list(self._chunk_offsets_s))


class _TeeAsyncByteStream(httpx.AsyncByteStream):
    """Async pass-through tee — see module docstring above."""

    def __init__(
        self,
        inner_response: httpx.Response,
        on_complete: Callable[[bytes, list[float]], None],
    ) -> None:
        self._inner_response = inner_response
        self._on_complete = on_complete
        self._chunks: list[bytes] = []
        self._chunk_offsets_s: list[float] = []
        self._start = time.monotonic()
        self._finished = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner_response.aiter_bytes():
                self._chunk_offsets_s.append(time.monotonic() - self._start)
                self._chunks.append(chunk)
                yield chunk
        finally:
            self._finish()

    async def aclose(self) -> None:
        await self._inner_response.aclose()
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._on_complete(b"".join(self._chunks), list(self._chunk_offsets_s))


class RecordingTransport(httpx.BaseTransport):
    """httpx transport that records every exchange to a Fixture.

    Parameters
    ----------
    fixture:
        Open Fixture instance where exchanges will be written.
    inner:
        The real transport to use for outbound requests.  Defaults to
        ``httpx.HTTPTransport()`` when None, which honours proxy/SSL settings
        from the surrounding httpx.Client.
    stream:
        When False (the default), the historical eager-buffering behavior:
        the full response body is read before this transport returns
        anything to the caller, and the fixture is written synchronously
        inside handle_request(). When True, opt into pass-through streaming
        capture: the caller receives each chunk as it actually arrives off
        the wire (no full-body buffering before the first byte reaches the
        caller) while the same bytes and their per-chunk arrival timestamps
        are tee'd into the fixture once the response stream is fully
        consumed or closed. Use stream=True for always-on recording of
        streaming/SSE endpoints, where the default mode's full-buffering
        would otherwise destroy real incremental token delivery for every
        request made while recording is active.
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: httpx.BaseTransport | None = None,
        *,
        stream: bool = False,
    ) -> None:
        self._fixture = fixture
        self._inner: httpx.BaseTransport = (
            inner if inner is not None else httpx.HTTPTransport()
        )
        self._stream = stream

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Forward the request, record the exchange, return the response.

        A pre-response exception (connection refused, DNS failure, TLS
        failure, an httpx.UnsupportedProtocol from a malformed/missing-scheme
        URL, ...) is recorded too — as a failed-before-response exchange
        (error_type/error_message, no response_status) — instead of being
        silently lost, then re-raised unchanged so the caller sees the exact
        same failure it would without recording active.
        """
        url = str(request.url)
        method = request.method
        req_headers = dict(request.headers)
        req_body = request.content.decode("utf-8", errors="replace")

        start = time.monotonic()
        try:
            response = self._inner.handle_request(request)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._fixture.record_exchange(
                url=url,
                method=method,
                request_headers=req_headers,
                request_body=req_body,
                duration_ms=duration_ms,
                error_type=type(exc).__qualname__,
                error_message=str(exc),
            )
            raise

        if self._stream:
            return self._handle_stream_response(
                request, response, url, method, req_headers, req_body, start
            )

        duration_ms = (time.monotonic() - start) * 1000

        # Read the body eagerly so we can persist it; httpx streams lazily by
        # default and the caller may never fully read it otherwise.
        response.read()

        resp_status = response.status_code
        resp_headers = dict(response.headers)
        resp_body = response.text

        self._fixture.record_exchange(
            url=url,
            method=method,
            request_headers=req_headers,
            request_body=req_body,
            response_status=resp_status,
            response_headers=resp_headers,
            response_body=resp_body,
            duration_ms=duration_ms,
        )

        # Reconstruct so the caller receives a fully-read response with the
        # same status, headers, and body as the original.
        return httpx.Response(
            status_code=resp_status,
            headers=resp_headers,
            content=response.content,
            request=request,
        )

    def _handle_stream_response(
        self,
        request: httpx.Request,
        response: httpx.Response,
        url: str,
        method: str,
        req_headers: dict[str, str],
        req_body: str,
        start: float,
    ) -> httpx.Response:
        """stream=True path: tee the response body to the caller and the
        fixture simultaneously instead of buffering it first."""
        resp_status = response.status_code
        resp_headers = dict(response.headers)

        def _on_complete(body_bytes: bytes, chunk_offsets_s: list[float]) -> None:
            duration_ms = (time.monotonic() - start) * 1000
            self._fixture.record_exchange(
                url=url,
                method=method,
                request_headers=req_headers,
                request_body=req_body,
                response_status=resp_status,
                response_headers=resp_headers,
                response_body=body_bytes.decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
                chunk_timestamps=chunk_offsets_s,
            )

        tee = _TeeSyncByteStream(response, _on_complete)
        return httpx.Response(
            status_code=resp_status,
            headers=resp_headers,
            stream=tee,
            request=request,
        )

    def close(self) -> None:
        self._inner.close()


class ReplayTransport(httpx.BaseTransport):
    """httpx transport that serves responses from a Fixture without network I/O.

    Limitation — replay cannot simulate a *modified* request: this transport
    only ever serves back the exact recorded ``response_body`` for a matching
    ``(method, url)`` pair (see ``handle_request`` below). It never re-derives
    what a request with different parameters would have returned — a
    different model, a different ``model_settings.reasoning_effort``/
    ``verbosity``, a different tool schema, a different prompt, etc. all
    still replay the *original* recorded response verbatim. If the change
    you're validating is itself a request parameter (e.g. switching
    ``reasoning_effort`` to fix a tool-calling regression), record a fresh
    run after making that change — replay only re-serves history, it does
    not re-run inference.

    Parameters
    ----------
    fixture:
        Open Fixture instance to serve recorded exchanges from.
    clock:
        Optional FixtureClock to advance with each exchange's recorded_at
        timestamp, reproducing original execution timing during replay.
    """

    def __init__(
        self,
        fixture: Fixture,
        clock: Any | None = None,
    ) -> None:
        self._fixture = fixture
        self._clock = clock

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Return the next recorded response for *(method, url)*."""
        url = str(request.url)
        method = request.method
        exchange: dict[str, Any] | None = self._fixture.next_exchange(url, method)

        if exchange is None:
            if guard_active():
                raise NetworkGuardError(
                    f"No recorded exchange for {method} {url} and "
                    "AGENT_TRACE_NETWORK_GUARD=1 is set.  "
                    "Run in recording mode first to capture this request."
                )
            # Guard is off: fall back to real network with a warning so
            # developers notice in CI logs without failing immediately.
            warnings.warn(
                f"agent-trace: no fixture entry for {method} {url}; "
                "falling through to live network.  Set AGENT_TRACE_NETWORK_GUARD=1 "
                "to make this an error.",
                stacklevel=2,
            )
            fallback = httpx.HTTPTransport()
            try:
                return fallback.handle_request(request)
            finally:
                fallback.close()

        # Advance the replay clock to the recorded wall time so that spans
        # created during replay carry meaningful relative timestamps.
        if self._clock is not None:
            self._clock.advance(float(exchange["recorded_at"]))

        return httpx.Response(
            status_code=int(exchange["response_status"]),
            headers=exchange["response_headers"],
            content=exchange["response_body"].encode("utf-8"),
            request=request,
        )

    def close(self) -> None:
        pass  # No resources to release; fixture lifecycle is managed externally.


class AsyncRecordingTransport(httpx.AsyncBaseTransport):
    """httpx async transport that records every exchange to a Fixture.

    Use this with ``httpx.AsyncClient`` — the sync ``RecordingTransport``
    cannot be used with async clients because ``AsyncClient`` requires
    ``handle_async_request``.

    Parameters
    ----------
    fixture:
        Open Fixture instance where exchanges will be written.
    inner:
        The real async transport to use for outbound requests.  Defaults to
        ``httpx.AsyncHTTPTransport()`` when None.
    stream:
        Same opt-in pass-through streaming capture mode as
        ``RecordingTransport(..., stream=True)`` — see that class's
        docstring. Defaults to False (the historical eager-buffering
        behavior).
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: httpx.AsyncBaseTransport | None = None,
        *,
        stream: bool = False,
    ) -> None:
        self._fixture = fixture
        self._inner: httpx.AsyncBaseTransport = (
            inner if inner is not None else httpx.AsyncHTTPTransport()
        )
        self._stream = stream

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Forward the request async, record the exchange, return the response.

        A pre-response exception is recorded too — see
        ``RecordingTransport.handle_request``'s docstring for the sync
        equivalent of this behavior.
        """
        url = str(request.url)
        method = request.method
        req_headers = dict(request.headers)
        req_body = request.content.decode("utf-8", errors="replace")

        start = time.monotonic()
        try:
            response = await self._inner.handle_async_request(request)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._fixture.record_exchange(
                url=url,
                method=method,
                request_headers=req_headers,
                request_body=req_body,
                duration_ms=duration_ms,
                error_type=type(exc).__qualname__,
                error_message=str(exc),
            )
            raise

        if self._stream:
            return self._handle_stream_response(
                request, response, url, method, req_headers, req_body, start
            )

        duration_ms = (time.monotonic() - start) * 1000
        await response.aread()

        resp_status = response.status_code
        resp_headers = dict(response.headers)
        resp_body = response.text

        self._fixture.record_exchange(
            url=url,
            method=method,
            request_headers=req_headers,
            request_body=req_body,
            response_status=resp_status,
            response_headers=resp_headers,
            response_body=resp_body,
            duration_ms=duration_ms,
        )

        return httpx.Response(
            status_code=resp_status,
            headers=resp_headers,
            content=response.content,
            request=request,
        )

    def _handle_stream_response(
        self,
        request: httpx.Request,
        response: httpx.Response,
        url: str,
        method: str,
        req_headers: dict[str, str],
        req_body: str,
        start: float,
    ) -> httpx.Response:
        """stream=True path: tee the response body to the caller and the
        fixture simultaneously instead of buffering it first."""
        resp_status = response.status_code
        resp_headers = dict(response.headers)

        def _on_complete(body_bytes: bytes, chunk_offsets_s: list[float]) -> None:
            duration_ms = (time.monotonic() - start) * 1000
            self._fixture.record_exchange(
                url=url,
                method=method,
                request_headers=req_headers,
                request_body=req_body,
                response_status=resp_status,
                response_headers=resp_headers,
                response_body=body_bytes.decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
                chunk_timestamps=chunk_offsets_s,
            )

        tee = _TeeAsyncByteStream(response, _on_complete)
        return httpx.Response(
            status_code=resp_status,
            headers=resp_headers,
            stream=tee,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


class AsyncReplayTransport(httpx.AsyncBaseTransport):
    """httpx async transport that serves responses from a Fixture without network I/O.

    Use this with ``httpx.AsyncClient`` — the sync ``ReplayTransport`` cannot
    be used with async clients because ``AsyncClient`` requires
    ``handle_async_request``.

    Same limitation as ``ReplayTransport``: it replays the exact recorded
    bytes for a matching ``(method, url)`` and cannot simulate what a
    *modified* request (different model, ``model_settings``, tool schema,
    prompt, ...) would have returned. See ``ReplayTransport``'s docstring.

    Parameters
    ----------
    fixture:
        Open Fixture instance to serve recorded exchanges from.
    clock:
        Optional FixtureClock to advance with each exchange's recorded_at
        timestamp, reproducing original execution timing during replay.
    """

    def __init__(
        self,
        fixture: Fixture,
        clock: Any | None = None,
    ) -> None:
        self._fixture = fixture
        self._clock = clock

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Return the next recorded response for *(method, url)*."""
        url = str(request.url)
        method = request.method
        exchange: dict[str, Any] | None = self._fixture.next_exchange(url, method)

        if exchange is None:
            if guard_active():
                raise NetworkGuardError(
                    f"No recorded exchange for {method} {url} and "
                    "AGENT_TRACE_NETWORK_GUARD=1 is set.  "
                    "Run in recording mode first to capture this request."
                )
            warnings.warn(
                f"agent-trace: no fixture entry for {method} {url}; "
                "falling through to live network.  Set AGENT_TRACE_NETWORK_GUARD=1 "
                "to make this an error.",
                stacklevel=2,
            )
            fallback = httpx.AsyncHTTPTransport()
            try:
                return await fallback.handle_async_request(request)
            finally:
                await fallback.aclose()

        if self._clock is not None:
            self._clock.advance(float(exchange["recorded_at"]))

        return httpx.Response(
            status_code=int(exchange["response_status"]),
            headers=exchange["response_headers"],
            content=exchange["response_body"].encode("utf-8"),
            request=request,
        )

    async def aclose(self) -> None:
        pass  # No resources to release; fixture lifecycle is managed externally.
