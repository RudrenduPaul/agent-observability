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
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: httpx.BaseTransport | None = None,
    ) -> None:
        self._fixture = fixture
        self._inner: httpx.BaseTransport = (
            inner if inner is not None else httpx.HTTPTransport()
        )

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
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._fixture = fixture
        self._inner: httpx.AsyncBaseTransport = (
            inner if inner is not None else httpx.AsyncHTTPTransport()
        )

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
