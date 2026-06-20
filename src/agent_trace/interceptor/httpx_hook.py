"""
httpx transports for recording and replaying HTTP exchanges.

RecordingTransport wraps a real transport: it lets the request go through,
reads the full response body, and saves the exchange to the fixture before
returning a reconstructed Response to the caller.  The caller sees no
difference — the response object behaves identically to the original.

ReplayTransport never touches the network.  It looks up the next recorded
response for the requested (method, url) and returns it.  If no fixture entry
is found and AGENT_TRACE_NETWORK_GUARD=1, it raises NetworkGuardError so that
test suites catch accidental live calls rather than silently using stale data.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from agent_trace._replay.fixture import Fixture

from agent_trace.core.exceptions import NetworkGuardError, guard_active

__all__ = [
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
        """Forward the request, record the exchange, return the response."""
        response = self._inner.handle_request(request)

        # Read the body eagerly so we can persist it; httpx streams lazily by
        # default and the caller may never fully read it otherwise.
        response.read()

        url = str(request.url)
        method = request.method
        req_headers = dict(request.headers)
        req_body = request.content.decode("utf-8", errors="replace")
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
