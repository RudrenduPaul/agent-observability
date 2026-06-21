"""
requests adapters for recording and replaying HTTP exchanges.

These are the requests-library equivalents of httpx_hook.py's transports.
RecordingAdapter wraps the real HTTPAdapter and saves each exchange.
ReplayAdapter serves responses from the fixture without touching the network.

Why two separate modules (httpx_hook + requests_patch)?
Many AI SDKs use httpx (e.g. the Anthropic Python SDK); others use requests
(e.g. OpenAI's legacy client).  Supporting both lets agent-trace intercept
the full HTTP layer regardless of which SDK the user chooses.
"""

from __future__ import annotations

import io
import logging
import warnings
from typing import TYPE_CHECKING, Any

from requests import PreparedRequest, Response
from requests.adapters import BaseAdapter, HTTPAdapter

if TYPE_CHECKING:
    from agent_trace._replay.fixture import Fixture

from agent_trace.core.exceptions import NetworkGuardError, guard_active

__all__ = [
    "NetworkGuardError",
    "RecordingAdapter",
    "ReplayAdapter",
]

logger = logging.getLogger(__name__)


class RecordingAdapter(HTTPAdapter):
    """requests HTTPAdapter that persists each exchange to a Fixture.

    Mount this adapter on a requests.Session before making calls:

        session = requests.Session()
        session.mount("https://", RecordingAdapter(fixture))

    Pass *inner* to wrap an existing adapter (preserves connection pooling
    and custom adapters) instead of creating a fresh HTTPAdapter pool.
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: BaseAdapter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fixture = fixture
        self._inner = inner

    def send(
        self,
        request: PreparedRequest,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        """Send the request, record the exchange, return the response."""
        if self._inner is not None:
            response: Response = self._inner.send(request, *args, **kwargs)
        else:
            response = super().send(request, *args, **kwargs)

        url = str(request.url or "")
        method = str(request.method or "GET").upper()
        req_headers = dict(request.headers or {})
        # PreparedRequest.body can be bytes, str, or None.
        body = request.body
        if isinstance(body, bytes):
            req_body = body.decode("utf-8", errors="replace")
        elif isinstance(body, str):
            req_body = body
        else:
            req_body = ""

        resp_headers = dict(response.headers)
        resp_body = response.text  # reads and caches the body

        self._fixture.record_exchange(
            url=url,
            method=method,
            request_headers=req_headers,
            request_body=req_body,
            response_status=response.status_code,
            response_headers=resp_headers,
            response_body=resp_body,
        )

        return response


class ReplayAdapter(BaseAdapter):
    """requests adapter that serves responses from a Fixture without network I/O.

    Mount this adapter on a requests.Session to intercept all outbound calls:

        session = requests.Session()
        session.mount("https://", ReplayAdapter(fixture))
        session.mount("http://", ReplayAdapter(fixture))
    """

    def __init__(self, fixture: Fixture) -> None:
        super().__init__()
        self._fixture = fixture

    def close(self) -> None:
        pass

    def send(
        self,
        request: PreparedRequest,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        """Return the next recorded response for this request without I/O."""
        url = str(request.url or "")
        method = str(request.method or "GET").upper()

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
            fallback = HTTPAdapter()
            try:
                return fallback.send(request, *args, **kwargs)
            finally:
                fallback.close()

        response = Response()
        response.status_code = int(exchange["response_status"])
        response.headers.update(exchange["response_headers"])
        # _content must be set as bytes so that response.text works correctly.
        content = exchange["response_body"].encode("utf-8")
        response._content = content
        response.encoding = "utf-8"
        response.url = url
        response.request = request
        # Wrap bytes in a BytesIO so raw.read() works if callers check it.
        response.raw = io.BytesIO(content)

        return response
