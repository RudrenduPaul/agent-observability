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
import os
import warnings
from typing import TYPE_CHECKING, Any

from requests import PreparedRequest, Response
from requests.adapters import HTTPAdapter

if TYPE_CHECKING:
    from agent_trace.replay.fixture import Fixture

__all__ = [
    "NetworkGuardError",
    "RecordingAdapter",
    "ReplayAdapter",
]

logger = logging.getLogger(__name__)


class NetworkGuardError(RuntimeError):
    """Raised when a live network call is made during guarded replay.

    Mirrors the httpx_hook version.  Both share the same environment variable
    (AGENT_TRACE_NETWORK_GUARD) so a single ``export`` enables the guard for
    all HTTP clients simultaneously.
    """


def _guard_active() -> bool:
    return os.environ.get("AGENT_TRACE_NETWORK_GUARD", "0") == "1"


class RecordingAdapter(HTTPAdapter):
    """requests HTTPAdapter that persists each exchange to a Fixture.

    Mount this adapter on a requests.Session before making calls:

        session = requests.Session()
        session.mount("https://", RecordingAdapter(fixture))
    """

    def __init__(self, fixture: Fixture, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fixture = fixture

    def send(
        self,
        request: PreparedRequest,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        """Send the request, record the exchange, return the response."""
        response: Response = super().send(request, *args, **kwargs)

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


class ReplayAdapter(HTTPAdapter):
    """requests HTTPAdapter that serves responses from a Fixture.

    Mount this adapter on a requests.Session to intercept all outbound calls:

        session = requests.Session()
        session.mount("https://", ReplayAdapter(fixture))
        session.mount("http://", ReplayAdapter(fixture))
    """

    def __init__(self, fixture: Fixture, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fixture = fixture

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
            if _guard_active():
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
            return super().send(request, *args, **kwargs)

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
