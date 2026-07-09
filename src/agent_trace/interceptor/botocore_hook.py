"""
botocore session wrappers for recording and replaying AWS SDK (boto3) HTTP
exchanges — the botocore equivalent of httpx_hook.py / requests_patch.py.

Every boto3 service client (bedrock-runtime, sagemaker-runtime, s3, ...)
ultimately routes its outbound HTTP request through a single low-level
object: ``botocore.httpsession.URLLib3Session``, whose ``send(request)``
method takes an ``AWSPreparedRequest`` (``.method``/``.url``/``.headers``/
``.body``) and returns an ``AWSResponse`` (``.status_code``/``.headers``/
``.content``/``.text``) — the same request-in/response-out shape as
``requests.HTTPAdapter.send`` and ``httpx.BaseTransport.handle_request``.

RecordingSession wraps a real session-like object (anything exposing
``.send(request)``): it lets the request go through, reads the full
response body, and saves the exchange to the fixture before returning the
response to the caller.

ReplaySession never touches the network: it looks up the next recorded
response for the requested (method, url) from the fixture, mirroring
ReplayTransport/ReplayAdapter.

Streaming operations (e.g. Bedrock's ``InvokeModel``, whose response body
is delivered via ``StreamingBody``, or ``InvokeModelWithResponseStream``/
``ConverseStream``, whose response is an AWS event stream) set
``request.stream_output = True``.  For those, botocore does *not* eagerly
buffer the response at the urllib3 layer, so ``response.content`` is only
populated the first time something reads it.  RecordingSession still drains
and records these — mirroring httpx_hook.py's ``response.read()``/
``response.aread()`` eager-buffering of every exchange, including SSE — but
afterwards replaces ``response.raw`` with a fresh in-memory buffer over the
same bytes so downstream body consumption (``StreamingBody.read()``, event
stream iteration) keeps working transparently.  True non-buffering
pass-through capture for streaming responses is tracked as a separate
[redacted], exactly as it is for the httpx/SSE case.
"""

from __future__ import annotations

import io
import logging
import warnings
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_trace._replay.fixture import Fixture

from agent_trace.core.exceptions import NetworkGuardError, guard_active

__all__ = [
    "NetworkGuardError",
    "RecordingSession",
    "ReplaySession",
]

logger = logging.getLogger(__name__)


class _SendableSession(Protocol):
    """Structural type for anything exposing botocore's session.send(request)."""

    def send(self, request: Any) -> Any: ...


def _stringify_headers(headers: Any) -> dict[str, str]:
    """Coerce a botocore/urllib3 header mapping to a plain ``dict[str, str]``.

    botocore request headers are commonly a mix of ``bytes`` and ``str``
    values (e.g. ``Content-Type``/``Authorization`` are bytes, while
    ``Content-Length`` is str) — confirmed via direct inspection of a real
    signed ``AWSPreparedRequest``.  ``Fixture.record_exchange`` JSON-encodes
    the header dict, which fails on bytes values, so normalise everything to
    str here.
    """
    result: dict[str, str] = {}
    for raw_key, raw_value in dict(headers).items():
        str_key = (
            raw_key.decode("utf-8", errors="replace")
            if isinstance(raw_key, bytes)
            else str(raw_key)
        )
        str_value = (
            raw_value.decode("utf-8", errors="replace")
            if isinstance(raw_value, bytes)
            else str(raw_value)
        )
        result[str_key] = str_value
    return result


def _stringify_body(body: Any) -> str:
    """Coerce a botocore request body (bytes | str | file-like | None) to str."""
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        return body
    return ""


class RecordingSession:
    """botocore session wrapper that records every exchange to a Fixture.

    Parameters
    ----------
    fixture:
        Open Fixture instance where exchanges will be written.
    inner:
        The real session (or session-like object exposing ``.send(request)``)
        to use for outbound requests.  Defaults to a fresh
        ``botocore.httpsession.URLLib3Session()`` when None.
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: _SendableSession | None = None,
    ) -> None:
        self._fixture = fixture
        if inner is not None:
            self._inner: _SendableSession = inner
        else:
            import botocore.httpsession

            self._inner = botocore.httpsession.URLLib3Session()

    def send(self, request: Any) -> Any:
        """Forward the request, record the exchange, return the response."""
        response = self._inner.send(request)

        # Force the response body to be fully read so we can persist it.
        # For non-streaming operations botocore already preloaded this at
        # the urllib3 layer (see URLLib3Session.send), so this is a no-op
        # cache hit; for streaming operations (stream_output=True) this is
        # the first read and drains the live connection.
        content: bytes = response.content

        if request.stream_output:
            # StreamingBody / EventStream wrap `response.raw` directly
            # rather than `.content` (see
            # botocore.endpoint.convert_to_response_dict).  We just drained
            # the real socket above to record it, so hand the caller a
            # fresh in-memory buffer over the same bytes — BytesIO supports
            # the same incremental .read()/.readinto() interface — instead
            # of the now-exhausted urllib3 stream.
            response.raw = io.BytesIO(content)

        self._fixture.record_exchange(
            url=str(request.url),
            method=str(request.method),
            request_headers=_stringify_headers(request.headers),
            request_body=_stringify_body(request.body),
            response_status=int(response.status_code),
            response_headers=_stringify_headers(response.headers),
            response_body=response.text,
        )

        return response

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()


class ReplaySession:
    """botocore session wrapper that serves responses from a Fixture without
    network I/O.
    """

    def __init__(
        self,
        fixture: Fixture,
        clock: Any | None = None,
    ) -> None:
        self._fixture = fixture
        self._clock = clock

    def send(self, request: Any) -> Any:
        """Return the next recorded response for *(method, url)*."""
        import botocore.awsrequest

        url = str(request.url)
        method = str(request.method)
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
            import botocore.httpsession

            fallback = botocore.httpsession.URLLib3Session()
            try:
                return fallback.send(request)
            finally:
                fallback.close()

        if self._clock is not None:
            self._clock.advance(float(exchange["recorded_at"]))

        content = exchange["response_body"].encode("utf-8")
        response = botocore.awsrequest.AWSResponse(
            url,
            int(exchange["response_status"]),
            exchange["response_headers"],
            io.BytesIO(content),
        )
        # AWSResponse.content lazily reads `.raw`; pre-seed the cache so
        # repeated access (ours and the caller's) doesn't require `.raw` to
        # still be positioned at the start.
        response._content = content
        return response

    def close(self) -> None:
        pass  # No resources to release; fixture lifecycle is managed externally.
