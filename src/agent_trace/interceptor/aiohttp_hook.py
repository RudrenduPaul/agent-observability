"""
aiohttp interceptor for recording HTTP exchanges made via aiohttp.ClientSession.

Why a third interceptor (alongside httpx_hook.py / requests_patch.py)?
LiteLLM — the standard route agent frameworks (crewAI's fallback path,
LangChain's non-native providers, etc.) use to reach non-OpenAI/Anthropic
backends such as Gemini/Vertex AI — defaults to routing outbound async calls
through an aiohttp-based transport (`_should_use_aiohttp_transport()` returns
True unless explicitly disabled). That traffic never touches
`httpx.Client`/`httpx.AsyncClient` or `requests.Session` at all, so
agent-trace's two existing interceptors silently miss it with no error or
warning.

`aiohttp.ClientSession` has no pluggable "transport" object the way httpx
does, so recording is implemented by wrapping `ClientSession._request` — the
single coroutine every request-verb helper (`get`/`post`/`request`/...)
funnels through, and the same method the `aioresponses` mocking library
itself patches for exactly this reason (see `patch("aiohttp.client.
ClientSession._request", ...)` in aioresponses' own source).

Deliberately NOT a `ClientSession` subclass: the installed aiohttp (3.12+)
raises `DeprecationWarning: Inheritance class ... from ClientSession is
discouraged` from `ClientSession.__init_subclass__` for any subclass,
confirmed by direct introspection of the installed package. `make_recording_
request` instead builds a plain replacement function for `ClientSession.
_request` that `Tracer._patch_aiohttp` assigns onto the class directly —
the same class-level monkey-patch pattern already used for `httpx.Client.
__init__`/`httpx.AsyncClient.__init__` and `requests.Session.get_adapter`.

The replacement wraps the real (or previously patched) `_request()` call,
eagerly reads the response body (aiohttp caches it on
`ClientResponse._body`, so callers can still call `.read()`/`.text()`/
`.json()` afterwards without re-consuming the stream), and persists the
exchange to the fixture before returning the original response object
unmodified.

Only a recording path is provided here (no replay counterpart): this
interceptor's job is to stop agent-trace from *silently missing* aiohttp
traffic during recording. Offline replay of aiohttp-captured fixtures is a
separate, larger piece of work (constructing a faithful fake
`aiohttp.ClientResponse` without a live connection) and is not required to
close the gap this interceptor targets.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import aiohttp
    from yarl import URL

    from agent_trace._replay.fixture import Fixture

__all__ = [
    "make_recording_request",
]

logger = logging.getLogger(__name__)


def _serialize_request_body(kwargs: dict[str, Any]) -> str:
    """Best-effort reconstruction of the body bytes aiohttp will send.

    Mirrors RecordingAdapter's (requests_patch.py) bytes/str/else fallback:
    `json=` is serialized the same way `json.dumps` would represent it (the
    exact separators aiohttp's internal `json_serialize` uses can differ,
    but the semantic content — e.g. the model/messages payload a developer
    needs to root-cause a bug — is preserved); `data=` is decoded when it's
    already bytes/str (the common case for SDKs like LiteLLM's aiohttp
    transport, which hand aiohttp pre-serialized JSON bytes); anything else
    (FormData, an IO stream, a dict passed as `data=`) falls back to `str()`
    rather than raising, since raw evidence beats no evidence.
    """
    body = kwargs.get("json")
    if body is not None:
        try:
            return json.dumps(body)
        except (TypeError, ValueError):
            return str(body)

    data = kwargs.get("data")
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    if isinstance(data, str):
        return data
    return str(data)


async def _record_exchange(
    fixture: Fixture,
    method: str,
    str_or_url: str | URL,
    kwargs: dict[str, Any],
    response: aiohttp.ClientResponse,
) -> None:
    """Read *response*'s body and persist the exchange to *fixture*.

    `response.read()` caches the body on `ClientResponse._body`, so calling
    it here does not prevent the caller from reading the response again via
    `.read()` / `.text()` / `.json()`.
    """
    await response.read()

    req_headers = dict(kwargs.get("headers") or {})
    req_body = _serialize_request_body(kwargs)

    resp_headers = dict(response.headers)
    resp_body = await response.text()

    fixture.record_exchange(
        url=str(str_or_url),
        method=str(method).upper(),
        request_headers=req_headers,
        request_body=req_body,
        response_status=response.status,
        response_headers=resp_headers,
        response_body=resp_body,
    )


def make_recording_request(
    fixture: Fixture,
    original_request: Callable[..., Awaitable[aiohttp.ClientResponse]],
) -> Callable[..., Awaitable[aiohttp.ClientResponse]]:
    """Build a replacement for `aiohttp.ClientSession._request` that records.

    *original_request* is the real (or previously-patched) `_request`
    method, called through unconditionally so behaviour — redirects, auth,
    proxies, etc. — is unchanged; only the body-eager-read + fixture-record
    step is added around it.  Used by `Tracer._patch_aiohttp` to patch
    `aiohttp.ClientSession` at the class level so plain `aiohttp.ClientSession()`
    call sites inside third-party SDKs are captured with zero code changes.
    """

    async def _patched_request(
        session_self: aiohttp.ClientSession,
        method: str,
        str_or_url: str | URL,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        response = await original_request(session_self, method, str_or_url, **kwargs)
        await _record_exchange(fixture, method, str_or_url, kwargs, response)
        return response

    return _patched_request
