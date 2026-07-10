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

import contextvars
import json
import logging
import time
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Generator

    from agent_trace._replay.fixture import Fixture

from agent_trace.core.exceptions import (
    NetworkGuardError,
    RunawayToolCallLoopError,
    guard_active,
)

__all__ = [
    "AsyncRecordingTransport",
    "AsyncReplayTransport",
    "NetworkGuardError",
    "RecordingTransport",
    "ReplayTransport",
    "RunawayToolCallLoopError",
    "correlation_context",
    "current_correlation_id",
    "pop_correlation_id",
    "push_correlation_id",
    "raise_on_loop_detected",
    "warn_on_loop_detected",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Every reconstructed httpx.Response below is built from body bytes that are
# already fully decoded (httpx's own Response.read()/.text/.content transparently
# gunzips/brotli-decodes based on Content-Encoding before we ever see them, and
# fixture-replayed bodies are the plain text persisted by record_exchange).
# Carrying the *original* response's Content-Encoding/Content-Length headers
# into that reconstruction is wrong on both counts: httpx.Response.__init__
# calls self.read() immediately, which re-applies the decoder implied by
# Content-Encoding to already-plain bytes and raises
# `httpx.DecodingError: Error -3 while decompressing data: incorrect header
# check` — reproduced live against the real (gzip-compressing-by-default)
# Gemini API, so this broke both recording (RecordingTransport/
# AsyncRecordingTransport) and replay (ReplayTransport/AsyncReplayTransport)
# for any upstream that compresses responses, not just Gemini. Content-Length
# is equally stale (it described the compressed byte count) and left in place
# would mislead any caller that trusts it over the actual body length.
# ---------------------------------------------------------------------------

_STALE_BODY_HEADERS = {"content-encoding", "content-length"}


def _strip_stale_body_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop headers that describe the *original* (possibly compressed) wire
    body, not the already-decoded bytes being used to reconstruct a Response."""
    return {k: v for k, v in headers.items() if k.lower() not in _STALE_BODY_HEADERS}


# ---------------------------------------------------------------------------
# Batch-input / graph-node correlation — ties a recorded HTTP exchange back
# to whichever concurrent batch input (e.g. LangChain's abatch(config={
# "max_concurrency": N})) or graph node it originated from, instead of
# leaving N interleaved exchanges with no way to tell them apart short of
# manually diffing recorded request bodies against the original input list
# (#30924, #6037, #13449).
#
# A plain contextvars.ContextVar rather than thread-local state: each
# asyncio.Task created via asyncio.create_task()/asyncio.gather() gets its
# own copy of the current Context at creation time, so setting a distinct
# correlation id per concurrently-scheduled coroutine correctly isolates
# each batch input's exchanges from its siblings' — the same isolation
# mechanism Tracer._active_trace_var/_active_fixture_var already rely on
# elsewhere in this codebase for overlapping start_trace() contexts.
# ---------------------------------------------------------------------------

_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_trace_correlation_id", default=None
)


@contextmanager
def correlation_context(correlation_id: str) -> Generator[None, None, None]:
    """Tag every HTTP exchange recorded while this context is active with
    *correlation_id*.

    Usage — tagging each item of a concurrent batch call so its recorded
    exchanges are distinguishable afterwards::

        async def _call_one(i, item):
            with correlation_context(f"batch-item-{i}"):
                return await chain.ainvoke(item)

        await asyncio.gather(*(_call_one(i, x) for i, x in enumerate(items)))

    Recover the grouped exchanges afterwards via
    ``Fixture.exchanges_for_correlation_id(correlation_id)`` or
    ``Fixture.correlation_ids()``. Nesting is supported — the innermost
    active context wins, and the previous value (possibly None) is restored
    on exit.
    """
    token = _correlation_id_var.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id_var.reset(token)


def current_correlation_id() -> str | None:
    """Return the correlation id set by the innermost active
    ``correlation_context()``, or None if none is active."""
    return _correlation_id_var.get()


def push_correlation_id(correlation_id: str) -> contextvars.Token[str | None]:
    """Manual ``set()`` counterpart to ``correlation_context()`` for
    callers that must set/reset across two separate call sites — e.g. a
    callback-based framework integration's ``on_X_start``/``on_X_end`` pair
    — rather than within a single enclosing ``with`` block.

    Pair every call with ``pop_correlation_id(token)`` once the
    corresponding span closes. See
    ``agent_trace.integrations.langgraph.LangGraphTracer`` for the actual
    usage: every span it opens (node/LLM/tool) pushes its own ``span_id``
    as the correlation id for the duration that span is open, so an HTTP
    exchange made anywhere inside it — including inside a supervisor
    topology's sub-agent node — is automatically tagged with the
    originating span, closing the "which node produced this HTTP call"
    gap (#6037) the same way ``correlation_context()`` closes it for a
    manually-tagged concurrent batch input (#30924).
    """
    return _correlation_id_var.set(correlation_id)


def pop_correlation_id(token: contextvars.Token[str | None]) -> None:
    """Reset counterpart to ``push_correlation_id()``."""
    _correlation_id_var.reset(token)


# ---------------------------------------------------------------------------
# Runaway-tool-call-loop guard — optional, opt-in via
# RecordingTransport(..., loop_guard_threshold=N). Counts *consecutive*
# tool-call-only responses (no final, tool-call-free assistant message) seen
# for the same host during the life of this transport instance and calls
# on_loop_detected(host, count) once the count reaches the threshold (and
# again on every subsequent response while it stays at/above threshold).
#
# This is a live, in-the-loop signal — not a replay-time diagnostic — for
# issue #3097 (a model that never stops emitting tool_calls, burning the
# full context window before anyone notices). Disabled by default
# (loop_guard_threshold=None) so it costs nothing for callers who don't want
# it: no per-response JSON parsing, no per-host bookkeeping.
# ---------------------------------------------------------------------------


def _response_host(url: str) -> str:
    """Best-effort hostname for *url*; falls back to the raw url string on
    any parse failure so the guard never raises for this reason alone."""
    try:
        host = httpx.URL(url).host
        return host or url
    except Exception:
        return url


def _is_tool_call_only_response(response_body: str) -> bool:
    """Best-effort detection of a chat-completion response that carries
    tool call(s) and nothing else — no final, human-readable assistant
    message alongside them.

    Recognizes two response shapes:
      - OpenAI/Groq/Azure-OpenAI-style Chat Completions:
        ``choices[0].message.tool_calls`` non-empty and
        ``choices[0].message.content`` empty/null, or
        ``choices[0].finish_reason == "tool_calls"``.
      - Anthropic Messages API: ``stop_reason == "tool_use"`` and no
        ``content`` block of type ``"text"`` carries non-empty text.

    Best-effort by design (same tradeoff as the rest of this module): a
    response body that isn't valid JSON, or doesn't match either shape,
    simply isn't flagged — this never raises for a malformed/unrecognized
    body, it just returns False.
    """
    try:
        data = json.loads(response_body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
                has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
                if has_tool_calls:
                    content = message.get("content")
                    empty_content = content in (None, "", [])
                    if empty_content or first.get("finish_reason") == "tool_calls":
                        return True

    if data.get("stop_reason") == "tool_use":
        content_blocks = data.get("content")
        if isinstance(content_blocks, list) and content_blocks:
            has_text = any(
                isinstance(block, dict)
                and block.get("type") == "text"
                and block.get("text")
                for block in content_blocks
            )
            if not has_text:
                return True

    return False


def warn_on_loop_detected(host: str, count: int) -> None:
    """Default ``on_loop_detected`` callback — emits a ``UserWarning`` and
    lets the caller continue (does not raise). Recording, and the run being
    recorded, both continue unaffected."""
    warnings.warn(
        f"agent-trace: {count} consecutive tool-call-only responses recorded "
        f"for host {host!r} — possible runaway tool-call loop (see "
        "https://github.com/langchain-ai/langgraph/issues/3097). Pass "
        "on_loop_detected=agent_trace.interceptor.httpx_hook."
        "raise_on_loop_detected to abort the run instead of warning.",
        stacklevel=3,
    )


def raise_on_loop_detected(host: str, count: int) -> None:
    """Opt-in ``on_loop_detected`` callback that raises
    :class:`RunawayToolCallLoopError` instead of warning — use this when you
    want a suspected runaway tool-call loop to actually stop the run rather
    than merely being logged."""
    raise RunawayToolCallLoopError(
        f"{count} consecutive tool-call-only responses recorded for host "
        f"{host!r} — aborting (see "
        "https://github.com/langchain-ai/langgraph/issues/3097)."
    )


class _LoopGuardMixin:
    """Shared consecutive-tool-call-only-response bookkeeping for
    RecordingTransport and AsyncRecordingTransport. Not a public class —
    both transports compose this state directly (see their __init__) rather
    than inheriting, so it stays a plain implementation detail."""

    def _init_loop_guard(
        self,
        loop_guard_threshold: int | None,
        on_loop_detected: Callable[[str, int], None] | None,
    ) -> None:
        self._loop_guard_threshold = loop_guard_threshold
        self._on_loop_detected = on_loop_detected or warn_on_loop_detected
        self._consecutive_tool_call_counts: dict[str, int] = {}

    def _check_loop_guard(self, url: str, response_body: str) -> None:
        if self._loop_guard_threshold is None:
            return
        host = _response_host(url)
        if _is_tool_call_only_response(response_body):
            count = self._consecutive_tool_call_counts.get(host, 0) + 1
            self._consecutive_tool_call_counts[host] = count
            if count >= self._loop_guard_threshold:
                self._on_loop_detected(host, count)
        else:
            self._consecutive_tool_call_counts[host] = 0


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


class RecordingTransport(httpx.BaseTransport, _LoopGuardMixin):
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
    loop_guard_threshold:
        Opt-in runaway-tool-call-loop guard, disabled by default (None).
        When set to an integer N, this transport counts *consecutive*
        tool-call-only responses (see ``_is_tool_call_only_response``)
        recorded for the same host and calls ``on_loop_detected(host,
        count)`` once that count reaches N — the live-recording signal for
        issue #3097 (a model that never stops emitting tool_calls, burning
        the full context window before anyone notices). A tool-call-only
        response is one whose assistant message carries tool call(s) and no
        other final content (OpenAI/Groq-style ``tool_calls`` with empty
        ``content``, or Anthropic ``stop_reason: "tool_use"`` with no text
        block) — any other response resets the count for that host to 0.
    on_loop_detected:
        Callback invoked as ``on_loop_detected(host, count)`` once
        ``loop_guard_threshold`` is reached. Defaults to
        :func:`warn_on_loop_detected` (emits a ``UserWarning``, does not
        interrupt the run). Pass :func:`raise_on_loop_detected` to raise
        :class:`~agent_trace.core.exceptions.RunawayToolCallLoopError`
        instead — the exchange is still recorded to the fixture before the
        exception propagates to the caller.
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: httpx.BaseTransport | None = None,
        *,
        stream: bool = False,
        loop_guard_threshold: int | None = None,
        on_loop_detected: Callable[[str, int], None] | None = None,
    ) -> None:
        self._fixture = fixture
        self._inner: httpx.BaseTransport = (
            inner if inner is not None else httpx.HTTPTransport()
        )
        self._stream = stream
        self._init_loop_guard(loop_guard_threshold, on_loop_detected)

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
                correlation_id=current_correlation_id(),
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
            correlation_id=current_correlation_id(),
        )
        self._check_loop_guard(url, resp_body)

        # Reconstruct so the caller receives a fully-read response with the
        # same status, headers, and body as the original. response.content is
        # already fully decoded, so the reconstructed Response must not carry
        # the original Content-Encoding/Content-Length headers (see
        # _strip_stale_body_headers).
        return httpx.Response(
            status_code=resp_status,
            headers=_strip_stale_body_headers(resp_headers),
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
            decoded_body = body_bytes.decode("utf-8", errors="replace")
            self._fixture.record_exchange(
                url=url,
                method=method,
                request_headers=req_headers,
                request_body=req_body,
                response_status=resp_status,
                response_headers=resp_headers,
                response_body=decoded_body,
                duration_ms=duration_ms,
                chunk_timestamps=chunk_offsets_s,
                correlation_id=current_correlation_id(),
            )
            self._check_loop_guard(url, decoded_body)

        tee = _TeeSyncByteStream(response, _on_complete)
        return httpx.Response(
            status_code=resp_status,
            headers=_strip_stale_body_headers(resp_headers),
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
            headers=_strip_stale_body_headers(exchange["response_headers"]),
            content=exchange["response_body"].encode("utf-8"),
            request=request,
        )

    def close(self) -> None:
        pass  # No resources to release; fixture lifecycle is managed externally.


class AsyncRecordingTransport(httpx.AsyncBaseTransport, _LoopGuardMixin):
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
    loop_guard_threshold, on_loop_detected:
        Same opt-in runaway-tool-call-loop guard as
        ``RecordingTransport`` — see that class's docstring.
    """

    def __init__(
        self,
        fixture: Fixture,
        inner: httpx.AsyncBaseTransport | None = None,
        *,
        stream: bool = False,
        loop_guard_threshold: int | None = None,
        on_loop_detected: Callable[[str, int], None] | None = None,
    ) -> None:
        self._fixture = fixture
        self._inner: httpx.AsyncBaseTransport = (
            inner if inner is not None else httpx.AsyncHTTPTransport()
        )
        self._stream = stream
        self._init_loop_guard(loop_guard_threshold, on_loop_detected)

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
                correlation_id=current_correlation_id(),
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
            correlation_id=current_correlation_id(),
        )
        self._check_loop_guard(url, resp_body)

        return httpx.Response(
            status_code=resp_status,
            headers=_strip_stale_body_headers(resp_headers),
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
            decoded_body = body_bytes.decode("utf-8", errors="replace")
            self._fixture.record_exchange(
                url=url,
                method=method,
                request_headers=req_headers,
                request_body=req_body,
                response_status=resp_status,
                response_headers=resp_headers,
                response_body=decoded_body,
                duration_ms=duration_ms,
                chunk_timestamps=chunk_offsets_s,
                correlation_id=current_correlation_id(),
            )
            self._check_loop_guard(url, decoded_body)

        tee = _TeeAsyncByteStream(response, _on_complete)
        return httpx.Response(
            status_code=resp_status,
            headers=_strip_stale_body_headers(resp_headers),
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
            headers=_strip_stale_body_headers(exchange["response_headers"]),
            content=exchange["response_body"].encode("utf-8"),
            request=request,
        )

    async def aclose(self) -> None:
        pass  # No resources to release; fixture lifecycle is managed externally.
