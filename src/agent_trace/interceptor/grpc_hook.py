"""
gRPC interceptors for recording and replaying LLM SDK traffic that goes over
the gRPC transport instead of HTTP (e.g. Vertex AI, and google-generativeai /
langchain-google-genai on paths that default ``transport`` to ``None``,
which resolves to ``grpc``/``grpc_asyncio`` rather than REST).

Why gRPC needs a module separate from httpx_hook.py / requests_patch.py
-------------------------------------------------------------------------
Google's Python client libraries build their gRPC channel via
``google.api_core.grpc_helpers.create_channel()`` (sync) or
``grpc_helpers_async.create_channel()`` (async), both of which call
``grpc.secure_channel(...)`` / ``grpc.aio.secure_channel(...)`` directly --
verified against the installed ``google-api-core`` package
(``grpc_helpers.py:378`` and ``grpc_helpers_async.py:307``: both do
``import grpc`` / ``from grpc import aio`` and then call the module-qualified
function). That path never touches ``httpx`` or ``requests``, so
agent-trace's existing interceptors record zero wire-level evidence for it.

Interception strategy
----------------------
``grpc.insecure_channel`` / ``grpc.secure_channel`` (and their ``grpc.aio``
equivalents) are plain module-level factory functions, not methods on a
shared base class the way ``httpx.Client.__init__`` is. We therefore
monkey-patch the *module attribute* (``grpc.secure_channel = patched``)
rather than a class method. This intercepts every caller that accesses the
function through the module object at call time (``import grpc;
grpc.secure_channel(...)``) -- confirmed to be exactly how
``google-api-core``'s ``grpc_helpers.py`` calls it. A call site that did
``from grpc import secure_channel`` before the patch was installed would
bypass it; no such call site was found in ``google-api-core``.

RPC-shape coverage
-------------------
Fully recorded/replayed (sync ``grpc`` and async ``grpc.aio``):
unary-unary (e.g. ``GenerateContent``) -- the shape non-streaming Gemini /
Vertex AI calls use.

Fully recorded/replayed (sync ``grpc`` only): unary-stream (e.g.
``StreamGenerateContent``) -- the shape streaming Gemini / Vertex AI chat
calls use.

NOT recorded: client-streaming and bidirectional-streaming RPCs.
``GRPCRecordingInterceptor`` does not implement
``StreamUnaryClientInterceptor`` / ``StreamStreamClientInterceptor``, so
``grpc.intercept_channel`` routes those calls straight to the real network,
unintercepted, per grpc's own fallback behaviour
(``grpc._interceptor._Channel.stream_unary`` / ``.stream_stream``: "if
isinstance(self._interceptor, grpc.StreamUnaryClientInterceptor): ... else:
return thunk(method)"). ``GRPCReplayInterceptor`` *does* implement both, but
only to raise a clear error rather than silently leaking a real network call
during what is supposed to be an offline replay. Async (``grpc.aio``)
streaming of any shape is not covered by this module at all -- Gemini/Vertex
AI's async streaming path is comparatively rare relative to sync, and out of
scope for this pass.
"""

from __future__ import annotations

import base64
import json
import logging
import warnings
from typing import TYPE_CHECKING, Any

import grpc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from agent_trace._replay.fixture import Fixture

from agent_trace.core.exceptions import NetworkGuardError, guard_active

__all__ = [
    "AsyncGRPCRecordingInterceptor",
    "AsyncGRPCReplayInterceptor",
    "GRPCRecordingInterceptor",
    "GRPCReplayInterceptor",
    "NetworkGuardError",
]

logger = logging.getLogger(__name__)

# Metadata keys used to smuggle protobuf reconstruction info through the
# fixture's response_headers column (which is otherwise a plain str->str
# metadata dict). Namespaced so they can never collide with a real gRPC
# trailing-metadata key (gRPC metadata keys are restricted to
# `[a-z0-9._-]+`, so an uppercase-and-underscore key like this is not a
# value a server could ever legitimately send).
_TYPE_KEY = "X-AGENT-TRACE-GRPC-RESPONSE-TYPE"
_KIND_KEY = "X-AGENT-TRACE-GRPC-KIND"
_KIND_UNARY = "unary"
_KIND_STREAM = "stream"

# Fixture.record_exchange()/next_exchange() upper-case whatever is passed as
# `method` (it's designed around HTTP verbs, which are conventionally
# upper-cased). A gRPC full method path (e.g.
# "/agenttrace.test.Echo/UnaryEcho") is case-sensitive, so it must NOT go in
# the `method` column -- it goes in `url` instead, alongside the target.
# `method` carries a constant RPC-shape marker instead (itself fine to
# upper-case, and useful as a coarse filter in fixture inspection tools).
_METHOD_UNARY_UNARY = "GRPC_UNARY_UNARY"
_METHOD_UNARY_STREAM = "GRPC_UNARY_STREAM"


def _exchange_url(target: str, method_path: str) -> str:
    return f"grpc://{target}{method_path}"


# int -> grpc.StatusCode lookup, built once, for reconstructing a status
# object from the int we persisted (Fixture.record_exchange only accepts a
# plain int for response_status).
_STATUS_BY_INT: dict[int, grpc.StatusCode] = {
    code.value[0]: code for code in grpc.StatusCode
}


def _status_from_int(value: int) -> grpc.StatusCode:
    return _STATUS_BY_INT.get(value, grpc.StatusCode.UNKNOWN)


def _method_name(method: bytes | str) -> str:
    """client_call_details.method is str on sync channels, bytes on aio."""
    if isinstance(method, bytes):
        return method.decode("utf-8", errors="replace")
    return method


def _metadata_to_dict(metadata: Sequence[Any] | None) -> dict[str, str]:
    """Flatten grpc metadata (list of (key, value) pairs) to a str dict."""
    if not metadata:
        return {}
    result: dict[str, str] = {}
    for item in metadata:
        key, value = item[0], item[1]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        result[str(key)] = str(value)
    return result


def _encode_message(message: Any) -> str:
    """Serialize a protobuf message to a fixture-storable (base64) string.

    Protobuf wire bytes are not valid UTF-8 in general, and the fixture's
    body columns are TEXT, so base64 round-trips exactly where a raw
    ``.decode("utf-8", errors="replace")`` (as httpx_hook.py uses for JSON
    HTTP bodies) would silently corrupt binary payloads.
    """
    return base64.b64encode(message.SerializeToString()).decode("ascii")


def _lookup_message_class(full_name: str) -> Any:
    """Resolve a protobuf message class from its fully-qualified proto name.

    Uses the default SymbolDatabase, which every generated protobuf module
    registers itself into at import time. This works as long as the SDK
    that produced the original message is importable in the replay
    environment too (it must be -- it's how the request message got built
    in the first place).
    """
    from google.protobuf import symbol_database

    return symbol_database.Default().GetSymbol(full_name)


def _decode_message(data: str, message_cls: Any) -> Any:
    msg = message_cls()
    msg.ParseFromString(base64.b64decode(data))
    return msg


def _build_response_from_exchange(exchange: dict[str, Any]) -> Any:
    type_name = exchange["response_headers"].get(_TYPE_KEY)
    if not type_name:
        raise NetworkGuardError(
            f"agent-trace: recorded gRPC exchange for {exchange['method']} is "
            "missing its response-type marker; it was likely recorded by an "
            "older/incompatible version of grpc_hook.py and cannot be replayed."
        )
    message_cls = _lookup_message_class(type_name)
    return _decode_message(exchange["response_body"], message_cls)


# ---------------------------------------------------------------------------
# Sync recording
# ---------------------------------------------------------------------------


class _RecordingStreamProxy:
    """Wraps a real streaming Call so consumption is recorded once it ends.

    Proxies iteration item-by-item (so the caller still streams normally,
    with no added buffering latency) while collecting every yielded message.
    Once the underlying iterator raises StopIteration, the collected
    messages plus the call's final status/trailing-metadata are persisted to
    the fixture exactly once via *on_complete*. All other attribute access
    (``cancel()``, ``code()`` mid-stream, etc.) is proxied straight through
    to the real call.
    """

    def __init__(
        self,
        call: Any,
        on_complete: Callable[[list[Any], Any], None],
    ) -> None:
        self._call = call
        self._iterator: Iterator[Any] = iter(call)
        self._items: list[Any] = []
        self._on_complete = on_complete
        self._done = False

    def __iter__(self) -> _RecordingStreamProxy:
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._iterator)
        except StopIteration:
            if not self._done:
                self._done = True
                self._on_complete(self._items, self._call)
            raise
        else:
            self._items.append(item)
            return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._call, name)


class GRPCRecordingInterceptor(
    grpc.UnaryUnaryClientInterceptor,  # type: ignore[misc]
    grpc.UnaryStreamClientInterceptor,  # type: ignore[misc]
):
    """Records every unary-unary and unary-stream gRPC exchange to a Fixture.

    Install via ``grpc.intercept_channel(channel,
    GRPCRecordingInterceptor(fixture, target))``. Client-streaming/bidi RPCs
    are intentionally not intercepted -- see the module docstring.

    Parameters
    ----------
    fixture:
        Open Fixture instance where exchanges will be written.
    target:
        The channel's target string (e.g. ``"generativelanguage.googleapis.com:443"``),
        stored as the fixture's ``url`` column so replay can look the
        exchange back up by (method, target).
    """

    def __init__(self, fixture: Fixture, target: str) -> None:
        self._fixture = fixture
        self._target = target

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: Any,
        request: Any,
    ) -> Any:
        """Forward the call, record once it resolves, return it unchanged."""
        call = continuation(client_call_details, request)
        # call.result() blocks until the RPC resolves (or raises grpc.RpcError
        # for a non-OK status). We only persist exchanges that resolve with a
        # concrete response -- mirroring RecordingTransport, which never sees
        # a Response object at all for a transport-level failure.
        response = call.result()
        self._record_unary(client_call_details, request, call, response)
        return call

    def _record_unary(
        self,
        client_call_details: Any,
        request: Any,
        call: Any,
        response: Any,
    ) -> None:
        status = call.code()
        resp_headers = _metadata_to_dict(call.trailing_metadata())
        resp_headers[_TYPE_KEY] = response.DESCRIPTOR.full_name
        resp_headers[_KIND_KEY] = _KIND_UNARY
        self._fixture.record_exchange(
            url=_exchange_url(self._target, _method_name(client_call_details.method)),
            method=_METHOD_UNARY_UNARY,
            request_headers=_metadata_to_dict(client_call_details.metadata),
            request_body=_encode_message(request),
            response_status=status.value[0] if status is not None else 0,
            response_headers=resp_headers,
            response_body=_encode_message(response),
        )

    def intercept_unary_stream(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: Any,
        request: Any,
    ) -> Any:
        """Forward the call and record the full stream once it's exhausted."""
        call = continuation(client_call_details, request)

        def _finish(items: list[Any], real_call: Any) -> None:
            self._record_stream(client_call_details, request, items, real_call)

        return _RecordingStreamProxy(call, _finish)

    def _record_stream(
        self,
        client_call_details: Any,
        request: Any,
        items: list[Any],
        call: Any,
    ) -> None:
        status = call.code()
        resp_headers = _metadata_to_dict(call.trailing_metadata())
        resp_headers[_KIND_KEY] = _KIND_STREAM
        if items:
            resp_headers[_TYPE_KEY] = items[0].DESCRIPTOR.full_name
        self._fixture.record_exchange(
            url=_exchange_url(self._target, _method_name(client_call_details.method)),
            method=_METHOD_UNARY_STREAM,
            request_headers=_metadata_to_dict(client_call_details.metadata),
            request_body=_encode_message(request),
            response_status=status.value[0] if status is not None else 0,
            response_headers=resp_headers,
            response_body=json.dumps([_encode_message(item) for item in items]),
        )


# ---------------------------------------------------------------------------
# Sync replay
# ---------------------------------------------------------------------------


class _ReplayUnaryCall(grpc.Call, grpc.Future):  # type: ignore[misc]
    """Minimal Call+Future so a replayed unary-unary response satisfies the
    ``call.result()`` contract that ``grpc._interceptor._UnaryUnaryMultiCallable``
    unconditionally invokes on whatever ``intercept_unary_unary`` returns.

    Modelled directly on grpc's own internal ``_interceptor._UnaryOutcome``.
    """

    def __init__(
        self,
        response: Any,
        status_code: int,
        trailing_metadata: Mapping[str, str],
    ) -> None:
        self._response = response
        self._status_code = status_code
        self._trailing_metadata = tuple(trailing_metadata.items())

    def initial_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._trailing_metadata

    def trailing_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._trailing_metadata

    def code(self) -> grpc.StatusCode:
        return _status_from_int(self._status_code)

    def details(self) -> str:
        return ""

    def is_active(self) -> bool:
        return False

    def time_remaining(self) -> float | None:
        return None

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return False

    def done(self) -> bool:
        return True

    def add_callback(self, callback: Callable[[], None]) -> bool:
        return False

    def result(self, timeout: float | None = None) -> Any:
        return self._response

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return None

    def traceback(self, timeout: float | None = None) -> Any:
        return None

    def add_done_callback(self, fn: Callable[[Any], None]) -> None:
        fn(self)


class GRPCReplayInterceptor(
    grpc.UnaryUnaryClientInterceptor,  # type: ignore[misc]
    grpc.UnaryStreamClientInterceptor,  # type: ignore[misc]
    grpc.StreamUnaryClientInterceptor,  # type: ignore[misc]
    grpc.StreamStreamClientInterceptor,  # type: ignore[misc]
):
    """Serves unary-unary/unary-stream gRPC calls from a Fixture, no network I/O.

    Implements ``StreamUnaryClientInterceptor``/``StreamStreamClientInterceptor``
    purely as a safety net: without them, grpc's own fallback would route
    client-streaming/bidi calls straight to the real network during what's
    meant to be an offline replay (see module docstring). Both raise
    immediately instead.
    """

    def __init__(self, fixture: Fixture, target: str) -> None:
        self._fixture = fixture
        self._target = target

    def _lookup(
        self, client_call_details: Any, kind_method: str
    ) -> dict[str, Any] | None:
        url = _exchange_url(self._target, _method_name(client_call_details.method))
        return self._fixture.next_exchange(url, kind_method)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: Any,
        request: Any,
    ) -> Any:
        exchange = self._lookup(client_call_details, _METHOD_UNARY_UNARY)
        if exchange is None:
            return self._fallback_unary(continuation, client_call_details, request)
        response = _build_response_from_exchange(exchange)
        return _ReplayUnaryCall(
            response,
            int(exchange["response_status"]),
            exchange["response_headers"],
        )

    def _fallback_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: Any,
        request: Any,
    ) -> Any:
        if guard_active():
            method = _method_name(client_call_details.method)
            raise NetworkGuardError(
                f"No recorded gRPC exchange for {method} on {self._target} and "
                "AGENT_TRACE_NETWORK_GUARD=1 is set. Run in recording mode "
                "first to capture this call."
            )
        warnings.warn(
            f"agent-trace: no fixture entry for gRPC call "
            f"{_method_name(client_call_details.method)} on {self._target}; "
            "falling through to live network. Set AGENT_TRACE_NETWORK_GUARD=1 "
            "to make this an error.",
            stacklevel=2,
        )
        return continuation(client_call_details, request)

    def intercept_unary_stream(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: Any,
        request: Any,
    ) -> Any:
        exchange = self._lookup(client_call_details, _METHOD_UNARY_STREAM)
        if exchange is None:
            if guard_active():
                raise NetworkGuardError(
                    f"No recorded gRPC exchange for "
                    f"{_method_name(client_call_details.method)} on {self._target} and "
                    "AGENT_TRACE_NETWORK_GUARD=1 is set. Run in recording mode first "
                    "to capture this call."
                )
            warnings.warn(
                f"agent-trace: no fixture entry for gRPC stream "
                f"{_method_name(client_call_details.method)} on {self._target}; "
                "falling through to live network. Set AGENT_TRACE_NETWORK_GUARD=1 "
                "to make this an error.",
                stacklevel=2,
            )
            return continuation(client_call_details, request)

        raw_items: list[str] = json.loads(exchange["response_body"])
        if not raw_items:
            return iter(())
        type_name = exchange["response_headers"].get(_TYPE_KEY)
        message_cls = _lookup_message_class(type_name) if type_name else None
        if message_cls is None:
            raise NetworkGuardError(
                f"agent-trace: recorded gRPC stream exchange for "
                f"{_method_name(client_call_details.method)} is missing its "
                "response-type marker and cannot be replayed."
            )
        return iter(_decode_message(item, message_cls) for item in raw_items)

    def intercept_stream_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: Any,
        request_iterator: Any,
    ) -> Any:
        raise NotImplementedError(
            "agent-trace: gRPC client-streaming replay is not supported yet "
            f"({_method_name(client_call_details.method)}). Recording is not "
            "supported for this RPC shape either, so no fixture data could "
            "exist for it; falling through to the real network during replay "
            "would defeat the point of replay, so this raises instead."
        )

    def intercept_stream_stream(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: Any,
        request_iterator: Any,
    ) -> Any:
        raise NotImplementedError(
            "agent-trace: gRPC bidirectional-streaming replay is not "
            f"supported yet ({_method_name(client_call_details.method)}). "
            "Recording is not supported for this RPC shape either, so no "
            "fixture data could exist for it; falling through to the real "
            "network during replay would defeat the point of replay, so "
            "this raises instead."
        )


# ---------------------------------------------------------------------------
# Async (grpc.aio) recording / replay -- unary-unary only, see module
# docstring for why streaming shapes aren't covered here.
# ---------------------------------------------------------------------------


def _get_aio_unary_unary_interceptor_base() -> Any:
    """Lazy import: grpc.aio pulls in the asyncio C-extension bits, which we
    don't want to force-import for consumers who only use sync grpc.
    """
    from grpc import aio

    return aio.UnaryUnaryClientInterceptor


class AsyncGRPCRecordingInterceptor:
    """Records unary-unary grpc.aio exchanges to a Fixture.

    Not defined as a direct subclass of ``grpc.aio.UnaryUnaryClientInterceptor``
    at import time so that importing this module never requires grpc's aio
    extra to be importable; the real base class is spliced in via
    :func:`_recording_interceptor_class` the first time an instance is built.
    """

    def __new__(cls, fixture: Fixture, target: str) -> AsyncGRPCRecordingInterceptor:
        impl_cls = _recording_interceptor_class()
        return impl_cls(fixture, target)  # type: ignore[no-any-return]


class AsyncGRPCReplayInterceptor:
    """Serves unary-unary grpc.aio calls from a Fixture, no network I/O."""

    def __new__(cls, fixture: Fixture, target: str) -> AsyncGRPCReplayInterceptor:
        impl_cls = _replay_interceptor_class()
        return impl_cls(fixture, target)  # type: ignore[no-any-return]


_AsyncGRPCRecordingImpl: type | None = None
_AsyncGRPCReplayImpl: type | None = None


def _recording_interceptor_class() -> type:
    global _AsyncGRPCRecordingImpl  # noqa: PLW0603
    if _AsyncGRPCRecordingImpl is not None:
        return _AsyncGRPCRecordingImpl

    base_cls = _get_aio_unary_unary_interceptor_base()

    class _Impl(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, fixture: Fixture, target: str) -> None:
            self._fixture = fixture
            self._target = target

        async def intercept_unary_unary(
            self,
            continuation: Callable[[Any, Any], Any],
            client_call_details: Any,
            request: Any,
        ) -> Any:
            call = await continuation(client_call_details, request)
            response = await call
            status = await call.code()
            trailing_metadata = await call.trailing_metadata()
            resp_headers = _metadata_to_dict(trailing_metadata)
            resp_headers[_TYPE_KEY] = response.DESCRIPTOR.full_name
            resp_headers[_KIND_KEY] = _KIND_UNARY
            self._fixture.record_exchange(
                url=_exchange_url(
                    self._target, _method_name(client_call_details.method)
                ),
                method=_METHOD_UNARY_UNARY,
                request_headers=_metadata_to_dict(client_call_details.metadata),
                request_body=_encode_message(request),
                response_status=status.value[0] if status is not None else 0,
                response_headers=resp_headers,
                response_body=_encode_message(response),
            )
            return response

    _AsyncGRPCRecordingImpl = _Impl
    return _Impl


def _replay_interceptor_class() -> type:
    global _AsyncGRPCReplayImpl  # noqa: PLW0603
    if _AsyncGRPCReplayImpl is not None:
        return _AsyncGRPCReplayImpl

    base_cls = _get_aio_unary_unary_interceptor_base()

    class _Impl(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, fixture: Fixture, target: str) -> None:
            self._fixture = fixture
            self._target = target

        async def intercept_unary_unary(
            self,
            continuation: Callable[[Any, Any], Any],
            client_call_details: Any,
            request: Any,
        ) -> Any:
            method = _method_name(client_call_details.method)
            url = _exchange_url(self._target, method)
            exchange = self._fixture.next_exchange(url, _METHOD_UNARY_UNARY)
            if exchange is None:
                if guard_active():
                    raise NetworkGuardError(
                        f"No recorded gRPC exchange for {method} on "
                        f"{self._target} and AGENT_TRACE_NETWORK_GUARD=1 is "
                        "set. Run in recording mode first to capture this call."
                    )
                warnings.warn(
                    f"agent-trace: no fixture entry for gRPC call {method} on "
                    f"{self._target}; falling through to live network. Set "
                    "AGENT_TRACE_NETWORK_GUARD=1 to make this an error.",
                    stacklevel=2,
                )
                call = await continuation(client_call_details, request)
                return await call
            return _build_response_from_exchange(exchange)

    _AsyncGRPCReplayImpl = _Impl
    return _Impl
