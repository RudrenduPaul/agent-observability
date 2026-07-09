"""
WebSocket connection wrappers for recording and replaying Realtime-API-style
duplex sessions (e.g. the OpenAI Agents SDK's Realtime API, which opens a
persistent ``websockets.asyncio.client.ClientConnection`` and exchanges many
JSON events over it — tool calls, handoffs, audio deltas — rather than the
discrete request/response model ``RecordingTransport``/``ReplayTransport``
(``httpx_hook.py``) assume).

RecordingWebSocketConnection wraps a real connection object (anything
exposing the ``websockets`` client protocol: async ``send``, async ``recv``,
async iteration, and ``close``): frames flow through unmodified in both
directions while each one is teed into the fixture store as it passes.

ReplayWebSocketConnection never touches the network. It serves previously
recorded inbound ("recv") frames from the fixture, in the order they were
captured, so a Realtime session can be replayed offline at zero API cost.
Frames sent by the caller during replay are accepted but not forwarded
anywhere — there is no live peer to send them to, matching how
``ReplayTransport`` ignores request bodies and only replays responses keyed
by (method, url).

RecordingConnect is a drop-in stand-in for ``websockets.connect(...)`` (the
callable this module patches) that supports both calling conventions the
``websockets`` library provides::

    ws = await websockets.connect(uri, **kwargs)
    async with websockets.connect(uri, **kwargs) as ws:

and returns a RecordingWebSocketConnection either way.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_trace._replay.fixture import Fixture

from agent_trace.core.exceptions import NetworkGuardError, guard_active

__all__ = [
    "NetworkGuardError",
    "RecordingConnect",
    "RecordingWebSocketConnection",
    "ReplayWebSocketConnection",
    "WebSocketReplayExhaustedError",
]

logger = logging.getLogger(__name__)

DIRECTION_SEND = "send"
DIRECTION_RECV = "recv"


class WebSocketReplayExhaustedError(EOFError):
    """Raised by :meth:`ReplayWebSocketConnection.recv` when no more recorded
    inbound frames remain and ``AGENT_TRACE_NETWORK_GUARD`` is not set.

    Unlike ``ReplayTransport``'s HTTP fallback (which can fall through to a
    real request), there is no live peer to fall back to for a duplex
    WebSocket session during replay, so this is always the terminal signal —
    the caller should treat it the way it would treat the underlying library's
    own "connection closed" exception.
    """


def _decode_for_recording(message: Any) -> tuple[str, str]:
    """Return (payload_as_text, frame_type) for a frame about to be recorded."""
    if isinstance(message, (bytes, bytearray)):
        return message.decode("utf-8", errors="replace"), "binary"
    return str(message), "text"


def _encode_from_frame(frame: dict[str, Any]) -> Any:
    """Inverse of _decode_for_recording — reconstruct the original message shape."""
    if frame["frame_type"] == "binary":
        return frame["payload"].encode("utf-8")
    return frame["payload"]


class RecordingWebSocketConnection:
    """Wraps a real WebSocket client connection, recording every frame.

    Duck-types the subset of ``websockets.asyncio.client.ClientConnection``
    that agent-trace's supported callers use — ``send``, ``recv``, async
    iteration (``async for message in ws``), and ``close`` — and forwards
    every other attribute access to the wrapped connection (e.g.
    ``close_code``, ``state``, ``request``, ``response``), so instrumented
    code can use this exactly like the real connection.

    Parameters
    ----------
    inner:
        The real, already-connected WebSocket client connection.
    fixture:
        Open Fixture instance where frames will be written.
    url:
        The URL the connection was opened against — stored alongside every
        frame so a fixture holding multiple connections can be filtered.
    connection_id:
        Identifier grouping frames from this connection in the fixture.
        Auto-generated if not supplied.
    """

    def __init__(
        self,
        inner: Any,
        fixture: Fixture,
        url: str,
        connection_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._fixture = fixture
        self._url = url
        self.connection_id: str = connection_id or uuid.uuid4().hex[:16]

    async def send(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Send *message*, recording it as an outbound ("send") frame first.

        Recording before the awaited send (rather than after) means a frame
        that raises mid-send (e.g. connection dropped) is still captured —
        matching the intent, if not the exact ordering, of
        ``RecordingTransport`` which always records a completed exchange.
        """
        payload, frame_type = _decode_for_recording(message)
        self._fixture.record_ws_frame(
            connection_id=self.connection_id,
            url=self._url,
            direction=DIRECTION_SEND,
            payload=payload,
            frame_type=frame_type,
        )
        await self._inner.send(message, *args, **kwargs)

    async def recv(self, *args: Any, **kwargs: Any) -> Any:
        """Receive the next message, recording it as an inbound ("recv") frame."""
        message = await self._inner.recv(*args, **kwargs)
        payload, frame_type = _decode_for_recording(message)
        self._fixture.record_ws_frame(
            connection_id=self.connection_id,
            url=self._url,
            direction=DIRECTION_RECV,
            payload=payload,
            frame_type=frame_type,
        )
        return message

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter_messages()

    async def _iter_messages(self) -> AsyncIterator[Any]:
        async for message in self._inner:
            payload, frame_type = _decode_for_recording(message)
            self._fixture.record_ws_frame(
                connection_id=self.connection_id,
                url=self._url,
                direction=DIRECTION_RECV,
                payload=payload,
                frame_type=frame_type,
            )
            yield message

    async def close(self, *args: Any, **kwargs: Any) -> None:
        await self._inner.close(*args, **kwargs)

    async def __aenter__(self) -> RecordingWebSocketConnection:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    def __getattr__(self, name: str) -> Any:
        # Passthrough for anything not explicitly wrapped above (close_code,
        # state, request, response, ping, pong, ...).
        return getattr(self._inner, name)


class ReplayWebSocketConnection:
    """Serves recorded WS frames from a Fixture without any network I/O.

    Parameters
    ----------
    fixture:
        Open Fixture instance to serve recorded frames from.
    url:
        The URL the connection would have been opened against — used only
        for NetworkGuardError messages.
    connection_id:
        Identifier selecting which connection's frames to serve. Must match
        the ``connection_id`` used when the session was recorded.
    clock:
        Optional FixtureClock to advance with each frame's recorded_at
        timestamp, reproducing original timing during replay.
    """

    def __init__(
        self,
        fixture: Fixture,
        url: str,
        connection_id: str,
        clock: Any | None = None,
    ) -> None:
        self._fixture = fixture
        self._url = url
        self.connection_id = connection_id
        self._clock = clock
        self.close_code: int | None = None

    async def send(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Accept a send during replay without forwarding it anywhere.

        There is no live peer in replay mode — this mirrors how
        ``ReplayTransport`` ignores the outbound request body and only ever
        replays canned responses.
        """
        return None

    async def recv(self, *args: Any, **kwargs: Any) -> Any:
        """Return the next recorded inbound frame, or raise/warn if exhausted."""
        frame = self._fixture.next_ws_frame(self.connection_id, DIRECTION_RECV)
        if frame is None:
            return self._on_exhausted()
        return self._deliver(frame)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter_messages()

    async def _iter_messages(self) -> AsyncIterator[Any]:
        while True:
            frame = self._fixture.next_ws_frame(self.connection_id, DIRECTION_RECV)
            if frame is None:
                # Mirrors a real connection closing: the async-for loop ends
                # normally, exactly like `_listen_for_messages`'s
                # `async for message in self._websocket:` does on
                # ConnectionClosedOK.
                return
            yield self._deliver(frame)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code

    async def __aenter__(self) -> ReplayWebSocketConnection:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    def _deliver(self, frame: dict[str, Any]) -> Any:
        if self._clock is not None:
            self._clock.advance(float(frame["recorded_at"]))
        return _encode_from_frame(frame)

    def _on_exhausted(self) -> Any:
        if guard_active():
            raise NetworkGuardError(
                f"No recorded inbound WS frame for connection "
                f"{self.connection_id!r} ({self._url}) and "
                "AGENT_TRACE_NETWORK_GUARD=1 is set.  Run in recording mode "
                "first to capture this session."
            )
        warnings.warn(
            f"agent-trace: no more recorded WS frames for connection "
            f"{self.connection_id!r} ({self._url}); there is no live "
            "connection to fall back to during replay.  Set "
            "AGENT_TRACE_NETWORK_GUARD=1 to make this an error.",
            stacklevel=2,
        )
        raise WebSocketReplayExhaustedError(
            f"agent-trace: WS fixture exhausted for connection "
            f"{self.connection_id!r} ({self._url})"
        )


class RecordingConnect:
    """Drop-in stand-in for ``websockets.connect(...)`` that records frames.

    Constructed with the *real* ``connect`` callable/class (so the patch
    installer keeps full control of what "real" means, exactly like
    ``RecordingTransport`` accepts an ``inner`` transport) plus the fixture
    to record into.  Supports both calling conventions ``websockets.connect``
    provides:

        ws = await websockets.connect(uri, **kwargs)
        async with websockets.connect(uri, **kwargs) as ws:

    and returns a :class:`RecordingWebSocketConnection` either way.
    """

    def __init__(
        self,
        real_connect: Any,
        fixture: Fixture,
        uri: str,
        *args: Any,
        connection_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._real = real_connect(uri, *args, **kwargs)
        self._fixture = fixture
        self._uri = uri
        self._connection_id = connection_id
        self._wrapped: RecordingWebSocketConnection | None = None

    def __await__(self) -> Any:
        return self._connect().__await__()

    async def _connect(self) -> RecordingWebSocketConnection:
        inner = await self._real
        self._wrapped = RecordingWebSocketConnection(
            inner, self._fixture, self._uri, connection_id=self._connection_id
        )
        return self._wrapped

    async def __aenter__(self) -> RecordingWebSocketConnection:
        return await self._connect()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._wrapped is not None:
            await self._wrapped.close()
