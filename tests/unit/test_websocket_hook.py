"""
Unit tests for agent_trace.interceptor.websocket_hook.

RecordingWebSocketConnection / ReplayWebSocketConnection / RecordingConnect /
WebSocketReplayExhaustedError.

Covers both a lightweight fake connection (fast, no sockets) and a live
round-trip against the real ``websockets`` package (a local
``websockets.serve`` server), plus a live round-trip through the real
OpenAI Agents SDK's Realtime WebSocket model
(``agents.realtime.openai_realtime.OpenAIRealtimeWebSocketModel``) to prove
the patch actually intercepts the SDK's own connection path, not just a
hand-rolled call to ``websockets.connect``.

AGENT_TRACE_NETWORK_GUARD=1 is set by pytest env (pyproject.toml), so
ReplayWebSocketConnection raises NetworkGuardError (not
WebSocketReplayExhaustedError) on exhaustion unless a test explicitly overrides it.
"""

from __future__ import annotations

import pytest
import websockets

from agent_trace import Tracer
from agent_trace._replay.fixture import Fixture
from agent_trace.core.exceptions import NetworkGuardError
from agent_trace.interceptor.websocket_hook import (
    RecordingConnect,
    RecordingWebSocketConnection,
    ReplayWebSocketConnection,
    WebSocketReplayExhaustedError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture(tmp_path) -> Fixture:
    return Fixture(tmp_path / "test.db", trace_id="test-trace")


class _FakeConnection:
    """Minimal stand-in for a websockets ClientConnection.

    Records what was sent to it and yields a fixed queue of inbound
    messages, without touching any real socket.
    """

    def __init__(self, inbound: list) -> None:
        self.sent: list = []
        self._inbound = list(inbound)
        self.closed = False
        self.close_code: int | None = None
        # Arbitrary passthrough attribute to verify __getattr__ delegation.
        self.request = "fake-request-object"

    async def send(self, message) -> None:
        self.sent.append(message)

    async def recv(self) -> object:
        if not self._inbound:
            raise EOFError("no more fake messages")
        return self._inbound.pop(0)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        while self._inbound:
            yield self._inbound.pop(0)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code


# ---------------------------------------------------------------------------
# RecordingWebSocketConnection — fake inner connection
# ---------------------------------------------------------------------------


class TestRecordingWebSocketConnection:
    async def test_send_records_outbound_frame_and_forwards(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=[])
        conn = RecordingWebSocketConnection(inner, fixture, "wss://example.com/rt")

        await conn.send("hello")

        assert inner.sent == ["hello"]
        frames = fixture.all_ws_frames()
        assert len(frames) == 1
        assert frames[0]["direction"] == "send"
        assert frames[0]["payload"] == "hello"
        assert frames[0]["frame_type"] == "text"
        fixture.close()

    async def test_recv_records_inbound_frame_and_returns_it(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=['{"type": "session.created"}'])
        conn = RecordingWebSocketConnection(inner, fixture, "wss://example.com/rt")

        message = await conn.recv()

        assert message == '{"type": "session.created"}'
        frames = fixture.all_ws_frames()
        assert len(frames) == 1
        assert frames[0]["direction"] == "recv"
        assert frames[0]["payload"] == '{"type": "session.created"}'
        fixture.close()

    async def test_async_iteration_records_every_inbound_frame_in_order(
        self, tmp_path
    ) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=["event-1", "event-2", "event-3"])
        conn = RecordingWebSocketConnection(inner, fixture, "wss://example.com/rt")

        received = [message async for message in conn]

        assert received == ["event-1", "event-2", "event-3"]
        frames = fixture.all_ws_frames()
        assert [f["payload"] for f in frames] == ["event-1", "event-2", "event-3"]
        assert all(f["direction"] == "recv" for f in frames)
        fixture.close()

    async def test_binary_frame_recorded_with_binary_frame_type(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=[])
        conn = RecordingWebSocketConnection(inner, fixture, "wss://example.com/rt")

        await conn.send(b"\x00\x01audio-bytes")

        frames = fixture.all_ws_frames()
        assert frames[0]["frame_type"] == "binary"
        fixture.close()

    async def test_close_forwards_to_inner(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=[])
        conn = RecordingWebSocketConnection(inner, fixture, "wss://example.com/rt")

        await conn.close()

        assert inner.closed is True
        fixture.close()

    async def test_getattr_passthrough_to_inner(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=[])
        conn = RecordingWebSocketConnection(inner, fixture, "wss://example.com/rt")

        assert conn.request == "fake-request-object"
        fixture.close()

    async def test_send_and_recv_share_one_connection_id(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=["reply"])
        conn = RecordingWebSocketConnection(inner, fixture, "wss://example.com/rt")

        await conn.send("request")
        await conn.recv()

        frames = fixture.all_ws_frames()
        assert len(frames) == 2
        assert (
            frames[0]["connection_id"]
            == frames[1]["connection_id"]
            == conn.connection_id
        )
        fixture.close()

    async def test_async_context_manager_closes_on_exit(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = _FakeConnection(inbound=[])

        async with RecordingWebSocketConnection(
            inner, fixture, "wss://example.com/rt"
        ) as conn:
            await conn.send("hi")

        assert inner.closed is True
        fixture.close()


# ---------------------------------------------------------------------------
# ReplayWebSocketConnection
# ---------------------------------------------------------------------------


class TestReplayWebSocketConnection:
    async def test_recv_serves_recorded_frames_in_order(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        fixture.record_ws_frame("conn-1", "wss://x", "recv", "first")
        fixture.record_ws_frame("conn-1", "wss://x", "recv", "second")

        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-1")

        assert await replay.recv() == "first"
        assert await replay.recv() == "second"
        fixture.close()

    async def test_async_iteration_ends_when_frames_exhausted(self, tmp_path) -> None:
        """Mirrors a real connection closing: the async-for loop just ends."""
        fixture = _make_fixture(tmp_path)
        fixture.record_ws_frame("conn-1", "wss://x", "recv", "only-frame")

        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-1")
        received = [message async for message in replay]

        assert received == ["only-frame"]
        fixture.close()

    async def test_ignores_frames_from_other_connections(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        fixture.record_ws_frame("conn-A", "wss://x", "recv", "belongs-to-A")
        fixture.record_ws_frame("conn-B", "wss://x", "recv", "belongs-to-B")

        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-A")
        received = [message async for message in replay]

        assert received == ["belongs-to-A"]
        fixture.close()

    async def test_send_during_replay_is_accepted_but_not_forwarded(
        self, tmp_path
    ) -> None:
        fixture = _make_fixture(tmp_path)
        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-1")

        # Must not raise — there is no live peer during replay.
        await replay.send("whatever")
        fixture.close()

    async def test_recv_raises_network_guard_error_when_exhausted_and_guard_on(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")
        fixture = _make_fixture(tmp_path)
        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-1")

        with pytest.raises(NetworkGuardError):
            await replay.recv()
        fixture.close()

    async def test_recv_raises_replay_exhausted_when_guard_off(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "0")
        fixture = _make_fixture(tmp_path)
        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-1")

        with pytest.warns(UserWarning):
            with pytest.raises(WebSocketReplayExhaustedError):
                await replay.recv()
        fixture.close()

    async def test_binary_frame_round_trips_as_bytes(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        fixture.record_ws_frame(
            "conn-1", "wss://x", "recv", "audio-bytes", frame_type="binary"
        )

        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-1")
        message = await replay.recv()

        assert isinstance(message, bytes)
        assert message == b"audio-bytes"
        fixture.close()

    async def test_close_records_close_code(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        replay = ReplayWebSocketConnection(fixture, "wss://x", "conn-1")

        await replay.close(code=1000)

        assert replay.close_code == 1000
        fixture.close()


# ---------------------------------------------------------------------------
# RecordingConnect — both websockets.connect calling conventions
# ---------------------------------------------------------------------------


class _FakeAwaitableConnect:
    """Stands in for a `websockets.connect(...)` instance: awaitable and an
    async context manager, exactly like the real `connect` class."""

    def __init__(self, uri: str, *args, **kwargs) -> None:
        self.uri = uri
        self._conn = _FakeConnection(inbound=["greeting"])

    def __await__(self):
        async def _get():
            return self._conn

        return _get().__await__()

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._conn.close()


class TestRecordingConnect:
    async def test_await_form_returns_recording_wrapper(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)

        ws = await RecordingConnect(
            _FakeAwaitableConnect, fixture, "wss://example.com/rt"
        )

        assert isinstance(ws, RecordingWebSocketConnection)
        message = await ws.recv()
        assert message == "greeting"
        assert fixture.all_ws_frames()[0]["direction"] == "recv"
        fixture.close()

    async def test_async_with_form_closes_inner_on_exit(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        captured_inner: list = []

        async with RecordingConnect(
            _FakeAwaitableConnect, fixture, "wss://example.com/rt"
        ) as ws:
            captured_inner.append(ws._inner)
            await ws.send("hi")

        assert captured_inner[0].closed is True
        fixture.close()


# ---------------------------------------------------------------------------
# Live round-trip against the real `websockets` package
# ---------------------------------------------------------------------------


class TestLiveWebsocketsRoundTrip:
    async def test_recording_connect_captures_real_server_round_trip(
        self, tmp_path
    ) -> None:
        """Full stack: a real `websockets.serve` echo server, `websockets.connect`
        wrapped by `RecordingConnect`, frames landing in a real Fixture."""

        async def handler(server_conn) -> None:
            async for message in server_conn:
                await server_conn.send(f"echo:{message}")

        fixture = _make_fixture(tmp_path)

        async with websockets.serve(handler, "localhost", 0) as server:
            port = server.sockets[0].getsockname()[1]
            url = f"ws://localhost:{port}"

            ws = await RecordingConnect(websockets.connect, fixture, url)
            await ws.send("hello")
            reply = await ws.recv()
            await ws.close()

        assert reply == "echo:hello"
        frames = fixture.all_ws_frames()
        assert [f["direction"] for f in frames] == ["send", "recv"]
        assert frames[0]["payload"] == "hello"
        assert frames[1]["payload"] == "echo:hello"
        fixture.close()

    async def test_tracer_patches_websockets_connect_transparently(
        self, tmp_path
    ) -> None:
        """Tracer.start_trace(record=True) patches the module-level
        `websockets.connect` so unmodified caller code (like the OpenAI Agents
        SDK's Realtime model) is captured with zero code changes."""

        async def handler(server_conn) -> None:
            async for message in server_conn:
                await server_conn.send(f"got:{message}")

        tracer = Tracer(trace_dir=tmp_path / "runs")

        async with websockets.serve(handler, "localhost", 0) as server:
            port = server.sockets[0].getsockname()[1]
            url = f"ws://localhost:{port}"

            with tracer.start_trace("ws-trace", record=True) as trace:
                ws = await websockets.connect(url)
                assert isinstance(ws, RecordingWebSocketConnection)
                await ws.send("patched")
                reply = await ws.recv()
                await ws.close()

            run_id = trace.run_id

        assert reply == "got:patched"

        fixture = Fixture(tmp_path / "runs" / run_id / "fixture.db")
        assert fixture.ws_frame_count() == 2
        fixture.close()

    async def test_websockets_connect_restored_after_trace_exits(
        self, tmp_path
    ) -> None:
        """The monkeypatch must not leak past the `start_trace` context."""
        tracer = Tracer(trace_dir=tmp_path / "runs")
        original = websockets.connect

        with tracer.start_trace("ws-trace-2", record=True):
            assert websockets.connect is not original

        assert websockets.connect is original


# ---------------------------------------------------------------------------
# Live round-trip through the real OpenAI Agents SDK Realtime model
# ---------------------------------------------------------------------------


class TestOpenAIAgentsRealtimeSDKIntegration:
    """Proves the interceptor intercepts the SDK's own connection path
    (`OpenAIRealtimeWebSocketModel._create_websocket_connection`), not just a
    hand-written call to `websockets.connect` — this is the actual consumer
    named in the backlog item."""

    async def test_realtime_model_connection_is_captured_when_patched(
        self, tmp_path
    ) -> None:
        agents_realtime = pytest.importorskip("agents.realtime.openai_realtime")

        async def handler(server_conn) -> None:
            async for message in server_conn:
                await server_conn.send(f"sdk-echo:{message}")

        fixture = _make_fixture(tmp_path)

        async with websockets.serve(handler, "localhost", 0) as server:
            port = server.sockets[0].getsockname()[1]
            url = f"ws://localhost:{port}"

            orig_connect = websockets.connect
            try:
                websockets.connect = lambda uri, **kw: RecordingConnect(
                    orig_connect, fixture, uri, **kw
                )
                model = agents_realtime.OpenAIRealtimeWebSocketModel()
                conn = await model._create_websocket_connection(url=url, headers={})
                assert isinstance(conn, RecordingWebSocketConnection)

                await conn.send("hello-from-sdk")
                reply = await conn.recv()
                await conn.close()
            finally:
                websockets.connect = orig_connect

        assert reply == "sdk-echo:hello-from-sdk"
        frames = fixture.all_ws_frames()
        assert [f["payload"] for f in frames] == [
            "hello-from-sdk",
            "sdk-echo:hello-from-sdk",
        ]
        fixture.close()
