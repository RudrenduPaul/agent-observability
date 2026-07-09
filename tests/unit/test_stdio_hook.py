"""
Unit tests for agent_trace.interceptor.stdio_hook.

_classify_frame / _RecordingReceiveStream / _RecordingSendStream.

Uses the real `mcp` package's `mcp.types`/`mcp.shared.message` models
(introspected against the actually-installed SDK) so the frame-shape
assumptions these tests encode are verified against the real API, not a
guess. `mcp` is an optional dependency — these tests are skipped when it
isn't installed (matches the langgraph/openai-agents integration pattern).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp not installed")

from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.stdio_hook import (
    _classify_frame,
    _RecordingReceiveStream,
    _RecordingSendStream,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path / "mcp_test.db", trace_id="test-trace")


class _FakeInnerStream:
    """Minimal stand-in for an anyio MemoryObject{Receive,Send}Stream.

    Queues items for receive(), records everything passed to send().
    """

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self.sent: list = []
        self.closed = False

    async def receive(self):
        import anyio

        if not self._items:
            raise anyio.EndOfStream
        return self._items.pop(0)

    async def send(self, item) -> None:
        self.sent.append(item)

    async def aclose(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# _classify_frame — verified against real mcp.types shapes
# ---------------------------------------------------------------------------


class TestClassifyFrame:
    def test_classifies_request(self) -> None:
        from mcp import types

        root = types.JSONRPCRequest(
            jsonrpc="2.0", id=1, method="tools/list", params=None
        )
        frame_type, rpc_id, method = _classify_frame(root)
        assert frame_type == "request"
        assert rpc_id == "1"
        assert method == "tools/list"

    def test_classifies_notification(self) -> None:
        from mcp import types

        root = types.JSONRPCNotification(
            jsonrpc="2.0", method="notifications/initialized", params=None
        )
        frame_type, rpc_id, method = _classify_frame(root)
        assert frame_type == "notification"
        assert rpc_id is None
        assert method == "notifications/initialized"

    def test_classifies_response(self) -> None:
        from mcp import types

        root = types.JSONRPCResponse(jsonrpc="2.0", id=1, result={"tools": []})
        frame_type, rpc_id, method = _classify_frame(root)
        assert frame_type == "response"
        assert rpc_id == "1"
        assert method is None

    def test_classifies_error(self) -> None:
        from mcp import types

        root = types.JSONRPCError(
            jsonrpc="2.0",
            id=2,
            error=types.ErrorData(code=-32601, message="Method not found"),
        )
        frame_type, rpc_id, method = _classify_frame(root)
        assert frame_type == "error"
        assert rpc_id == "2"
        assert method is None

    def test_string_request_id_preserved(self) -> None:
        from mcp import types

        root = types.JSONRPCRequest(
            jsonrpc="2.0", id="req-abc", method="tools/call", params=None
        )
        _, rpc_id, _ = _classify_frame(root)
        assert rpc_id == "req-abc"


# ---------------------------------------------------------------------------
# _RecordingReceiveStream
# ---------------------------------------------------------------------------


class TestRecordingReceiveStream:
    async def test_records_response_frame_and_forwards_it(self, tmp_path) -> None:
        from mcp import types
        from mcp.shared.message import SessionMessage

        message = SessionMessage(
            types.JSONRPCMessage(
                types.JSONRPCResponse(jsonrpc="2.0", id=1, result={"tools": []})
            )
        )
        inner = _FakeInnerStream([message])
        fixture = _make_fixture(tmp_path)
        stream = _RecordingReceiveStream(inner, fixture, "my-server --flag")

        received = await stream.receive()

        assert received is message  # forwarded through unchanged
        assert fixture.mcp_frame_count() == 1
        frames = fixture.all_mcp_frames()
        assert frames[0]["direction"] == "from_server"
        assert frames[0]["frame_type"] == "response"
        assert frames[0]["rpc_id"] == "1"
        assert frames[0]["server_command"] == "my-server --flag"
        payload = json.loads(frames[0]["payload"])
        assert payload["result"] == {"tools": []}
        fixture.close()

    async def test_non_session_message_items_are_forwarded_without_recording(
        self, tmp_path
    ) -> None:
        # read_stream can carry bare Exceptions for malformed frames the real
        # client failed to parse — nothing to record on the wire for those.
        boom = ValueError("malformed JSONRPC")
        inner = _FakeInnerStream([boom])
        fixture = _make_fixture(tmp_path)
        stream = _RecordingReceiveStream(inner, fixture, "my-server")

        received = await stream.receive()

        assert received is boom
        assert fixture.mcp_frame_count() == 0
        fixture.close()

    async def test_async_iteration_stops_at_end_of_stream(self, tmp_path) -> None:
        from mcp import types
        from mcp.shared.message import SessionMessage

        message = SessionMessage(
            types.JSONRPCMessage(
                types.JSONRPCNotification(
                    jsonrpc="2.0", method="notifications/initialized", params=None
                )
            )
        )
        inner = _FakeInnerStream([message])
        fixture = _make_fixture(tmp_path)
        stream = _RecordingReceiveStream(inner, fixture, "my-server")

        collected = [item async for item in stream]

        assert collected == [message]
        assert fixture.mcp_frame_count() == 1
        assert fixture.all_mcp_frames()[0]["frame_type"] == "notification"
        fixture.close()

    async def test_aclose_closes_inner_stream(self, tmp_path) -> None:
        inner = _FakeInnerStream([])
        fixture = _make_fixture(tmp_path)
        stream = _RecordingReceiveStream(inner, fixture, "my-server")

        await stream.aclose()

        assert inner.closed is True
        fixture.close()


# ---------------------------------------------------------------------------
# _RecordingSendStream
# ---------------------------------------------------------------------------


class TestRecordingSendStream:
    async def test_records_request_frame_and_forwards_it(self, tmp_path) -> None:
        from mcp import types
        from mcp.shared.message import SessionMessage

        message = SessionMessage(
            types.JSONRPCMessage(
                types.JSONRPCRequest(
                    jsonrpc="2.0",
                    id=7,
                    method="tools/call",
                    params={"name": "search", "arguments": {"q": "agents"}},
                )
            )
        )
        inner = _FakeInnerStream([])
        fixture = _make_fixture(tmp_path)
        stream = _RecordingSendStream(inner, fixture, "my-server")

        await stream.send(message)

        assert inner.sent == [message]  # forwarded through unchanged
        assert fixture.mcp_frame_count() == 1
        frames = fixture.all_mcp_frames()
        assert frames[0]["direction"] == "to_server"
        assert frames[0]["frame_type"] == "request"
        assert frames[0]["method"] == "tools/call"
        assert frames[0]["rpc_id"] == "7"
        fixture.close()

    async def test_aclose_closes_inner_stream(self, tmp_path) -> None:
        inner = _FakeInnerStream([])
        fixture = _make_fixture(tmp_path)
        stream = _RecordingSendStream(inner, fixture, "my-server")

        await stream.aclose()

        assert inner.closed is True
        fixture.close()


# ---------------------------------------------------------------------------
# recording_stdio_client — requires _require_mcp() to succeed
# ---------------------------------------------------------------------------


class TestRequireMcp:
    def test_require_mcp_returns_module_when_installed(self) -> None:
        from agent_trace.interceptor.stdio_hook import _require_mcp

        mcp_module = _require_mcp()
        assert mcp_module.__name__ == "mcp"
