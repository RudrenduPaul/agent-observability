"""
Integration tests for the MCP stdio capture layer + MCP integration module.

Unlike the OpenAI Agents / LangGraph integration tests, this needs no live
API key: it spawns a real local MCP server subprocess (a tiny FastMCP stdio
server, tests/integration/_mcp_stdio_server.py) over the actual stdio
transport, so `recording_stdio_client` + `instrument_session` are exercised
against genuine JSON-RPC traffic end to end.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp not installed")

_SERVER_SCRIPT = Path(__file__).parent / "_mcp_stdio_server.py"


@pytest.mark.integration
class TestMCPStdioCapture:
    async def test_recording_stdio_client_captures_real_jsonrpc_frames(
        self, tmp_path: Path
    ) -> None:
        """initialize + list_tools + call_tool over a real subprocess, fully captured."""
        from mcp import ClientSession, StdioServerParameters

        from agent_trace._replay.fixture import Fixture
        from agent_trace.interceptor.stdio_hook import recording_stdio_client

        server = StdioServerParameters(
            command=sys.executable, args=[str(_SERVER_SCRIPT)]
        )
        fixture = Fixture(tmp_path / "mcp_fixture.db", trace_id="mcp-it-001")

        async with recording_stdio_client(server, fixture) as (read, write):
            async with ClientSession(read, write) as session:
                init_result = await session.initialize()
                tools_result = await session.list_tools()
                call_result = await session.call_tool("add", {"a": 2, "b": 3})

        # --- the real MCP round-trip actually worked ---
        assert init_result.serverInfo.name == "agent-trace-test-server"
        assert {t.name for t in tools_result.tools} == {"add"}
        assert call_result.isError is False

        # --- and every JSON-RPC frame was captured to the fixture ---
        frames = fixture.all_mcp_frames()
        assert len(frames) > 0

        methods_to_server = {
            f["method"] for f in frames if f["direction"] == "to_server" and f["method"]
        }
        assert "initialize" in methods_to_server
        assert "tools/list" in methods_to_server
        assert "tools/call" in methods_to_server

        response_frames = [f for f in frames if f["frame_type"] == "response"]
        assert len(response_frames) >= 3  # initialize, list_tools, call_tool

        # call_tool's request payload round-trips the real arguments sent.
        call_tool_requests = [
            f
            for f in frames
            if f["direction"] == "to_server" and f["method"] == "tools/call"
        ]
        assert len(call_tool_requests) == 1
        payload = json.loads(call_tool_requests[0]["payload"])
        assert payload["params"]["name"] == "add"
        assert payload["params"]["arguments"] == {"a": 2, "b": 3}

        # notifications (no id) are captured too — e.g. notifications/initialized.
        notifications = [f for f in frames if f["frame_type"] == "notification"]
        assert any(f["method"] == "notifications/initialized" for f in notifications)

        # server_command correctly identifies which subprocess this came from.
        assert all(sys.executable in f["server_command"] for f in frames)

        fixture.close()

    async def test_recording_stdio_client_is_transparent_to_replay_free_usage(
        self, tmp_path: Path
    ) -> None:
        """The wrapped streams behave exactly like the unwrapped ones to the caller."""
        from mcp import ClientSession, StdioServerParameters

        from agent_trace._replay.fixture import Fixture
        from agent_trace.interceptor.stdio_hook import recording_stdio_client

        server = StdioServerParameters(
            command=sys.executable, args=[str(_SERVER_SCRIPT)]
        )
        fixture = Fixture(tmp_path / "mcp_fixture2.db", trace_id="mcp-it-002")

        async with recording_stdio_client(server, fixture) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("add", {"a": 10, "b": 32})

        fixture.close()

        assert result.isError is False
        assert result.structuredContent == {"result": 42}


@pytest.mark.integration
class TestMCPIntegrationModule:
    async def test_instrument_session_emits_spans_for_real_mcp_session(
        self, tmp_path: Path
    ) -> None:
        """instrument_session against a real ClientSession over a real subprocess."""
        from mcp import ClientSession, StdioServerParameters

        from agent_trace import SpanStatus, Tracer
        from agent_trace._replay.fixture import Fixture
        from agent_trace.integrations.mcp import instrument_session
        from agent_trace.interceptor.stdio_hook import recording_stdio_client

        server = StdioServerParameters(
            command=sys.executable, args=[str(_SERVER_SCRIPT)]
        )
        fixture = Fixture(tmp_path / "mcp_fixture3.db", trace_id="mcp-it-003")
        tracer = Tracer(trace_dir=tmp_path)

        with tracer.start_trace("mcp_integration_test") as trace:
            async with recording_stdio_client(server, fixture) as (read, write):
                async with ClientSession(read, write) as session:
                    instrument_session(session, tracer=tracer, trace=trace)
                    await session.initialize()
                    await session.list_tools()
                    result = await session.call_tool("add", {"a": 5, "b": 5})

        assert result.structuredContent == {"result": 10}

        span_names = [s.name for s in trace.spans]
        assert "mcp:initialize" in span_names
        assert "mcp:list_tools" in span_names
        assert "mcp:tool:add" in span_names

        list_tools_span = next(s for s in trace.spans if s.name == "mcp:list_tools")
        assert list_tools_span.status == SpanStatus.OK
        assert list_tools_span.attributes["mcp.tool_count"] == 1
        assert list_tools_span.attributes["mcp.tool_names"] == "add"

        call_tool_span = next(s for s in trace.spans if s.name == "mcp:tool:add")
        assert call_tool_span.status == SpanStatus.OK
        assert call_tool_span.attributes["tool.is_error"] is False

        # Both capture layers agree this run happened: spans on the Trace,
        # raw JSON-RPC frames in the Fixture — independently verifiable.
        assert fixture.mcp_frame_count() > 0

        fixture.close()
