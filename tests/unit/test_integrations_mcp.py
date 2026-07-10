"""
Unit tests for agent_trace.integrations.mcp.

instrument_session / instrument_multi_server_client.

Tests do NOT require the mcp / langchain-mcp-adapters packages — instrument_
session()/instrument_multi_server_client() are pure duck-typed instance
monkeypatches (mirroring AgentTraceHook's design), so plain mock objects
exercise them exactly the way a real ClientSession/MultiServerMCPClient
would be exercised. Real-package end-to-end coverage lives in
tests/integration/test_mcp.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.mcp import (
    instrument_multi_server_client,
    instrument_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracer_and_trace(tmp_path: Path):
    t = Tracer(trace_dir=tmp_path)
    cm = t.start_trace("mcp-unit-test")
    trace = cm.__enter__()
    return t, trace, cm


def _fake_initialize_result(
    protocol_version: str = "2025-06-18", server_name: str = "my-server"
) -> MagicMock:
    result = MagicMock()
    result.protocolVersion = protocol_version
    result.serverInfo = MagicMock()
    result.serverInfo.name = server_name
    return result


def _fake_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _fake_list_tools_result(names: list[str]) -> MagicMock:
    result = MagicMock()
    result.tools = [_fake_tool(n) for n in names]
    return result


def _fake_call_tool_result(is_error: bool = False) -> MagicMock:
    result = MagicMock()
    result.isError = is_error
    return result


def _fake_session() -> MagicMock:
    # spec= restricts attribute access to exactly these names, so
    # getattr(session, "_agent_trace_instrumented", False) correctly falls
    # back to False (a bare MagicMock() auto-vivifies *any* attribute access,
    # which would make that getattr() call always return a truthy mock).
    session = MagicMock(spec=["initialize", "list_tools", "call_tool"])
    session.initialize = AsyncMock(return_value=_fake_initialize_result())
    session.list_tools = AsyncMock(
        return_value=_fake_list_tools_result(["search", "fetch"])
    )
    session.call_tool = AsyncMock(return_value=_fake_call_tool_result())
    return session


# ---------------------------------------------------------------------------
# instrument_session — initialize
# ---------------------------------------------------------------------------


class TestInstrumentSessionInitialize:
    async def test_wraps_initialize_and_creates_span(self, tmp_path: Path) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        instrument_session(session, tracer=t, trace=trace)

        result = await session.initialize()
        cm.__exit__(None, None, None)

        assert result.protocolVersion == "2025-06-18"
        spans = [s for s in trace.spans if s.name == "mcp:initialize"]
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.OK
        assert spans[0].attributes["mcp.protocol_version"] == "2025-06-18"
        assert spans[0].attributes["mcp.server_name"] == "my-server"

    async def test_initialize_error_records_exception_and_reraises(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        session.initialize = AsyncMock(side_effect=RuntimeError("handshake failed"))
        instrument_session(session, tracer=t, trace=trace)

        with pytest.raises(RuntimeError, match="handshake failed"):
            await session.initialize()
        cm.__exit__(None, None, None)

        spans = [s for s in trace.spans if s.name == "mcp:initialize"]
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.ERROR
        exception_events = [e for e in spans[0].events if e.name == "exception"]
        assert exception_events[0].attributes["exception.message"] == (
            "handshake failed"
        )


# ---------------------------------------------------------------------------
# instrument_session — list_tools
# ---------------------------------------------------------------------------


class TestInstrumentSessionListTools:
    async def test_wraps_list_tools_and_records_tool_names(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        instrument_session(session, tracer=t, trace=trace)

        result = await session.list_tools()
        cm.__exit__(None, None, None)

        assert [tool.name for tool in result.tools] == ["search", "fetch"]
        spans = [s for s in trace.spans if s.name == "mcp:list_tools"]
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.OK
        assert spans[0].attributes["mcp.tool_count"] == 2
        assert spans[0].attributes["mcp.tool_names"] == "search,fetch"

    async def test_list_tools_empty_result_records_zero_count(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        session.list_tools = AsyncMock(return_value=_fake_list_tools_result([]))
        instrument_session(session, tracer=t, trace=trace)

        await session.list_tools()
        cm.__exit__(None, None, None)

        spans = [s for s in trace.spans if s.name == "mcp:list_tools"]
        assert spans[0].attributes["mcp.tool_count"] == 0
        assert "mcp.tool_names" not in spans[0].attributes


# ---------------------------------------------------------------------------
# instrument_session — call_tool
# ---------------------------------------------------------------------------


class TestInstrumentSessionCallTool:
    async def test_wraps_call_tool_success(self, tmp_path: Path) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        instrument_session(session, tracer=t, trace=trace)

        result = await session.call_tool("search", {"query": "agents"})
        cm.__exit__(None, None, None)

        assert result.isError is False
        spans = [s for s in trace.spans if s.name == "mcp:tool:search"]
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.OK
        assert spans[0].attributes["tool.name"] == "search"
        assert spans[0].attributes["tool.argument_keys"] == "query"
        assert spans[0].attributes["tool.is_error"] is False

    async def test_call_tool_marks_span_error_when_result_is_error(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        session.call_tool = AsyncMock(return_value=_fake_call_tool_result(True))
        instrument_session(session, tracer=t, trace=trace)

        result = await session.call_tool("search", {"query": "agents"})
        cm.__exit__(None, None, None)

        assert result.isError is True
        spans = [s for s in trace.spans if s.name == "mcp:tool:search"]
        assert spans[0].status == SpanStatus.ERROR
        assert spans[0].attributes["tool.is_error"] is True

    async def test_call_tool_raising_records_exception(self, tmp_path: Path) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        session.call_tool = AsyncMock(side_effect=RuntimeError("transport closed"))
        instrument_session(session, tracer=t, trace=trace)

        with pytest.raises(RuntimeError, match="transport closed"):
            await session.call_tool("search", {"query": "agents"})
        cm.__exit__(None, None, None)

        spans = [s for s in trace.spans if s.name == "mcp:tool:search"]
        assert spans[0].status == SpanStatus.ERROR

    async def test_call_tool_forwards_positional_and_keyword_args(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        original_call_tool = session.call_tool
        instrument_session(session, tracer=t, trace=trace)

        await session.call_tool("search", {"q": "x"}, meta={"trace": "abc"})
        cm.__exit__(None, None, None)

        original_call_tool.assert_awaited_once_with(
            "search", {"q": "x"}, meta={"trace": "abc"}
        )


# ---------------------------------------------------------------------------
# instrument_session — idempotency
# ---------------------------------------------------------------------------


class TestInstrumentSessionIdempotent:
    async def test_double_instrumentation_does_not_double_wrap(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        session = _fake_session()
        instrument_session(session, tracer=t, trace=trace)
        wrapped_once = session.initialize
        instrument_session(session, tracer=t, trace=trace)
        cm.__exit__(None, None, None)

        assert session.initialize is wrapped_once


# ---------------------------------------------------------------------------
# instrument_multi_server_client
# ---------------------------------------------------------------------------


class _FakeSessionContext:
    """Async context manager returning a fresh fake session, like
    MultiServerMCPClient.session()'s @asynccontextmanager-decorated method.
    """

    def __init__(self, session: MagicMock) -> None:
        self._session = session

    async def __aenter__(self) -> MagicMock:
        return self._session

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


class TestInstrumentMultiServerClient:
    async def test_session_yielded_through_client_is_instrumented(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        underlying_session = _fake_session()
        client = MagicMock(spec=["session"])
        client.session = MagicMock(
            return_value=_FakeSessionContext(underlying_session)
        )

        instrument_multi_server_client(client, tracer=t, trace=trace)

        async with client.session("math") as session:
            await session.initialize()
        cm.__exit__(None, None, None)

        spans = [s for s in trace.spans if s.name == "mcp:initialize"]
        assert len(spans) == 1

    async def test_double_instrumentation_does_not_double_wrap(
        self, tmp_path: Path
    ) -> None:
        t, trace, cm = _tracer_and_trace(tmp_path)
        client = MagicMock(spec=["session"])
        client.session = MagicMock(
            return_value=_FakeSessionContext(_fake_session())
        )

        instrument_multi_server_client(client, tracer=t, trace=trace)
        wrapped_once = client.session
        instrument_multi_server_client(client, tracer=t, trace=trace)
        cm.__exit__(None, None, None)

        assert client.session is wrapped_once
