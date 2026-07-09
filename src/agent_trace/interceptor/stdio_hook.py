"""
MCP stdio-transport capture layer.

``agent-trace``'s only capture layer historically has been HTTP-level
(``httpx``/``requests`` — see ``httpx_hook.py``/``requests_patch.py``). MCP's
``stdio`` transport communicates over subprocess stdin/stdout pipes instead —
zero HTTP traffic touches the wire, so an MCP-stdio-related failure (e.g. a
startup/tool-loading crash) is completely invisible to those interceptors,
regardless of which framework wraps the MCP client.

``recording_stdio_client`` wraps ``mcp.client.stdio.stdio_client`` the same
way ``RecordingTransport`` wraps a real ``httpx`` transport: the real
subprocess/streams are used unmodified, every JSON-RPC frame that flows
through is persisted to a :class:`~agent_trace._replay.fixture.Fixture`
*before* being handed back to the caller, and the caller (typically an MCP
``ClientSession``) is none the wiser.

Usage::

    from mcp import ClientSession, StdioServerParameters
    from agent_trace.interceptor.stdio_hook import recording_stdio_client

    server = StdioServerParameters(command="my-mcp-server", args=[])
    async with recording_stdio_client(server, fixture) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from agent_trace._replay.fixture import Fixture

__all__ = ["recording_stdio_client"]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "MCP stdio recording requires the mcp package.\n"
    "Install it with:\n\n"
    "    pip install mcp\n"
)


def _require_mcp() -> Any:
    """Lazy import guard — raises a clear error if mcp is absent."""
    try:
        import mcp

        return mcp
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _classify_frame(root: Any) -> tuple[str, str | None, str | None]:
    """Classify a ``JSONRPCMessage.root`` union member.

    Returns ``(frame_type, rpc_id, method)`` where ``frame_type`` is one of
    ``"request"``, ``"notification"``, ``"response"``, ``"error"``, or
    ``"unknown"``.

    Distinguishing the four ``mcp.types`` JSON-RPC frame shapes:

    - ``JSONRPCRequest``:      has ``method`` *and* ``id``.
    - ``JSONRPCNotification``: has ``method``, no ``id``.
    - ``JSONRPCResponse``:     has ``id`` and ``result``, no ``method``.
    - ``JSONRPCError``:        has ``id`` and ``error``, no ``method``.
    """
    rpc_id = str(root.id) if getattr(root, "id", None) is not None else None
    method = getattr(root, "method", None)
    if getattr(root, "error", None) is not None:
        return "error", rpc_id, method
    if hasattr(root, "result"):
        return "response", rpc_id, method
    if method is not None and rpc_id is not None:
        return "request", rpc_id, method
    if method is not None:
        return "notification", rpc_id, method
    return "unknown", rpc_id, method  # pragma: no cover — defensive fallback


class _RecordingReceiveStream:
    """Tees every ``SessionMessage`` read from *inner* into a Fixture.

    Duck-types anyio's ``ObjectReceiveStream`` interface (``receive``,
    ``aclose``, ``__aiter__``/``__anext__``, async-context-manager) so it can
    be handed directly to ``ClientSession`` in place of the real stream.
    """

    def __init__(
        self,
        inner: Any,
        fixture: Fixture,
        server_command: str,
    ) -> None:
        self._inner = inner
        self._fixture = fixture
        self._server_command = server_command

    def __aiter__(self) -> _RecordingReceiveStream:
        return self

    async def __anext__(self) -> Any:
        import anyio

        try:
            return await self.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration from None

    async def receive(self) -> Any:
        item = await self._inner.receive()
        self._record(item)
        return item

    def _record(self, item: Any) -> None:
        from mcp.shared.message import SessionMessage

        # read_stream carries `SessionMessage | Exception` — malformed frames
        # the real client failed to parse arrive as bare exceptions; nothing
        # to record on the wire for those.
        if not isinstance(item, SessionMessage):
            return
        try:
            root = item.message.root
            frame_type, rpc_id, method = _classify_frame(root)
            payload = item.message.model_dump_json(by_alias=True, exclude_none=True)
            self._fixture.record_mcp_frame(
                server_command=self._server_command,
                direction="from_server",
                frame_type=frame_type,
                rpc_id=rpc_id,
                method=method,
                payload=payload,
            )
        except Exception:
            logger.debug(
                "agent-trace: failed to record inbound MCP frame", exc_info=True
            )

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _RecordingReceiveStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


class _RecordingSendStream:
    """Tees every ``SessionMessage`` sent through *inner* into a Fixture."""

    def __init__(
        self,
        inner: Any,
        fixture: Fixture,
        server_command: str,
    ) -> None:
        self._inner = inner
        self._fixture = fixture
        self._server_command = server_command

    async def send(self, item: Any) -> None:
        self._record(item)
        await self._inner.send(item)

    def _record(self, item: Any) -> None:
        from mcp.shared.message import SessionMessage

        if not isinstance(item, SessionMessage):
            return  # pragma: no cover — write_stream is SessionMessage-only
        try:
            root = item.message.root
            frame_type, rpc_id, method = _classify_frame(root)
            payload = item.message.model_dump_json(by_alias=True, exclude_none=True)
            self._fixture.record_mcp_frame(
                server_command=self._server_command,
                direction="to_server",
                frame_type=frame_type,
                rpc_id=rpc_id,
                method=method,
                payload=payload,
            )
        except Exception:
            logger.debug(
                "agent-trace: failed to record outbound MCP frame", exc_info=True
            )

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _RecordingSendStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


@asynccontextmanager
async def recording_stdio_client(
    server: Any,
    fixture: Fixture,
    errlog: TextIO | None = None,
) -> AsyncIterator[tuple[_RecordingReceiveStream, _RecordingSendStream]]:
    """Wrap ``mcp.client.stdio.stdio_client`` to record every JSON-RPC frame.

    Parameters
    ----------
    server:
        An ``mcp.StdioServerParameters`` instance describing the subprocess
        to launch (``command``, ``args``, ``env``, ``cwd``).
    fixture:
        Open :class:`~agent_trace._replay.fixture.Fixture` to record into.
    errlog:
        Passed through to ``stdio_client`` for the child process's stderr
        (defaults to ``sys.stderr``, matching the real client's default).

    Yields
    ------
    tuple[_RecordingReceiveStream, _RecordingSendStream]
        Drop-in replacements for the ``(read_stream, write_stream)`` pair
        ``stdio_client`` normally yields — pass these straight into
        ``mcp.ClientSession``.
    """
    _require_mcp()
    from mcp.client.stdio import stdio_client

    command = f"{server.command} {' '.join(server.args)}".strip()
    effective_errlog: TextIO = errlog if errlog is not None else sys.stderr

    async with stdio_client(server, errlog=effective_errlog) as (
        read_stream,
        write_stream,
    ):
        recording_read = _RecordingReceiveStream(read_stream, fixture, command)
        recording_write = _RecordingSendStream(write_stream, fixture, command)
        yield recording_read, recording_write
