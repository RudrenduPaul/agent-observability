"""
MCP-aware integration module.

Today's two framework integrations (``langgraph.py``, ``openai_agents.py``)
both assume an agent/graph invocation is already underway when they attach —
they hook callbacks that fire *during* ``graph.invoke()``/``Runner.run()``.
MCP tool-loading failures frequently happen *before* any of that: at
``ClientSession``/``MultiServerMCPClient`` construction time, while the
client is initializing the connection or listing tools, with no agent
invocation in progress at all.

This module instruments an MCP client's lifecycle directly — independent of
whether an agent/graph invocation is running — by wrapping the instance
methods ``mcp.client.session.ClientSession`` actually exposes:
``initialize``, ``list_tools``, and ``call_tool``.

Usage (``ClientSession`` directly)::

    from mcp import ClientSession
    from agent_trace import Tracer
    from agent_trace.integrations.mcp import instrument_session

    t = Tracer()
    with t.start_trace("mcp_run") as trace:
        async with ClientSession(read_stream, write_stream) as session:
            instrument_session(session, tracer=t, trace=trace)
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("search", {"query": "agents"})

Usage (``MultiServerMCPClient`` from ``langchain-mcp-adapters``)::

    from langchain_mcp_adapters.client import MultiServerMCPClient
    from agent_trace.integrations.mcp import instrument_multi_server_client

    client = MultiServerMCPClient({...})
    instrument_multi_server_client(client, tracer=t, trace=trace)
    all_tools = await client.get_tools()  # now spanned + traces every session
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from agent_trace.core.span import SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = [
    "instrument_multi_server_client",
    "instrument_session",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Instance attribute set on any object this module has already wrapped, so
# repeated instrument_*() calls on the same instance are a no-op instead of
# double-wrapping (each wrap layer would otherwise double-count spans).
_INSTRUMENTED_MARKER = "_agent_trace_instrumented"


def _tool_names(list_tools_result: Any) -> list[str]:
    """Best-effort extraction of tool names from a ListToolsResult."""
    tools = getattr(list_tools_result, "tools", None) or []
    names: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if name is not None:
            names.append(str(name))
    return names


def _make_traced_initialize(tracer: Tracer, original: Callable[..., Any]) -> Any:
    """Build the ``mcp:initialize``-spanned replacement for ``session.initialize``."""

    async def traced_initialize(*args: Any, **kwargs: Any) -> Any:
        span = tracer.start_span("mcp:initialize")
        try:
            result = await original(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)
            raise
        try:
            protocol_version = getattr(result, "protocolVersion", None)
            if protocol_version is not None:
                span.set_attribute("mcp.protocol_version", str(protocol_version))
            server_info = getattr(result, "serverInfo", None)
            server_name = getattr(server_info, "name", None)
            if server_name is not None:
                span.set_attribute("mcp.server_name", str(server_name))
        except Exception:
            logger.debug(
                "agent-trace: failed to record MCP initialize result",
                exc_info=True,
            )
        span.end(SpanStatus.OK)
        return result

    return traced_initialize


def _make_traced_list_tools(tracer: Tracer, original: Callable[..., Any]) -> Any:
    """Build the ``mcp:list_tools``-spanned replacement for ``session.list_tools``."""

    async def traced_list_tools(*args: Any, **kwargs: Any) -> Any:
        span = tracer.start_span("mcp:list_tools")
        try:
            result = await original(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)
            raise
        try:
            names = _tool_names(result)
            span.set_attribute("mcp.tool_count", len(names))
            if names:
                span.set_attribute("mcp.tool_names", ",".join(names))
        except Exception:
            logger.debug(
                "agent-trace: failed to record MCP list_tools result",
                exc_info=True,
            )
        span.end(SpanStatus.OK)
        return result

    return traced_list_tools


def _make_traced_call_tool(tracer: Tracer, original: Callable[..., Any]) -> Any:
    """Build the ``mcp:tool:<name>``-spanned replacement for ``session.call_tool``."""

    async def traced_call_tool(
        name: str, arguments: dict[str, Any] | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        span = tracer.start_span(f"mcp:tool:{name}")
        span.set_attribute("tool.name", name)
        if arguments:
            span.set_attribute("tool.argument_keys", ",".join(sorted(arguments)))
        try:
            result = await original(name, arguments, *args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)
            raise
        try:
            is_error = bool(getattr(result, "isError", False))
            span.set_attribute("tool.is_error", is_error)
            span.end(SpanStatus.ERROR if is_error else SpanStatus.OK)
        except Exception:
            logger.debug(
                "agent-trace: failed to record MCP call_tool result",
                exc_info=True,
            )
            span.end(SpanStatus.OK)
        return result

    return traced_call_tool


def instrument_session(
    session: Any,
    *,
    tracer: Tracer,
    trace: Trace,
) -> Any:
    """Wrap a ``ClientSession``'s lifecycle methods to emit agent-trace spans.

    Patches the *instance* (not the class), so other, uninstrumented sessions
    are unaffected. Idempotent — calling this twice on the same session
    returns it unchanged the second time.

    Spans emitted:

    - ``mcp:initialize`` — around ``session.initialize()``; records the
      negotiated protocol version and server name once known.
    - ``mcp:list_tools`` — around ``session.list_tools()``; records the
      number of tools returned and their names.
    - ``mcp:tool:<name>`` — around each ``session.call_tool(name, ...)``;
      records the tool name, argument keys, and whether the call errored.

    Parameters
    ----------
    session:
        An ``mcp.ClientSession`` instance (or anything duck-typed the same
        way — ``initialize``/``list_tools``/``call_tool`` coroutine methods).
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
        Accepted for API symmetry with ``LangGraphTracer``/``AgentTraceHook``
        (spans attach to whichever trace is active via
        ``tracer.start_span`` — same as those integrations).

    Returns
    -------
    Any
        The same *session* instance, for chaining.
    """
    if getattr(session, _INSTRUMENTED_MARKER, False):
        return session

    session.initialize = _make_traced_initialize(tracer, session.initialize)
    session.list_tools = _make_traced_list_tools(tracer, session.list_tools)
    session.call_tool = _make_traced_call_tool(tracer, session.call_tool)
    setattr(session, _INSTRUMENTED_MARKER, True)
    return session


def instrument_multi_server_client(
    client: Any,
    *,
    tracer: Tracer,
    trace: Trace,
) -> Any:
    """Instrument a ``langchain_mcp_adapters.client.MultiServerMCPClient``.

    Wraps ``client.session()`` so that every ``ClientSession`` it yields is
    automatically passed through :func:`instrument_session` — covering both
    the "new session per tool call" usage (``client.get_tools()``) and the
    "explicit session" usage (``async with client.session(name) as s``),
    since both paths route through ``session()`` internally.

    Parameters
    ----------
    client:
        A ``MultiServerMCPClient`` instance (duck-typed: anything exposing an
        async-context-manager ``session(server_name)`` method).
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.

    Returns
    -------
    Any
        The same *client* instance, for chaining.
    """
    if getattr(client, _INSTRUMENTED_MARKER, False):
        return client

    original_session_cm: Callable[..., Any] = client.session

    def traced_session(*args: Any, **kwargs: Any) -> Any:
        # client.session() is itself an @asynccontextmanager — call it to get
        # the underlying async context manager, wrap what it yields.
        inner_cm = original_session_cm(*args, **kwargs)
        return _InstrumentingSessionContext(inner_cm, tracer=tracer, trace=trace)

    client.session = traced_session
    setattr(client, _INSTRUMENTED_MARKER, True)
    return client


class _InstrumentingSessionContext:
    """Async context manager that instruments whatever session it yields.

    Wraps the async context manager returned by
    ``MultiServerMCPClient.session()`` so the ``ClientSession`` it yields is
    run through :func:`instrument_session` before being handed to the caller.
    """

    def __init__(self, inner_cm: Any, *, tracer: Tracer, trace: Trace) -> None:
        self._inner_cm = inner_cm
        self._tracer = tracer
        self._trace = trace

    async def __aenter__(self) -> Any:
        session = await self._inner_cm.__aenter__()
        return instrument_session(session, tracer=self._tracer, trace=self._trace)

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._inner_cm.__aexit__(*exc_info)
