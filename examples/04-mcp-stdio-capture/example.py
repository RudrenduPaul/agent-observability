"""
MCP stdio capture example.
Run: uv run python examples/04-mcp-stdio-capture/example.py

No LLM API calls required — this spawns a real local MCP server (mcp_server.py
in this directory) over the actual stdio subprocess transport, so it shows
genuine JSON-RPC traffic, not a simulation.

What this demonstrates:

  1. `recording_stdio_client()` — captures every JSON-RPC frame that crosses
     the subprocess's stdin/stdout into fixture.db, the transport-level
     capture agent-trace's httpx/requests interceptors cannot see at all
     (MCP's stdio transport makes zero HTTP calls).
  2. `instrument_session()` — emits agent-trace spans for MCP lifecycle
     events (initialize / list_tools / call_tool) independent of whether any
     agent/graph invocation is running — MCP tool-loading failures often
     happen at client-construction time, before an agent even starts.
  3. Both capture layers running together: the span tree shows *what*
     happened at the tool-call level, fixture.db shows *exactly what bytes*
     went over the wire — including the one that fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_trace import tracer
from agent_trace._replay.fixture import Fixture
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.mcp import instrument_session
from agent_trace.interceptor.stdio_hook import recording_stdio_client

_SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"


async def run_mcp_session() -> None:
    from mcp import ClientSession, StdioServerParameters

    server = StdioServerParameters(command=sys.executable, args=[str(_SERVER_SCRIPT)])

    with tracer.start_trace("mcp_stdio_example") as trace:
        run_dir = Path.home() / ".agent-trace" / "runs" / trace.run_id
        fixture = Fixture(run_dir / "fixture.db", trace_id=trace.trace_id)

        async with recording_stdio_client(server, fixture) as (read, write):
            async with ClientSession(read, write) as session:
                instrument_session(session, tracer=tracer, trace=trace)

                print("Connecting to MCP server over stdio...")
                init_result = await session.initialize()
                print(f"Connected to: {init_result.serverInfo.name}")

                print("\nListing tools...")
                tools = await session.list_tools()
                print(f"Available tools: {[t.name for t in tools.tools]}")

                print("\nCalling add(2, 3)...")
                result = await session.call_tool("add", {"a": 2, "b": 3})
                print(f"Result: {result.structuredContent}")

                print("\nCalling broken_tool() — expected to fail...")
                broken_result = await session.call_tool("broken_tool", {})
                print(f"isError: {broken_result.isError}")

        print("\n--- Span tree ---")
        StdoutExporter().export(trace)

        print(f"\nTrace saved to: {run_dir}")
        print(f"MCP frames captured: {fixture.mcp_frame_count()}")
        fixture.close()


def main() -> None:
    import asyncio

    asyncio.run(run_mcp_session())


if __name__ == "__main__":
    main()
