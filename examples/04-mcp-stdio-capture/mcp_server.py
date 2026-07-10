"""
Trivial stdio MCP server used by example.py in this directory.

Launched as a subprocess — not meant to be run directly (though `python
mcp_server.py` does work, and will just sit there talking JSON-RPC on
stdin/stdout until you Ctrl-C it).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agent-trace-example-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def broken_tool() -> int:
    """A tool that always raises, to demonstrate error-status capture."""
    raise ValueError("this tool is intentionally broken")


if __name__ == "__main__":
    mcp.run(transport="stdio")
