"""
Trivial stdio MCP server used only as a subprocess fixture by
tests/integration/test_mcp.py — exposes one tool, `add`.

Not a test file itself (no test_ prefix) — launched via
`StdioServerParameters(command=sys.executable, args=[__file__])`.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agent-trace-test-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
