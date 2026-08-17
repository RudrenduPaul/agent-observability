"""MCP server wrapper for the agent-trace CLI (agent-observability-trace-cli).

Exposes a single generic tool, `run`, that shells out to the installed
`agent-trace` console script with `--json` appended, parses the JSON it
prints to stdout, and returns the parsed object as the tool result. This is
the generic subprocess-spawn-and-JSON-parse MCP wrapper pattern (one tool
per CLI, description sourced from the CLI's own --help rather than
hand-written) meant to be reusable across other CLI packages in the
portfolio with only the _CLI_BIN name changing.

Transport: stdio -- the MCP client spawns this module as a subprocess and
speaks JSON-RPC 2.0 over stdin/stdout, so stdout is reserved for the
protocol. Any diagnostic output from this wrapper goes to stderr; print()
is never used here for that reason.

Run directly (once installed via the `mcp` optional dependency group,
`pip install agent-observability-trace-cli[mcp]`):
    agent-trace-mcp
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from mcp.server import MCPServer

from agent_trace import __version__

_CLI_BIN = "agent-trace"

# Subcommands that argparse actually accepts --json on today (see
# src/agent_trace/_cli.py's `main()` — only show/version have no --json
# flag and will error with "unrecognized arguments: --json" if it's
# appended). Surfaced in the tool description below so a calling agent knows
# which subcommands are safe to invoke through this generic wrapper.
_JSON_CAPABLE_SUBCOMMANDS = ("list", "replay", "inspect", "diff", "run")


def _capture_cli_help() -> str:
    """Return `agent-trace --help` output, used as the `run` tool's
    description so the exposed subcommand/flag list stays in sync with the
    real CLI instead of being hand-copied and drifting out of date. Falls
    back to a short static description if the CLI isn't resolvable on PATH
    for some reason."""
    try:
        result = subprocess.run(  # noqa: S603
            [_CLI_BIN, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        help_text = (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"agent-trace-mcp: could not capture --help ({exc})", file=sys.stderr)
        help_text = ""

    if not help_text:
        help_text = "Run the agent-trace CLI (agent-observability-trace-cli)."

    json_capable = ", ".join(_JSON_CAPABLE_SUBCOMMANDS)
    return (
        "Run the agent-trace CLI with the given subcommand/args; --json is "
        "appended automatically so the result is machine-readable JSON. "
        f"Only these subcommands support --json today: {json_capable}. "
        "Real `agent-trace --help` output:\n\n"
        f"{help_text}"
    )


mcp = MCPServer(
    name="agent-trace",
    version=__version__,
    instructions=(
        "Wraps the agent-trace CLI (agent-observability-trace-cli): AI "
        "agent observability with deterministic record/replay. Use the "
        "`run` tool with a subcommand and its args, e.g. "
        'run(["list"]) or run(["inspect", "<run_id>"]).'
    ),
)


@mcp.tool(description=_capture_cli_help())
def run(args: list[str]) -> dict[str, Any]:
    """Shell out to `agent-trace <args...> --json` and return the parsed
    JSON result as a dict. Errors (non-zero exit, unparsable stdout) are
    returned as a dict with an "error" key rather than raised, so a calling
    agent gets a structured result either way.

    Example: run(["list"]) -> {"trace_dir": "...", "runs": [...]}
    """
    command = [_CLI_BIN, *args, "--json"]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"failed to exec {command!r}: {exc}"}

    if result.returncode != 0:
        return {
            "error": f"{_CLI_BIN} exited with code {result.returncode}",
            "stderr": result.stderr.strip(),
            "command": command,
        }

    try:
        parsed: dict[str, Any] = json.loads(result.stdout)
        return parsed
    except json.JSONDecodeError as exc:
        return {
            "error": f"could not parse JSON output: {exc}",
            "stdout": result.stdout,
            "command": command,
        }


def main() -> None:
    """Console-script entry point (`agent-trace-mcp`). Runs the server over
    stdio -- nothing in this module or anything it imports may print to
    stdout, since stdout is the JSON-RPC transport once this is running."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
