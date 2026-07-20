# Example 04 — MCP stdio Capture

Shows agent-trace's two MCP-specific capture layers running together against
a **real** local MCP server subprocess — no LLM API key needed, no network
calls, no mocking. Everything in this example is genuine JSON-RPC traffic
over a real stdin/stdout pipe.

## Why this exists

agent-trace's original capture layer is HTTP-level (`httpx`/`requests`). MCP's
`stdio` transport talks over subprocess pipes instead — zero HTTP traffic
touches the wire, so an MCP-stdio-related failure (a startup crash, a
tool-loading error, a malformed tool-call response) was previously invisible
to agent-trace no matter which framework wrapped the MCP client.

## What it demonstrates

- **`recording_stdio_client()`** (`agent_trace.interceptor.stdio_hook`) —
  wraps `mcp.client.stdio.stdio_client` the same way `RecordingTransport`
  wraps a real `httpx` transport: every JSON-RPC frame that crosses the
  subprocess's stdin/stdout is persisted to `fixture.db` before being handed
  back to the caller.
- **`instrument_session()`** (`agent_trace.integrations.mcp`) — wraps a
  `ClientSession`'s `initialize`/`list_tools`/`call_tool` methods to emit
  spans, independent of whether any agent/graph invocation is running. MCP
  tool-loading failures frequently happen at client-construction time,
  before an agent even starts — this is the capture layer for that case.
- Both layers running together on the **same run**: the span tree answers
  "what happened, and did it fail", `fixture.db` answers "what bytes
  actually went over the wire" — including the one that failed.

## The server

`mcp_server.py` in this directory is a tiny `FastMCP` stdio server with two
tools:

- `add(a, b)` — succeeds
- `broken_tool()` — always raises, to show error-status capture end to end

## How to run

```bash
# From the repo root:
pip install agent-observability-trace-cli[mcp]
uv run python examples/04-mcp-stdio-capture/example.py

# Or with plain Python:
pip install agent-observability-trace-cli[mcp]
python examples/04-mcp-stdio-capture/example.py
```

## What the output looks like

```
Connecting to MCP server over stdio...
Connected to: agent-trace-example-server

Listing tools...
Available tools: ['add', 'broken_tool']

Calling add(2, 3)...
Result: {'result': 5}

Calling broken_tool() — expected to fail...
isError: True

--- Span tree ---
Trace: mcp_stdio_example  run_00ef7c5a83fa  (284.5 ms total)
├── mcp:initialize      OK     (278.1 ms)
├── mcp:list_tools      OK     (2.9 ms)
├── mcp:tool:add        OK     (2.2 ms)
└── mcp:tool:broken_tool ERROR (1.2 ms)

Trace saved to: /Users/you/.agent-trace/runs/run_00ef7c5a83fa
MCP frames captured: 9
```

## What gets saved to disk

After the run, two files are written to `~/.agent-trace/runs/run_<id>/`:

- `trace.json` — the span tree (`mcp:initialize`, `mcp:list_tools`,
  `mcp:tool:add`, `mcp:tool:broken_tool`), with the failed tool call marked
  `ERROR`
- `fixture.db` — every JSON-RPC frame that crossed the subprocess's
  stdin/stdout, in order: `initialize` request/response,
  `notifications/initialized`, `tools/list` request/response, two
  `tools/call` request/response pairs

Inspect the raw frames directly:

```python
from agent_trace._replay.fixture import Fixture

fixture = Fixture(Path("~/.agent-trace/runs/run_<id>/fixture.db").expanduser())
for frame in fixture.all_mcp_frames():
    print(frame["direction"], frame["frame_type"], frame["method"], frame["payload"])
```
