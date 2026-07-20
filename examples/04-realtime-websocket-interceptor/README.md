# Example 04 — Realtime-API WebSocket Interceptor

Demonstrates recording and offline-replaying a persistent duplex WebSocket
session — the shape of the OpenAI Agents SDK's Realtime API (a long-lived
connection carrying many JSON events: tool calls, handoffs, audio deltas)
rather than the discrete request/response model the httpx interceptor
(`agent_trace.interceptor.httpx_hook`) assumes.

## What it demonstrates

- `Tracer.start_trace("name", record=True)` transparently patches
  `websockets.connect` for the duration of the `with` block, the same way it
  already patches `httpx.Client`/`httpx.AsyncClient` — no code changes
  needed on the caller's side.
- `RecordingWebSocketConnection` tees every inbound and outbound frame into
  the trace's `fixture.db` as it passes through, without altering behavior.
- `ReplayWebSocketConnection` serves the recorded inbound frames back in
  order with zero network I/O, so a captured Realtime session can be
  replayed offline at zero API cost.
- `Fixture.record_ws_frame()` / `next_ws_frame()` / `all_ws_frames()` — the
  WS-frame storage API added alongside the existing `http_exchanges` table.

## Why this matters

The OpenAI Agents SDK's `OpenAIRealtimeWebSocketModel._create_websocket_connection`
calls `websockets.connect(url, **kwargs)` internally
(`agents/realtime/openai_realtime.py`) — exactly the same call this example
makes by hand. Because the patch operates on the `websockets.connect`
module attribute, it captures the SDK's own Realtime session with **zero**
integration code, the same way agent-trace's httpx patch already captures
any SDK built on `httpx.Client`.

## How to run

```bash
# From the repo root:
uv run python examples/04-realtime-websocket-interceptor/example.py

# Or with plain Python:
pip install "agent-observability-trace-cli[realtime]"
python examples/04-realtime-websocket-interceptor/example.py
```

No OpenAI API key required — a local `websockets.serve` echo server stands
in for the Realtime endpoint, so the example is fully offline-runnable.

## What the output looks like

```
Recording a live Realtime-style WebSocket session...
  <- {"type": "session.created"}
  -> "tell me a joke"
  <- {"type": "response.output_text.delta", "in_reply_to": "tell me a joke"}

Saved to: /Users/you/.agent-trace/runs/run_<id>

--- Offline replay ---
  Recorded 3 frames for connection '<connection_id>':
    [recv] {"type": "session.created"}
    [send] "tell me a joke"
    [recv] {"type": "response.output_text.delta", "in_reply_to": "tell me a joke"}

  Replaying inbound frames (no network call made)...
  <- {"type": "session.created"}
  <- {"type": "response.output_text.delta", "in_reply_to": "tell me a joke"}
```

## What gets saved to disk

After the recording run, `~/.agent-trace/runs/run_<id>/` contains:

- `trace.json` — the span tree for the `realtime_session` trace
- `fixture.db` — a SQLite database with a `ws_frames` table holding every
  send/recv frame from the session, in original order, keyed by
  `connection_id`

## Using this against the real Realtime API

Swap the local `websockets.serve` echo server for a real connection and
this example becomes a real capture of an `agents.realtime` session:

```python
from agents.realtime.openai_realtime import OpenAIRealtimeWebSocketModel
from agent_trace import tracer

with tracer.start_trace("realtime_agent_run", record=True):
    model = OpenAIRealtimeWebSocketModel()
    await model.connect(options)  # websockets.connect is already patched
    # ... use the SDK normally; every frame lands in fixture.db
```
