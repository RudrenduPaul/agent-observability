# Example 04 — gRPC Record and Replay

Demonstrates recording and replaying a **gRPC** call, not just HTTP. This is
the piece that closes the gap for LLM SDKs that default to a gRPC transport
instead of REST — e.g. Vertex AI's mTLS-authenticated path, or
`google-generativeai` on pre-4.0 SDK versions.

No credentials or external network calls are needed to run this: it spins
up a tiny local gRPC "echo" server (`echo.proto`) that stands in for a real
LLM service, so the whole example runs on `127.0.0.1`.

## What it demonstrates

- `tracer.start_trace("name", record=True)` transparently intercepts
  `grpc.insecure_channel(...)` / `grpc.secure_channel(...)` — the "SDK" code
  in `call_echo_service()` is plain grpc usage with zero agent-trace-specific
  code, exactly how `google-api-core`'s `grpc_helpers.create_channel()`
  builds a channel for Vertex AI / `google-generativeai` under the hood.
- The recorded gRPC exchange lands in the same `fixture.db` SQLite file that
  httpx/requests exchanges use — one fixture, one file, regardless of
  transport.
- `replay(run_id)` serves the exact same gRPC call from the fixture with the
  local server **stopped**, proving no live gRPC I/O occurs during replay.

## How to run

```bash
# From the repo root:
uv run python examples/04-grpc-record-replay/example.py

# Or with plain Python:
pip install "agent-observability-trace-cli[grpc]"
python examples/04-grpc-record-replay/example.py
```

## What the output looks like

```
--- Recording ---
Live call result: 'echo:hello from the real network'
Recorded exchange saved to: /tmp/agent-trace-grpc-example-.../run_.../fixture.db

--- Replaying (local server is now stopped) ---
Replayed call result: 'echo:hello from the real network'

Replay matched the original recording -- no live gRPC call was made.
```

## Files

- `echo.proto` — a tiny two-RPC service (`UnaryEcho`, `StreamingEcho`) used
  only to stand in for a real LLM API in this example; not shipped as part
  of agent-trace itself.
- `echo_pb2.py` / `echo_pb2_grpc.py` — pre-compiled from `echo.proto` via
  `grpcio-tools` (`python -m grpc_tools.protoc`), checked in so the example
  runs with only `grpcio` installed, no `protoc` step required.
- `example.py` — starts the local server, records one call via
  `tracer.start_trace(record=True)`, stops the server, then replays the same
  call via `replay()`.

## Coverage note

This example (and the interceptor itself) covers **unary-unary** and
**unary-stream** gRPC calls — the shapes Gemini/Vertex AI's `GenerateContent`
and `StreamGenerateContent` actually use. Client-streaming and
bidirectional-streaming RPCs are out of scope for this pass; see
`src/agent_trace/interceptor/grpc_hook.py`'s module docstring for the full
rationale.
