"""
Realtime-API WebSocket interceptor example.
Run: uv run python examples/04-realtime-websocket-interceptor/example.py

Demonstrates recording and offline-replaying a persistent duplex WebSocket
session — the shape of the OpenAI Agents SDK's Realtime API (a long-lived
connection carrying many JSON events: tool calls, handoffs, audio deltas)
rather than the discrete request/response model the httpx interceptor
assumes.

No OpenAI API key required: a local `websockets.serve` echo server stands in
for the Realtime endpoint, so this is runnable offline. The interception
mechanism is identical to what happens against the real API — the SDK's
`OpenAIRealtimeWebSocketModel._create_websocket_connection` calls
`websockets.connect(url, **kwargs)` exactly like this example does, so the
same `Tracer.start_trace(record=True)` patch captures it with zero code
changes on the SDK side.

Requires the `websockets` package:
    pip install "agent-observability-trace-cli[realtime]"
"""

from __future__ import annotations

import asyncio

import websockets

from agent_trace import Tracer
from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.websocket_hook import ReplayWebSocketConnection


async def fake_realtime_server(server_conn: object) -> None:
    """Stand-in for OpenAI's Realtime API: echoes a couple of session events."""
    await server_conn.send('{"type": "session.created"}')  # type: ignore[attr-defined]
    async for message in server_conn:  # type: ignore[attr-defined]
        await server_conn.send(  # type: ignore[attr-defined]
            f'{{"type": "response.output_text.delta", "in_reply_to": {message}}}'
        )


async def record_a_realtime_session(tracer: Tracer) -> str:
    """Open a duplex WS "Realtime" session and capture every frame to a fixture."""
    async with websockets.serve(fake_realtime_server, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://localhost:{port}"

        with tracer.start_trace("realtime_session", record=True) as trace:
            # This call is completely unmodified — `Tracer.start_trace(record=True)`
            # has already monkey-patched `websockets.connect` for the duration of
            # this `with` block, so it returns a RecordingWebSocketConnection that
            # tees every frame into the trace's fixture.db as it passes through.
            ws = await websockets.connect(url)

            greeting = str(await ws.recv())
            print(f"  <- {greeting}")

            await ws.send('"tell me a joke"')
            print('  -> "tell me a joke"')

            reply = str(await ws.recv())
            print(f"  <- {reply}")

            await ws.close()

        run_id = trace.run_id

    return run_id


async def replay_the_session_offline(tracer: Tracer, run_id: str) -> None:
    """Replay the recorded session with zero network I/O."""
    fixture_path = tracer._trace_dir / run_id / "fixture.db"
    fixture = Fixture(fixture_path)

    # The recording used one connection; grab its id from the captured frames
    # so the replay connection knows which frames belong to it.
    frames = fixture.all_ws_frames()
    connection_id = frames[0]["connection_id"]
    print(f"  Recorded {len(frames)} frames for connection {connection_id!r}:")
    for f in frames:
        print(f"    [{f['direction']:>4}] {f['payload']}")

    replay = ReplayWebSocketConnection(
        fixture, url="ws://replayed", connection_id=connection_id
    )

    print("\n  Replaying inbound frames (no network call made)...")
    async for message in replay:
        print(f"  <- {message}")

    fixture.close()


async def main() -> None:
    tracer = Tracer()

    print("Recording a live Realtime-style WebSocket session...")
    run_id = await record_a_realtime_session(tracer)
    print(f"\nSaved to: {tracer._trace_dir / run_id}")

    print("\n--- Offline replay ---")
    await replay_the_session_offline(tracer, run_id)


if __name__ == "__main__":
    asyncio.run(main())
