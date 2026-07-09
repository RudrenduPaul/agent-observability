"""
Example 04 — Record and replay a gRPC call.

Demonstrates agent-trace's gRPC interceptor: the same record/replay story
that examples 01-03 show for HTTP, but for LLM SDKs that build their
transport with `grpc.secure_channel()` / `grpc.insecure_channel()` instead
of `httpx`/`requests` -- e.g. Vertex AI's mTLS-authenticated path, or
`google-generativeai` pre-4.0 (see the backlog item this closes:
"Add gRPC interceptor for LLM SDKs that default to grpc transport").

No credentials or external network calls needed: this spins up a tiny local
gRPC "echo" server on 127.0.0.1 (defined in echo.proto) to stand in for a
real LLM service, so the whole example runs offline.

What it demonstrates
---------------------
- `tracer.start_trace("name", record=True)` transparently intercepts
  `grpc.insecure_channel(...)` -- no code changes needed in the "SDK" (the
  echo client below is written exactly as if agent-trace didn't exist).
- The recorded exchange lands in the same `fixture.db` SQLite file that
  httpx/requests exchanges use.
- `replay(run_id)` serves the exact same gRPC call from the fixture with the
  local server *stopped* -- proving no real gRPC I/O occurs during replay.
"""

from __future__ import annotations

import sys
import tempfile
from concurrent import futures
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).parent))
import echo_pb2
import echo_pb2_grpc

from agent_trace import Tracer, replay


class _EchoServicer(echo_pb2_grpc.EchoServicer):
    def UnaryEcho(self, request, context):  # noqa: N802
        return echo_pb2.EchoResponse(message=f"echo:{request.message}")


def _start_local_server() -> tuple[grpc.Server, str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    echo_pb2_grpc.add_EchoServicer_to_server(_EchoServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def call_echo_service(target: str, message: str) -> str:
    """Stand-in for an LLM SDK call -- plain grpc.insecure_channel usage,
    with zero agent-trace-specific code. This is exactly how google-api-core
    (and therefore Vertex AI / google-generativeai) builds its channel.
    """
    channel = grpc.insecure_channel(target)
    stub = echo_pb2_grpc.EchoStub(channel)
    response = stub.UnaryEcho(echo_pb2.EchoRequest(message=message))
    return response.message


def main() -> None:
    trace_dir = Path(tempfile.mkdtemp(prefix="agent-trace-grpc-example-"))
    tracer = Tracer(trace_dir=trace_dir)
    server, target = _start_local_server()

    print("--- Recording ---")
    with tracer.start_trace("grpc-example", record=True) as trace:
        run_id = trace.run_id
        result = call_echo_service(target, "hello from the real network")
        print(f"Live call result: {result!r}")

    fixture_db = trace_dir / run_id / "fixture.db"
    print(f"Recorded exchange saved to: {fixture_db}")

    # Stop the local server entirely -- replay must not need it.
    server.stop(grace=None)
    print("\n--- Replaying (local server is now stopped) ---")
    with replay(run_id, trace_dir=trace_dir):
        replayed = call_echo_service(target, "hello from the real network")
        print(f"Replayed call result: {replayed!r}")

    assert result == replayed, "replayed response should match the recording"
    print("\nReplay matched the original recording -- no live gRPC call was made.")


if __name__ == "__main__":
    main()
