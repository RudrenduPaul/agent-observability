"""
Record & replay an AWS Bedrock call made through boto3 — no AWS account
required.

Run:
    uv run python examples/04-bedrock-record-replay/example.py record
    uv run python examples/04-bedrock-record-replay/example.py replay

This demonstrates the botocore/boto3 HTTP interceptor: agent-trace captures
AWS SDK traffic (Bedrock, SageMaker, or any other boto3-based service) the
same way it already captures httpx/requests traffic — just wrap the call in
`tracer.start_trace(record=True)`, no extra client wiring needed.

boto3 clients built with native AWS providers (e.g. crewAI's Bedrock
completion provider, or a LangChain `ChatBedrock` on the boto3 path) never
touch httpx.Client or requests.Session, so without this interceptor those
calls were previously invisible to agent-trace at the HTTP layer.

To keep this example self-contained (no AWS account, no credentials, no
network egress, zero API cost) it stands up a local HTTP server that mimics
the shape of the Bedrock Runtime `InvokeModel` API and points a real boto3
client at it via `endpoint_url`.  Everything below — request signing,
response parsing, `StreamingBody` handling — is the real botocore code path;
only the network endpoint is fake.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import boto3

from agent_trace import replay, tracer

RUN_ID = "example-04-bedrock"
FAKE_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
FAKE_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# Fixed (not OS-assigned) port so the exact URL recorded during record() —
# which is what next_exchange() matches on — is reproducible in replay(),
# even though the fake server isn't running then.  Fixture lookups match on
# the literal (method, url) pair, so record and replay must agree on it.
FAKE_ENDPOINT = "http://127.0.0.1:18743"


class _FakeBedrockHandler(BaseHTTPRequestHandler):
    """Stands in for the Bedrock Runtime InvokeModel endpoint."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        request_body = json.loads(self.rfile.read(length) or b"{}")
        prompt = request_body.get("prompt", "").strip()

        payload = json.dumps(
            {"completion": f"(fake model) You said: {prompt!r}"}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:  # silence stdlib logging
        pass


def invoke_claude(client: object, prompt: str) -> str:
    """A minimal Bedrock InvokeModel call — the same shape a real agent makes."""
    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-v2",
        body=json.dumps({"prompt": prompt, "max_tokens_to_sample": 50}),
        contentType="application/json",
        accept="application/json",
    )
    parsed = json.loads(response["body"].read())
    return str(parsed["completion"])


def record() -> None:
    server = HTTPServer(("127.0.0.1", 18743), _FakeBedrockHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    session = boto3.Session(
        region_name="us-east-1",
        aws_access_key_id=FAKE_ACCESS_KEY,
        aws_secret_access_key=FAKE_SECRET_KEY,
    )
    client = session.client(
        "bedrock-runtime",
        endpoint_url=FAKE_ENDPOINT,
        region_name="us-east-1",
    )

    print("Recording a Bedrock InvokeModel call via boto3...")
    with tracer.start_trace("bedrock-demo", record=True, run_id=RUN_ID):
        answer = invoke_claude(client, "What is agent-trace?")
        print(f"Model said: {answer}")

    server.shutdown()

    fixture_path = Path.home() / ".agent-trace" / "runs" / RUN_ID / "fixture.db"
    print(f"\nCaptured 1 AWS SDK exchange to: {fixture_path}")
    print("Replay it offline any time with:")
    print("    uv run python examples/04-bedrock-record-replay/example.py replay")


def replay_recorded() -> None:
    print("Replaying the Bedrock call from fixture.db — zero network calls...")

    # These credentials are never actually used, and the fake server from
    # record() need not even be running: the ReplaySession installed by
    # replay() intercepts every botocore request by (method, url) and serves
    # the recorded response directly, without touching the network.
    session = boto3.Session(
        region_name="us-east-1",
        aws_access_key_id=FAKE_ACCESS_KEY,
        aws_secret_access_key=FAKE_SECRET_KEY,
    )
    client = session.client(
        "bedrock-runtime",
        endpoint_url=FAKE_ENDPOINT,
        region_name="us-east-1",
    )

    with replay(RUN_ID):
        answer = invoke_claude(client, "What is agent-trace?")

    print(f"Model said (replayed, offline, zero cost): {answer}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    if mode == "record":
        record()
    elif mode == "replay":
        replay_recorded()
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (expected 'record' or 'replay')")
