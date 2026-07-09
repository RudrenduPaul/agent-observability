"""
A plain OpenAI SDK call, no LangGraph orchestration (#31227, #31192, #3994).

Every other example in this repo assumes a LangGraph graph being invoked.
This one shows that agent-trace's HTTP interceptor layer is completely
framework-agnostic: `@tracer.instrument(record=True)` (or
`Tracer.start_trace(record=True)`) works identically around a bare
`openai.OpenAI().embeddings.create(...)` call, with zero LangGraph/
LangChain code involved — the same capture mechanism `#31227`'s
`OpenAIEmbeddings` token-limit-400 bug, `#31192`'s DocumentCompressorPipeline
issue, and `#3994`'s pydantic-ai/OpenRouter response-shape issue all hit,
independent of which (if any) agent framework sits on top.

To keep this self-contained (no API key, no network egress, zero cost) it
stands up a local HTTP server mimicking OpenAI's real `/v1/embeddings`
endpoint shape and points a real `openai.OpenAI(base_url=...)` client at
it — the actual OpenAI SDK code path (request building, retries, response
parsing) is exercised; only the network endpoint is fake.

Run:
    python examples/16-openai-sdk-plain-embeddings/example.py

No API key required.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

try:
    import openai
except ImportError:
    sys.exit("openai is not installed.\nRun: pip install openai")

from agent_trace import Tracer
from agent_trace._replay.fixture import Fixture

TRACE_DIR = Path.home() / ".agent-trace" / "runs"
FAKE_ENDPOINT = "http://127.0.0.1:18744/v1"

# The exact shape of #31227's real failure: OpenAIEmbeddings silently
# batches input beyond the model's token limit, and the API rejects it
# with a 400 whose body names the actual limit.
TOKEN_LIMIT_ERROR_BODY = {
    "error": {
        "message": "This model's maximum context length is 8192 tokens, "
        "however you requested 12000 tokens in the input for embedding "
        "generation. Please reduce your input.",
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
    }
}


class _FakeEmbeddingsHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        inputs = body.get("input")
        if isinstance(inputs, list):
            total_len = sum(len(str(i)) for i in inputs)
        else:
            total_len = len(str(inputs))

        if total_len > 500:  # stand-in for "exceeds the model's token limit"
            payload = json.dumps(TOKEN_LIMIT_ERROR_BODY).encode()
            self.send_response(400)
        else:
            payload = json.dumps(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.01] * 8}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                }
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence the default request-logging to stderr


def _start_fake_server() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 18744), _FakeEmbeddingsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    server = _start_fake_server()
    try:
        t = Tracer(trace_dir=TRACE_DIR)
        client = openai.OpenAI(base_url=FAKE_ENDPOINT, api_key="fake-key-not-used")

        print("--- Call 1: normal-sized input (succeeds) ---")
        with t.start_trace("plain-openai-embeddings", record=True) as trace:
            result = client.embeddings.create(
                model="text-embedding-3-small", input="a short sentence"
            )
            print(f"  embedding dims: {len(result.data[0].embedding)}")

        print("\n--- Call 2: oversized input (real #31227 shape: 400) ---")
        with t.start_trace("plain-openai-embeddings-error", record=True) as trace_err:
            try:
                client.embeddings.create(
                    model="text-embedding-3-small", input=["x" * 100] * 10
                )
            except openai.APIStatusError as exc:
                print(f"  caught (expected): {type(exc).__name__}: {exc}")

        print("\n--- Inspecting the captured error via Fixture.all_exchanges() ---")
        with Fixture(TRACE_DIR / trace_err.run_id / "fixture.db") as fixture:
            for exchange in fixture.all_exchanges():
                if exchange["response_status"] and exchange["response_status"] >= 400:
                    print(f"  {exchange['method']} {exchange['url']} -> "
                          f"{exchange['response_status']}")
                    print(f"  body: {exchange['response_body']}")

        print(f"\nSuccessful run ID: {trace.run_id}")
        print(f"Error run ID:      {trace_err.run_id}")
        print(
            "\nNo LangGraph, no LangChain callback — @tracer.instrument/"
            "Tracer.start_trace(record=True) recorded both calls purely via "
            "the framework-agnostic httpx interceptor "
            "(src/agent_trace/interceptor/httpx_hook.py). This is the same "
            "capture path any non-LangGraph SDK usage — pydantic-ai's "
            "OpenRouter provider, a plain OpenAIEmbeddings call, a "
            "DocumentCompressorPipeline — goes through today."
        )
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
