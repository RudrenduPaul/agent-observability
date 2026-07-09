"""
Recording a self-hosted / custom-`base_url` OpenAI-compatible LLM server
feeding a prebuilt `ToolNode` (#3538).

Issue #3538 used exactly this shape — `ChatOpenAI(base_url="http://host.
docker.internal:9118", ...)` (a local llama.cpp-style server) feeding a
prebuilt `ToolNode` — and the actual failure (the local server returning an
**empty-string** `tool_calls[].id`, not a missing one) was only visible at
the wire level.

No existing example or doc shows wiring agent-trace's HTTP interceptor into
this configuration. The good news: **no explicit wiring is required at
all**. `Tracer._patch_httpx()` (`src/agent_trace/__init__.py`) patches
`httpx.Client._transport_for_url` at the *class* level, at request-dispatch
time — so any `httpx.Client` instance constructed by any SDK (including the
one `openai.OpenAI(base_url=...)`/`langchain_openai.ChatOpenAI(base_url=
...)` build internally for a custom `base_url`) is automatically
intercepted the moment `Tracer.start_trace(record=True)` is active. The
older pattern of manually passing `http_client=httpx.Client(transport=
RecordingTransport(...))` into `ChatOpenAI` is not necessary with the
current interceptor.

This example uses the plain `openai` SDK client (the same `openai.OpenAI`
client `langchain_openai.ChatOpenAI` wraps internally) pointed at a local
HTTP server that mimics a self-hosted OpenAI-compatible endpoint returning
the exact malformed shape #3538 hit: a `tool_calls[]` entry with
`"id": ""`.

Run:
    python examples/17-langgraph-toolnode-custom-provider/example.py

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
from agent_trace import _inspect as ins
from agent_trace._replay.fixture import Fixture

TRACE_DIR = Path.home() / ".agent-trace" / "runs"
FAKE_ENDPOINT = "http://127.0.0.1:18745/v1"


class _SelfHostedServerHandler(BaseHTTPRequestHandler):
    """Stands in for a local llama.cpp-style OpenAI-compatible server that
    emits a tool_calls[] entry with an empty-string id — #3538's exact
    malformed shape."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body, unused here

        payload = json.dumps(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": "local-llama-3-8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "",  # <-- #3538's bug: empty, not missing
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Boston"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _start_fake_server() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 18745), _SelfHostedServerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    server = _start_fake_server()
    try:
        t = Tracer(trace_dir=TRACE_DIR)
        # No http_client=RecordingTransport(...) wiring needed — plain
        # openai.OpenAI(base_url=...), exactly like a real ChatOpenAI(
        # base_url="http://host.docker.internal:9118") would construct
        # internally, is intercepted automatically.
        client = openai.OpenAI(base_url=FAKE_ENDPOINT, api_key="fake-key-not-used")

        with t.start_trace("toolnode-custom-provider", record=True) as trace:
            client.chat.completions.create(
                model="local-llama-3-8b",
                messages=[{"role": "user", "content": "what's the weather in Boston?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            )

        print(f"Recorded run: {trace.run_id}")
        print("\n--- agent-trace inspect (missing_tool_call_id check) ---")
        with Fixture(TRACE_DIR / trace.run_id / "fixture.db") as fixture:
            exchanges = fixture.all_exchanges()
        flags = ins.check_missing_tool_call_id(exchanges)
        for flag in flags:
            print(f"  FLAGGED: {flag['detail']}")
        if not flags:
            print("  (unexpected: no flag raised)")

    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
