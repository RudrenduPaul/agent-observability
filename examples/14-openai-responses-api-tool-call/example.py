"""
OpenAI Responses API (`use_responses_api=True`) function_call/call_id
capture — and the orphaned-call_id diagnostic (#33895).

The Responses API uses a materially different message shape from Chat
Completions: instead of `messages[].tool_calls[].id` (assistant turn) and
`messages[].tool_call_id` (tool turn), it uses a flat `input` list of typed
items — `{"type": "function_call", "call_id": ...}` and
`{"type": "function_call_output", "call_id": ...}` — that must pair up by
`call_id`. Issue #33895 ("No call message found for call_*") is exactly
what happens when that pairing breaks: a `function_call` item with no
matching `function_call_output` sent back in the next turn.

`check_orphaned_tool_call_ids`/`check_missing_tool_call_id`
(`src/agent_trace/_inspect.py`) only understand the Chat Completions shape
— they never look at `input`/`function_call`/`function_call_output` at
all, so they miss this failure class entirely. This example shows the
dedicated Responses-API-aware check that closes that gap:
`check_orphaned_responses_api_call_ids`, wired into
`agent-trace inspect <run_id>` as the `orphaned_responses_api_call_ids`
check.

No API key required — this makes real HTTP calls (through a real,
`RecordingTransport`-patched `httpx.Client`) to a mock transport shaped
exactly like OpenAI's actual Responses API request/response bodies.

Run:
    python examples/14-openai-responses-api-tool-call/example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from agent_trace import Tracer
from agent_trace import _inspect as ins
from agent_trace._replay.fixture import Fixture

TRACE_DIR = Path.home() / ".agent-trace" / "runs"


def _mock_responses_api(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_123",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}]}
            ],
        },
        request=request,
    )


def _record_turn(client: httpx.Client, input_items: list[dict]) -> None:
    client.post(
        "https://api.openai.com/v1/responses",
        json={"model": "gpt-oss-20b", "input": input_items},
    )


def main() -> None:
    t = Tracer(trace_dir=TRACE_DIR)

    print("--- Turn A: a paired function_call / function_call_output (healthy) ---")
    with t.start_trace("responses-api-paired", record=True) as trace_ok:
        client = httpx.Client(transport=httpx.MockTransport(_mock_responses_api))
        _record_turn(
            client,
            [
                {"type": "message", "role": "user", "content": "what's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Boston"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ],
        )

    print("\n--- Turn B: an orphaned function_call, no matching output (#33895) ---")
    with t.start_trace("responses-api-orphaned", record=True) as trace_bad:
        client = httpx.Client(transport=httpx.MockTransport(_mock_responses_api))
        _record_turn(
            client,
            [
                {"type": "message", "role": "user", "content": "what's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Boston"}',
                },
                # No function_call_output for call_1 — this is #33895's
                # exact "No call message found for call_1" trigger shape.
            ],
        )

    for label, trace in (("Turn A (paired)", trace_ok), ("Turn B (orphaned)", trace_bad)):
        with Fixture(TRACE_DIR / trace.run_id / "fixture.db") as fixture:
            exchanges = fixture.all_exchanges()
        flags = ins.check_orphaned_responses_api_call_ids(exchanges)
        print(f"\n{label}: agent-trace inspect {trace.run_id}")
        if flags:
            for flag in flags:
                print(f"  FLAGGED: {flag['detail']}")
        else:
            print("  clean — no orphaned Responses API call_id(s)")

        # For contrast: the Chat-Completions-only checks see nothing here
        # at all (this body has no top-level `messages` field), confirming
        # they cannot substitute for the Responses-API-aware check above.
        chat_completions_flags = ins.check_orphaned_tool_call_ids(exchanges)
        print(
            f"  check_orphaned_tool_call_ids (Chat Completions shape only): "
            f"{len(chat_completions_flags)} flag(s) — blind to this shape"
        )


if __name__ == "__main__":
    main()
