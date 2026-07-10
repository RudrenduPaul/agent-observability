# Example 14 — OpenAI Responses API `function_call`/`call_id` Capture

Closes the gap flagged against [issue #33895](https://github.com/langchain-ai/langchain/issues/33895)
("No call message found for call_*" with `gpt-oss-20b`).

The Responses API uses a materially different message shape from Chat
Completions: a flat `input` list of typed items —
`{"type": "function_call", "call_id": ...}` and
`{"type": "function_call_output", "call_id": ...}` — instead of Chat
Completions' `messages[].tool_calls[].id` / `messages[].tool_call_id`.
#33895's exact failure is a `function_call` item with no matching
`function_call_output` sent back in the next turn.

No API key required — this makes real HTTP calls (through a real,
`RecordingTransport`-patched `httpx.Client`) to a mock transport shaped
exactly like OpenAI's actual Responses API bodies.

## What this shows

- `check_orphaned_tool_call_ids`/`check_missing_tool_call_id`
  (`src/agent_trace/_inspect.py`) only understand the Chat Completions
  shape — they never look at `input`/`function_call`/
  `function_call_output`, so they're blind to this failure class entirely
  (demonstrated directly in the output).
- `check_orphaned_responses_api_call_ids` — the Responses-API-aware
  equivalent added in this pass — correctly flags the orphaned `call_id`,
  wired into `agent-trace inspect <run_id>` as the
  `orphaned_responses_api_call_ids` check.

## Run

```bash
python examples/14-openai-responses-api-tool-call/example.py
```

Or record a run and inspect it directly:

```bash
agent-trace inspect <run_id>
```
