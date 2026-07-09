# Example 12 — `create_react_agent(response_format=...)` + Tools Failure Mode

Closes the gap flagged against [issue #4940](https://github.com/langchain-ai/langgraph/issues/4940):
combining structured output (`response_format=SomeSchema`) with tool
calling is a recurring 400-producing failure class on Anthropic/Bedrock
models — if the model's final turn doesn't cleanly close out every pending
`tool_use` block before LangGraph asks it to emit the structured payload,
the provider rejects the request.

No API key required — `FakeAnthropicChatModel` makes one real HTTP call
(through a real, `RecordingTransport`-patched `httpx.Client`) to a mock
transport returning Anthropic's actual error body for this shape.

## What this shows

1. **The raw fixture capture** of the 400 — `Fixture.all_exchanges()`
   (`fixture.db`), the ground truth every diagnosis ultimately depends on.
2. **What the LangGraph integration span shows for the same failure** —
   since the HTTP-error-response-body-on-span fix shipped
   (`Span.record_exception`, `src/agent_trace/core/span.py`), this is no
   longer a generic `"400 Bad Request"` one-liner: Anthropic's actual
   rejection text (`"tool_use ids were found without tool_result blocks
   immediately after..."`) is attached directly to every ERROR span up the
   call stack, visible via `agent-trace show <run_id>` (or this example's
   own `_print_errors_only` call) without ever touching `fixture.db` by
   hand.

## Run

```bash
pip install agent-trace[langgraph]
python examples/12-langgraph-structured-response-tool-conflict/example.py
```
