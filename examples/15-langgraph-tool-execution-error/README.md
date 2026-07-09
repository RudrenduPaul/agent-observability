# Example 15 — A Tool's Own Logic Raising an Exception

Closes the gap flagged against [issue #30708](https://github.com/langchain-ai/langchain/issues/30708)
(`astream_events` never emits `on_tool_error`). No existing example shows a
*tool's own code* raising — `examples/02-langgraph-failure-replay` only
exercises HTTP-level failures.

No API key required — `FakeChatModel` always calls `divide(10, 0)`, and
`ToolNode(..., handle_tool_errors=False)` lets the resulting
`ZeroDivisionError` actually propagate instead of being swallowed into a
`ToolMessage` (LangGraph's default `handle_tool_errors=True` behavior) —
the exact "the tool raised, and nothing downstream told me" shape #30708's
reporter hit.

## What this shows

`LangGraphTracer.on_tool_error` (`src/agent_trace/integrations/
langgraph.py`) closes the tool's span `ERROR` with the exception captured
via `Span.record_exception` — independent of whatever event types the
installed `langchain-core`/`langgraph` version's own `astream_events`
implementation happens to support, since this is agent-trace's own
callback handler, not a consumer of `astream_events`.

## Run

```bash
pip install agent-trace[langgraph]
python examples/15-langgraph-tool-execution-error/example.py
```
