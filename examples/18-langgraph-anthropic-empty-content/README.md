# Example 18 — Anthropic Empty-Content Message Replay 400

Closes the last open gap on [issue #3168](https://github.com/langchain-ai/langgraph/issues/3168):
Anthropic's Messages API rejects any request where an `assistant` message
has empty `content` (`""` or `[]`) unless it's the *last* message in the
array — `"messages.N: all messages must have non-empty content"`. LangGraph
tool-calling turns routinely produce an `AIMessage` with empty text content
(only `tool_calls`), and once a later turn's request is built that message
is no longer final — it's now buried mid-array, which is exactly the shape
Anthropic's API rejects.

No API key required — `FakeAnthropicChatModel` makes one real HTTP call
(through a real, `RecordingTransport`-patched `httpx.Client`) to a mock
transport returning Anthropic's actual error body for this shape.

## What this shows

1. **The raw fixture capture** of the 400 — `Fixture.all_exchanges()`
   (`fixture.db`), the ground truth every diagnosis ultimately depends on.
2. **What the LangGraph integration span shows for the same failure** —
   Anthropic's actual rejection text attached directly to the ERROR span via
   `Span.record_exception`, visible via `agent-trace show <run_id>` without
   ever touching `fixture.db` by hand.
3. **`check_empty_content_not_final` firing on the captured fixture** — the
   automated check (`src/agent_trace/_inspect.py`, wired into
   `agent-trace inspect` via `run_all_exchange_checks`) that flags a
   non-final assistant message with empty content *before* it ever reaches
   Anthropic, instead of requiring a developer to notice the 400 and
   hand-diff the request body.

## Run

```bash
pip install agent-observability-trace-cli[langgraph]
python examples/18-langgraph-anthropic-empty-content/example.py
```
