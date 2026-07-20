# Example 13 — Streaming Token Usage: the LangGraphTracer Callback Gap

Closes the gap flagged against [issue #3911](https://github.com/langchain-ai/langgraph/issues/3911):
LangChain's own `get_openai_callback()` reports zero tokens under
`stream_mode="messages"` because it only reads
`response.llm_output["token_usage"]`, which many streaming
configurations never populate — usage instead lands on the final streamed
chunk's own `AIMessageChunk.usage_metadata`.

No API key required — `FakeStreamingChatModel` reproduces the exact shape:
`llm_output` stays empty for the whole run, and usage only appears on the
last chunk's `usage_metadata`.

## What this shows

1. **What a naive `llm_output`-only reader sees** — nothing. This
   reproduces #3911's exact symptom (zero/missing usage under streaming).
2. **What `LangGraphTracer.on_llm_end` captures today** — correct token
   counts, via the `response.generations[0][0].message.usage_metadata`
   fallback in `_extract_token_usage()`
   (`src/agent_trace/integrations/langgraph.py`). A `create_react_agent`
   graph built on a real streaming `ChatOpenAI`/`AzureChatOpenAI` model goes
   through the identical `on_chat_model_start`/`on_llm_end` callback path,
   so the same fallback applies inside a full graph run too.

## Run

```bash
pip install agent-observability-trace-cli[langgraph]
python examples/13-langgraph-streaming-token-usage/example.py
```
