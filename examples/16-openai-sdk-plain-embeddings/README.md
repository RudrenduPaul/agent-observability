# Example 16 — A Plain OpenAI SDK Call, No LangGraph Orchestration

Closes the gap flagged against [issue #31227](https://github.com/langchain-ai/langchain/issues/31227),
[#31192](https://github.com/langchain-ai/langchain/issues/31192), and
[#3994](https://github.com/pydantic/pydantic-ai/issues/3994): every other
example assumes a LangGraph graph being invoked, which leaves no reference
implementation for the (much larger) population of users hitting plain-SDK
or non-LangGraph-LangChain failures.

No API key required — a local HTTP server mimics OpenAI's real
`/v1/embeddings` endpoint shape; a real `openai.OpenAI(base_url=...)`
client is pointed at it, so the actual OpenAI SDK code path (request
building, retries, response parsing) is exercised.

## What this shows

- `@tracer.instrument(record=True)`/`Tracer.start_trace(record=True)` works
  identically around a bare `client.embeddings.create(...)` call — zero
  LangGraph/LangChain code involved. The interceptor layer
  (`src/agent_trace/interceptor/httpx_hook.py`) is completely
  framework-agnostic.
- A successful call and a `400` (the real #31227 shape: `OpenAIEmbeddings`
  silently batching input past the model's token limit) are both captured.
- The captured error body is inspected directly via
  `Fixture.all_exchanges()` — no framework span, no LangGraph integration,
  just the raw HTTP capture every other integration in this repo is built
  on top of.

## Run

```bash
pip install openai
python examples/16-openai-sdk-plain-embeddings/example.py
```
