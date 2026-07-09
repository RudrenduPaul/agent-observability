# Example 17 — Self-Hosted / Custom `base_url` LLM Server + Prebuilt `ToolNode`

Closes the gap flagged against [issue #3538](https://github.com/langchain-ai/langgraph/issues/3538):
`ChatOpenAI(base_url="http://host.docker.internal:9118", ...)` (a local
llama.cpp-style server) feeding a prebuilt `ToolNode`, where the actual
failure — the local server returning an **empty-string** `tool_calls[].id`,
not a missing one — was only visible at the wire level, with no example or
doc showing how to attach agent-trace's HTTP interceptor to this
configuration.

## No manual wiring required

`Tracer._patch_httpx()` patches `httpx.Client._transport_for_url` at the
class level, at request-dispatch time — so any `httpx.Client` instance
constructed by any SDK, including the one `openai.OpenAI(base_url=...)`/
`langchain_openai.ChatOpenAI(base_url=...)` build internally for a custom
`base_url`, is intercepted automatically the moment
`Tracer.start_trace(record=True)` is active. The older pattern of manually
passing `http_client=httpx.Client(transport=RecordingTransport(...))` into
`ChatOpenAI` is not necessary with the current interceptor.

No API key required — this example uses the plain `openai` SDK client
(what `ChatOpenAI` wraps internally) pointed at a local HTTP server that
mimics a self-hosted OpenAI-compatible endpoint returning #3538's exact
malformed shape.

## What this shows

- A self-hosted-style provider response is captured with zero explicit
  interceptor wiring — just `Tracer.start_trace(record=True)` around the
  call.
- `check_missing_tool_call_id` (`src/agent_trace/_inspect.py`) — already
  wired into `agent-trace inspect <run_id>` — flags the empty-string
  `tool_calls[].id`, the exact #3538 shape.

## Run

```bash
pip install openai
python examples/17-langgraph-toolnode-custom-provider/example.py
```
