# Example 04 — pydantic-ai Retry-Attempt Diagnosis

This example shows the `agent_trace.integrations.pydantic_ai` framework
integration: every model request and tool call inside a pydantic-ai `Agent`
run is captured as its own span, tagged with *why* it happened — a fresh
turn, or a retry forced by `ModelRetry` (raised from a tool implementation
or an `@agent.output_validator`).

The generic httpx interceptor already captures the raw HTTP bytes for every
pydantic-ai call with zero framework-specific code (see the "generic client"
example) — but it has no way to tell you *which* retry attempt a given
exchange belongs to, or whether an exchange was a judge/validator retry at
all. This integration adds that attribution at the framework level, closing
exactly the gap identified against pydantic-ai issues
[#5508](https://github.com/pydantic/pydantic-ai/issues/5508) (~1/3 of calls
failing, with nothing tagging which retry attempt succeeded) and
[#4919](https://github.com/pydantic/pydantic-ai/issues/4919) (a 48KB retry
payload stored as one opaque blob).

## Prerequisites

```bash
pip install "agent-trace[pydantic-ai]"
```

No API key is needed — the example runs against pydantic-ai's built-in
`TestModel` (a fully offline, deterministic stand-in) by default.

## The agent

An `Agent` with:

- `flaky_lookup` — a tool that raises `ModelRetry` on its first call and
  succeeds on the second, simulating a transient tool failure.
- `must_pass_on_second_look` — an `@agent.output_validator` that rejects the
  model's first structured-output attempt and accepts the second, simulating
  a validation rule that occasionally rejects a model's first answer.

## Run it

```bash
python examples/04-pydantic-ai-retry-diagnosis/example.py
```

Output:

```
Result: 0

Run ID: run_17576a358529

Span tree (note llm.is_retry / llm.retry_tool_name / tool.retried):

Trace: pydantic-ai-retry-diagnosis  run_17576a358529  ((9.2 ms total))
└── agent:retry-demo-agent  OK  (9.2 ms)
    ├── llm:test  OK  (0.8 ms)
    ├── tool:flaky_lookup  OK  (0.5 ms)
    ├── llm:test  OK  (0.7 ms)
    ├── tool:flaky_lookup  OK  (0.4 ms)
    ├── llm:test  OK  (0.7 ms)
    ├── tool:final_result  OK  (0.4 ms)
    ├── llm:test  OK  (0.6 ms)
    └── tool:final_result  OK  (0.4 ms)

2 model-call span(s) tagged as retries.
2 tool span(s) tagged as retried (ModelRetry).
```

Each `llm:test` span carries attributes not visible in the tree view above —
inspect `trace.json` (or `agent-trace show <run_id>`) to see them:

- `llm.is_retry` / `llm.retry_index` — this model call is retry attempt N
- `llm.retry_tool_name` — set when the retry was forced by a tool's
  `ModelRetry` (as opposed to the output validator)
- `llm.retry_reason` — `"output_validator"` when no tool name is present
- `llm.usage.prompt_tokens` / `llm.usage.completion_tokens`
- `tool.retried` — the tool raised `ModelRetry` (closed `OK`, not `ERROR` —
  it's pydantic-ai's own soft-retry control-flow signal, not an application
  failure)

## Against a real provider

```bash
export OPENAI_API_KEY=your-key
python examples/04-pydantic-ai-retry-diagnosis/example.py --model openai:gpt-4o-mini
```

`Agent("openai:gpt-4o-mini", ...)` uses the `openai` SDK's `AsyncOpenAI`
client under the hood, which agent-trace's existing httpx interceptor
already records/replays exactly as it does in the other examples — the
`record=True` passed to `start_trace()` here captures those HTTP exchanges
into `fixture.db` alongside the pydantic-ai-level span attribution above.

## See also

- `src/agent_trace/integrations/pydantic_ai.py` — the integration itself
- `docs/integrations/langgraph.md` — the equivalent guide for LangGraph (a
  pydantic-ai guide should be added alongside it)
- `tests/integration/test_pydantic_ai.py` — real-package tests covering
  retry attribution, tool-call spans, and error propagation
