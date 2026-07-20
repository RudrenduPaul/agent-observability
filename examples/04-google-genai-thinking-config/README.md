# Example 04 — Google GenAI thinking-config capture

This example shows what `agent_trace.integrations.google_genai` adds on top
of the generic `httpx` interceptor: Gemini's "thinking" fields
(`thinkingConfig` / `includeThoughts` / `thinkingBudget`) and the resulting
thoughts-token count, captured directly as span attributes instead of being
left as opaque JSON a developer has to hand-read out of `fixture.db`.

## Prerequisites

```bash
pip install agent-observability-trace-cli[google-genai]
```

No live Gemini API key is required — the example stubs the network call
boundary (`client.models.generate_content` / a `FakeListChatModel`) so it
runs offline with zero API cost, exactly like `01-basic-trace`.

## What it shows

1. **Raw SDK path** — `google.genai.Client` instrumented with
   `instrument_client()`. One `generate_content()` call with an explicit
   `ThinkingConfig(include_thoughts=True, thinking_budget=1024)` produces a
   span carrying:
   - `google_genai.include_thoughts`
   - `google_genai.thinking_budget`
   - `llm.usage.prompt_tokens` / `completion_tokens` / `total_tokens`
   - `google_genai.usage.thoughts_tokens`

2. **LangChain path** — `langchain_google_genai.ChatGoogleGenerativeAI`
   instrumented with `GoogleGenAITracer`, comparing:
   - a bare `llm.invoke(...)` → `google_genai.invocation_context =
     "direct_invocation"`
   - the same call routed through an LCEL chain
     (`prompt | llm | parser`) → `google_genai.invocation_context =
     "lcel_chain"`

   This distinction is what issue #31767 needed to debug "why does the
   chain-routed call behave differently from the direct-model call" — with
   the generic interceptor alone, both look like identical raw HTTP bodies.

## Run

```bash
uv run python examples/04-google-genai-thinking-config/example.py
```
