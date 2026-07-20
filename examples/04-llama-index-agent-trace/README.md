# Example 04 — llama_index Agent Trace

Demonstrates the llama_index framework integration (`LlamaIndexTracer`). No
API key or network access required — this uses llama_index's own `MockLLM`
and a plain Python `FunctionTool`, so the example is fully reproducible
offline.

## What it demonstrates

- `LlamaIndexTracer(tracer=tracer, trace=trace)` used as a context manager —
  installs onto llama_index's global `Dispatcher` on `__enter__`, uninstalls
  on `__exit__`
- Every llama_index-instrumented call (`MockLLM.chat`, `MockLLM.complete`,
  `FunctionTool.call`) becomes an agent-trace span, with parent/child
  nesting taken directly from llama_index's own dispatcher span tree (e.g.
  `MockLLM.complete` nests under the `MockLLM.chat` that called it)
- Span enrichment from llama_index's semantic instrumentation events:
  `LLMChatStartEvent`/`LLMChatEndEvent` populate `llm.messages_count`,
  `llm.model`, `llm.last_message_role`, `llm.last_message_content`,
  `llm.response_content`, `llm.has_tool_calls`
- Attribution of each span back to the llama_index class that produced it
  (`llama_index.class`, `llama_index.span_id` attributes)

## How to run

```bash
# From the repo root:
uv run python examples/04-llama-index-agent-trace/example.py

# Or with plain Python:
pip install agent-observability-trace-cli[llama-index]
python examples/04-llama-index-agent-trace/example.py
```

## What the output looks like

```
Question: What's the weather in San Francisco?

--- Span tree ---
Trace: llama_index_agent_flow  run_31702ee5cf79  ((0.4 ms total))
├── MockLLM.chat  OK  (0.2 ms)
│   └── MockLLM.complete  OK  (0.1 ms)
├── FunctionTool.call  OK  (0.0 ms)
└── MockLLM.chat  OK  (0.1 ms)
    └── MockLLM.complete  OK  (0.0 ms)

--- Selected span attributes ---
MockLLM.chat: {'llama_index.class': 'MockLLM', 'llm.messages_count': 1, ...}
...

Trace saved to: ~/.agent-trace/runs/run_31702ee5cf79

--- Final answer ---
user: What's the weather in San Francisco?
tool: It is sunny and 72F in San Francisco.
assistant:
```

## Why the context-manager form

Unlike the LangGraph/OpenAI Agents integrations, llama_index does not accept
a per-call `callbacks=[...]` list — its instrumentation is a global
`Dispatcher` tree. `LlamaIndexTracer` is installed onto (by default) the root
dispatcher for the duration of a `with` block, so every llama_index call
inside that block is captured, and nothing outside it is. For long-lived
processes (e.g. a server), call `.install()`/`.uninstall()` manually instead
of using the context-manager form.
