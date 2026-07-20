# Example 04 — AutoGen Agent Observability

Demonstrates `agent_trace.integrations.autogen` against the modern
autogen-agentchat / autogen-ext (v0.4+/v0.7.x) architecture. Runs with
**zero API key and zero network I/O**:

- The LLM turn uses autogen-ext's own `ReplayChatCompletionClient` test
  double instead of a real model client, so the tool-call loop still runs
  through real `AssistantAgent` code with no external calls.
- The code-execution step runs a real local Python subprocess through
  `LocalCommandLineCodeExecutor` (no network either).

## What it demonstrates

- `instrument_agent(agent, tracer=tracer, trace=trace)` — wraps an
  `AssistantAgent` so every turn gets an `agent:<name>` span tagged with
  the agent's name, plus `tool_call_request`/`tool_call_execution` span
  events and accumulated `llm.usage.*` token counts.
- `instrument_code_executor(executor, tracer=tracer, trace=trace)` — wraps
  a `LocalCommandLineCodeExecutor` so every `execute_code_blocks` call is
  recorded as a `code_execution` span carrying the executed code, working
  directory, combined stdout+stderr output, and exit code — independent of
  and in addition to any LLM HTTP capture, since code execution never
  touches HTTP.

## How to run

```bash
# From the repo root:
uv run --extra autogen python examples/04-autogen-agent-observability/example.py

# Or with plain Python:
pip install "agent-observability-trace-cli[autogen]"
python examples/04-autogen-agent-observability/example.py
```

## What the output looks like

```
Running an AssistantAgent tool-call turn (zero API cost via ReplayChatCompletionClient) plus a real local code execution...

Trace: autogen-example  autogen-example-run  ((19.6 ms total))
├── agent:support_agent  OK  (1.2 ms)
└── code_execution  OK  (17.9 ms)

Final agent response: I found 3 documents about the refund policy.
```

Inspect the full trace (including the `tool_call_request`/
`tool_call_execution` span events and `code_execution.output`/
`code_execution.exit_code` attributes) with:

```bash
agent-trace show autogen-example-run
```

## Wiring real LLM traffic through RecordingTransport

This example uses `ReplayChatCompletionClient` to stay free and offline.
For a real `OpenAIChatCompletionClient`/`AzureOpenAIChatCompletionClient`,
either:

1. Construct the client *after* entering `tracer.start_trace(record=True)`
   with no explicit `http_client=` kwarg — agent-trace's existing global
   `httpx.AsyncClient` patch captures it automatically, or
2. Pass `http_client=recording_http_client(fixture)` explicitly (see
   `agent_trace.integrations.autogen.recording_http_client`'s docstring)
   for explicit, documented control instead of relying on the global patch
   — this is also the path required for legacy AutoGen 0.2's `config_list`.
