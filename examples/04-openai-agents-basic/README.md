# Example 04 — OpenAI Agents SDK

This example is the worked example for
`agent_trace.integrations.openai_agents` — the first one in this repo.  It
shows the full span tree the integration produces for a realistic multi-agent
run: an agent that calls a tool, then hands off to a second agent.

## Prerequisites

```bash
pip install agent-trace[openai-agents]
export OPENAI_API_KEY=your-key   # needed only for the record/streamed steps
```

## The pipeline

Two `Agent`s:

```
researcher --[lookup_fact tool]--> handoff --> writer
```

- `researcher` — calls the `lookup_fact` function tool, then hands off to `writer`
- `writer` — writes the final one-sentence answer using the handed-off context

Instrumented via `instrument_runner()` (`AgentTraceHook` + `Runner.run`), this
produces:

```
agent_run
├── agent:researcher
│   ├── llm:<model>
│   ├── tool:lookup_fact
│   └── handoff:researcher->writer     (duration-based — see below)
└── agent:writer
    └── llm:<model>
```

## Step 1 — Record

```bash
python examples/04-openai-agents-basic/example.py record
```

Output includes the span tree, e.g.:

```
Recording run for: 'What year was the Rust programming language first released?'
(This makes real API calls. Ensure OPENAI_API_KEY is set.)

Final output: Rust 1.0 was released in 2015.

Run ID: run_a1b2c3d4e5f6

Trace: openai-agents-basic  [run_a1b2c3d4e5f6]
├── agent_run                          OK  (912.4 ms)
│   ├── agent:researcher               OK  (611.2 ms)
│   │   ├── llm:gpt-4o                 OK  (398.5 ms)
│   │   ├── tool:lookup_fact           OK  (0.1 ms)
│   │   └── handoff:researcher->writer OK  (0.3 ms)
│   └── agent:writer                   OK  (289.7 ms)
│       └── llm:gpt-4o                 OK  (271.9 ms)

Replay with:
  python examples/04-openai-agents-basic/example.py replay run_a1b2c3d4e5f6
```

## Step 2 — Replay

```bash
python examples/04-openai-agents-basic/example.py replay run_a1b2c3d4e5f6
```

No API calls are made — every HTTP exchange the SDK made during recording is
served from `fixture.db`. See [`docs/integrations/openai-agents.md`](../../docs/integrations/openai-agents.md#5-known-limitations-and-sdk-version-requirements)
for what replay can and cannot reproduce (it replays exact recorded bytes; it
cannot simulate what a *different* request — e.g. a changed
`model_settings.reasoning_effort` — would have returned).

## Step 3 — Streamed variant

```bash
python examples/04-openai-agents-basic/example.py streamed
```

Runs the same pipeline through `instrument_runner_streamed()`
(`Runner.run_streamed()` + `stream_events()`) instead of `instrument_runner()`
(`Runner.run()`). Prints each event type as it's actually delivered to the
consumer, and records a `SpanEvent` on the `agent_run_streamed` root span for
every one of them — this is the call shape `instrument_runner()` alone cannot
instrument, since a `RunHooks` callback firing does not imply the
corresponding event reached `stream_events()`'s consumer.

## What this example demonstrates about the integration

| Span attribute | Where it comes from |
|---|---|
| `agent.name`, `agent.model` | `on_agent_start` |
| `llm.model`, `llm.model_settings.reasoning_effort`, `llm.model_settings.verbosity` | `on_llm_start` |
| `llm.usage.*`, `llm.response.has_tool_calls` | `on_llm_end` |
| `tool.name`, `tool.result`, `tool.result_length` | `on_tool_start`/`on_tool_end` |
| `handoff.from_agent`, `handoff.to_agent` + `duration_ms` | `on_handoff` + `on_agent_end`/`on_agent_start` |

If the run raises (e.g. a provider error mid-turn), `instrument_runner`/
`instrument_runner_streamed` close every span still open in the hook's
registry as `ERROR` with the exception attached, instead of leaving them open
forever — see `AgentTraceHook._close_open_spans_with_exception` in
[`src/agent_trace/integrations/openai_agents.py`](../../src/agent_trace/integrations/openai_agents.py).

## Realtime API

The Realtime API (`agents.realtime`) has no `RunHooks`-style callback surface
at all, so it needs a different attach point: `AgentTraceRealtimeHook.wrap()`
wraps `RealtimeSession`'s own async-iterator protocol instead. See the
`AgentTraceRealtimeHook` class docstring in the same file — there's no
end-to-end example here yet since it requires a live voice/audio session.

## See also

- `docs/integrations/openai-agents.md` — full integration guide and known limitations
- `examples/02-langgraph-failure-replay/` — the same record/replay workflow for LangGraph
