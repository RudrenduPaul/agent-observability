# Example 10 — LangGraph Checkpointer + Interrupt + Resume

Every other example that touches a checkpointer either doesn't pause the
graph (`02-langgraph-failure-replay` is linear, no checkpointer at all) or
exercises checkpointer *writes* without an interrupt/resume step
(`05-parallel-command-parent-routing`). This example fills that gap — it's
the reference implementation for the checkpointer/interrupt/resume failure
class behind [issue #531](https://github.com/langchain-ai/langgraph/issues/531)
and [issue #4217](https://github.com/langchain-ai/langgraph/issues/4217).

No API key required — every "LLM decision" is a plain Python stand-in.

## Part A — interrupt() + resume (happy path)

Compiles a graph with an `InMemorySaver` checkpointer (wrapped in
`TracingCheckpointSaver` so every `put`/`put_writes`/serde call gets a
span), pauses it mid-run via `interrupt()`, then resumes with
`Command(resume=True)`. Run it to see:

- `checkpoint:put`/`checkpoint:put_writes` spans around every checkpoint
  write
- `checkpoint:serde:dumps_typed`/`loads_typed` spans around every
  (de)serialization at the checkpoint boundary
- `node:propose` closing `OK` (not `ERROR`) with `langgraph.interrupted=true`
  on the paused run

## Part B — the #4217 bug, reproduced live

Builds a two-tool `create_react_agent`-shaped graph (`agent` node emits
`tool_calls`, `tools` is a real `ToolNode`, `interrupt_before=["tools"]`) —
the same structure #4217's reporter used — then calls
`graph.update_state(config, {"messages": [tool_message]})` two ways:

- **without** `as_node` — reproduces the exact bug: the pregel scheduler's
  `next` task list comes back empty and the resumed run silently does
  nothing further (the "nothing happened, no error" experience #4217's
  reporter hit)
- **with** `as_node="tools"` — the maintainer's documented fix: the `agent`
  node is scheduled correctly and the run continues

Both calls go through `agent_trace.integrations.langgraph_checkpoint.
traced_update_state()`, which records `checkpoint.as_node_provided` and
`checkpoint.zero_tasks_scheduled` on a `checkpoint:update_state` span. The
example then prints exactly what `agent-trace show <run_id>` (or `replay`)
prints for each run — the warning fires only for the broken one.

## Run

```bash
pip install agent-observability-trace-cli[langgraph]
python examples/10-langgraph-checkpointer-interrupt-resume/example.py
```

Then inspect either run directly:

```bash
agent-trace show <broken run ID>     # prints the "Zero tasks scheduled..." warning
agent-trace show <fixed run ID>      # prints nothing extra — the resume worked
agent-trace inspect <broken run ID>  # runs the full pattern-check library too
```

## What this demonstrates (and what it doesn't)

agent-trace now gives a developer an automated, one-line signal for #4217's
exact failure mode instead of the silent `next: []`/zero-callback experience
the original reporter had to escalate to a LangGraph maintainer to
diagnose. It does **not** predict or prevent the underlying pregel
scheduling behavior — `as_node` inference is LangGraph's own documented
contract — it only makes the resulting broken state (an update that
schedules no further tasks) visible after the fact.
