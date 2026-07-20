# Example 05 — Parallel-Tool Command.PARENT Routing (Capture and Gap)

Demonstrates the failure class behind issue
[#7129](https://github.com/langchain-ai/langgraph/issues/7129): parallel
tool calls that each return `Command(graph=Command.PARENT, update={...})`
to hand a value up to the parent graph, where LangGraph's own internal
merge can silently keep only one of N proposed updates.

No API key required — every "model" decision is a plain Python stand-in.

## Prerequisites

```bash
pip install agent-observability-trace-cli[langgraph]
```

## Part A — the capture gap (a real, live graph run)

Three tool nodes are fanned out in parallel via `Send`, each returning a
`Command(graph=Command.PARENT, ...)`. `LangGraphTracer`'s tool spans
(`on_tool_start`/`on_tool_end`) capture only a name and `OK` status for
each — the actual Command *payload* (what was proposed, and to which
channel) is invisible in the span tree.

This part deliberately uses three distinct channels (no LangGraph-level
conflict) so the run completes cleanly and the *capture* gap — not a
crash — is what's on display:

```
6 tool-dispatch span(s) captured.
Notice: each span shows only a name + OK status. The actual
Command(graph=Command.PARENT, update={...}) payload each task proposed —
the exact thing you'd need to see to root-cause a dropped update — is not
on any span today.
```

## Part B — the new diagnostic

`agent_trace.integrations.langgraph_state_diff.wrap_checkpointer()` wraps
any `BaseCheckpointSaver` to snapshot every superstep's *proposed* writes
(`put_writes`, one call per parallel task) against what actually landed in
the finalized checkpoint (`put`). When N>1 tasks propose a write to the
same channel and fewer than N survive, it emits a `checkpoint:superstep_merge`
span with the exact drop as a countable fact:

```
1 superstep-merge span(s) captured:
  channel='result'  proposed=3  survived=1  dropped=2  dropped_task_ids='tool_call_alpha,tool_call_gamma'
```

Reproducing the exact multi-agent/version-specific trigger conditions for
#7129 inside a minimal, no-API-key example isn't reliable across LangGraph
versions — a live run of the natural "two parallel writes to one plain
channel" shape in the currently installed LangGraph version raises a loud
`InvalidUpdateError` rather than silently dropping (see
`src/agent_trace/integrations/langgraph_state_diff.py`'s module docstring).
Part B demonstrates the diagnostic's mechanism directly against the real
`BaseCheckpointSaver` contract — the same `put_writes`/`put` calls Pregel's
own scheduler makes internally regardless of which LangGraph mechanism
produced them — instead of chasing a brittle full reproduction.

## Run

```bash
python examples/05-parallel-command-parent-routing/example.py
```

## Usage in your own graph

```python
from agent_trace import tracer
from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = wrap_checkpointer(InMemorySaver(), tracer=tracer)
graph = builder.compile(checkpointer=checkpointer)

with tracer.start_trace("my-graph") as trace:
    graph.invoke(input, config={"configurable": {"thread_id": "t1"}})
# trace now carries a "checkpoint:superstep_merge" span (if any superstep
# had >1 task propose a write to the same channel with fewer than N survivors).
```

## See also

- `src/agent_trace/integrations/langgraph_state_diff.py` — the diagnostic module
- `examples/06-langgraph-handoff-parallel-tools/` — the related #5277
  `INVALID_CHAT_HISTORY` race (multi-agent handoff + parallel tool calls,
  a distinct failure shape captured via the callback layer, not the
  checkpointer)
