# LangGraph Integration

agent-trace records and replays LangGraph runs by intercepting HTTP at the
transport layer. Every LLM call and tool API call made inside a graph node is
captured verbatim and can be replayed without touching the real endpoints.

---

## 1. Install

```bash
pip install agent-trace[langgraph]
```

This installs `langgraph>=0.2,<1.0` and `langchain-core>=0.3` alongside
`agent-trace`. Pin LangGraph versions carefully — agent-trace's callback
integration targets the `0.2.x` graph API. Check the `[project.optional-dependencies]`
section of `pyproject.toml` for the exact pinned range before upgrading.

---

## 2. Quick start

A minimal 2-node StateGraph with recording and replay:

```python
from __future__ import annotations
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
import httpx

from agent_trace import tracer, replay
from agent_trace.integrations.langgraph import LangGraphTracer
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.core.trace import Trace
import json
from pathlib import Path

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class AgentState(TypedDict):
    question: str
    research: str
    answer: str


def research_node(state: AgentState) -> AgentState:
    """Call the LLM to research the question."""
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": f"Research briefly: {state['question']}"}
            ],
            "max_tokens": 200,
        },
    )
    resp.raise_for_status()
    return {"research": resp.json()["choices"][0]["message"]["content"]}


def respond_node(state: AgentState) -> AgentState:
    """Synthesize a final answer from the research."""
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n"
                        f"Research: {state['research']}\n"
                        "Give a one-sentence answer."
                    ),
                }
            ],
            "max_tokens": 100,
        },
    )
    resp.raise_for_status()
    return {"answer": resp.json()["choices"][0]["message"]["content"]}


graph = (
    StateGraph(AgentState)
    .add_node("research", research_node)
    .add_node("respond", respond_node)
    .add_edge("research", "respond")
    .add_edge("respond", END)
    .set_entry_point("research")
    .compile()
)


# --- Record (requires OPENAI_API_KEY) ---
with tracer.start_trace("langgraph-quickstart", record=True) as trace:
    result = graph.invoke(
        {"question": "What is LangGraph?"},
        config={"callbacks": [LangGraphTracer(tracer=tracer, trace=trace)]},
    )
    run_id = trace.run_id

print(f"Recorded: {run_id}")
print(f"Answer: {result['answer']}")

# --- Replay (no API key needed) ---
with replay(run_id) as ctx:
    result2 = graph.invoke({"question": "What is LangGraph?"})

print(f"Replayed answer: {result2['answer']}")
assert result["answer"] == result2["answer"], "Replay must return identical answer"

# --- Show span tree ---
trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
trace_obj = Trace.from_dict(json.loads(trace_path.read_text()))
StdoutExporter().export(trace_obj)
```

---

## 3. What gets traced

agent-trace traces the following during a LangGraph recording:

| What | How |
|------|-----|
| LLM API calls (OpenAI, Anthropic, etc.) | httpx/requests transport interception |
| Tool API calls (any HTTP endpoint called inside a node) | same transport interception |
| Span tree with node boundaries | `LangGraphTracer` callback handler (automatic) |
| ChatModel spans (ChatOpenAI, ChatAnthropic, etc.) | `LangGraphTracer.on_chat_model_start` callback |
| Exception details | `span.record_exception(exc)` on any unhandled exception |

Pass `LangGraphTracer` in `config["callbacks"]` to get automatic spans for every
node, LLM call, and tool call — no manual `tracer.span(...)` instrumentation
required inside node functions.

---

## 4. Span attributes added

When you manually instrument node functions with `tracer.span(...)`, you can
set these standard attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `langgraph.node_name` | str | Name of the LangGraph node |
| `llm.model` | str | Model identifier, e.g. `"gpt-4o-mini"` |
| `llm.token_count.prompt` | int | Prompt token count from the API response |
| `llm.token_count.completion` | int | Completion token count |
| `llm.token_count.total` | int | Total tokens billed |
| `tool.name` | str | Name of the tool called |
| `tool.input` | str | JSON-serialized tool input (keep short) |

Example:

```python
def research_node(state: AgentState) -> AgentState:
    with tracer.span("research") as span:
        span.set_attribute("langgraph.node_name", "research")
        span.set_attribute("llm.model", "gpt-4o-mini")

        resp = httpx.post(...)
        resp.raise_for_status()

        data = resp.json()
        usage = data.get("usage", {})
        span.set_attribute("llm.token_count.prompt", usage.get("prompt_tokens", 0))
        span.set_attribute("llm.token_count.completion", usage.get("completion_tokens", 0))
        span.set_attribute("llm.token_count.total", usage.get("total_tokens", 0))

        return {"research": data["choices"][0]["message"]["content"]}
```

---

## 5. Replaying a LangGraph failure — step by step

1. **Run the graph with `record=True`** until it fails:

```python
with tracer.start_trace("debug-run", record=True) as trace:
    try:
        result = graph.invoke({"question": "..."})
    except Exception as exc:
        print(f"Failed at run {trace.run_id}: {exc}")
        # The fixture.db is still written even if the graph raised.
        # Spans that were open when the exception propagated are closed
        # with status=ERROR by the Tracer.__exit__ logic.
```

2. **Inspect the trace** to see which node failed:

```bash
agent-trace show <run_id>
```

Look for spans with `[ERR]` status and `exception.message` attributes.

3. **Replay and add debug instrumentation**:

```python
with replay("<run_id>") as ctx:
    # Add a breakpoint or extra logging — it is safe because no API calls
    # will be made regardless of how many times you re-enter this block.
    import pdb; pdb.set_trace()
    result = graph.invoke({"question": "..."})
```

4. **Iterate** — modify node logic, re-replay, check the output. No tokens
   are spent until you re-record with the fix applied.

---

## 6. Known limitations

- **LangGraph version pin:** agent-trace targets `langgraph>=0.2,<1.0`. The
  `StateGraph` API changed in 0.2. Check the pinned range in `pyproject.toml`
  before upgrading LangGraph.

- **Async graphs:** `graph.ainvoke(...)` uses `httpx.AsyncClient` internally.
  agent-trace v0.1 only patches `httpx.Client` (synchronous). Async graphs
  will pass through to the live network during replay. Async support is planned
  for v0.2.

- **Streaming:** LLM streaming calls (SSE responses) are recorded as a single
  buffered response. The recorded body is the full concatenated stream. During
  replay, the response is returned as a single chunk rather than as an SSE
  stream. If your node code reads `.iter_text()` or `.iter_lines()`, it will
  still work because httpx's response model supports iteration over a single
  content buffer, but chunking behavior will differ.

- **Conditional edges:** LangGraph's conditional routing is driven by the
  return value of your routing function, which depends on node outputs. Since
  node outputs are deterministic during replay, conditional edges route the
  same way they did during recording.

- **Conditional-edge dispatch exceptions (`trace=False`):** LangGraph
  deliberately builds a conditional edge's routing dispatch
  (`add_conditional_edges`) as an internal component with `trace=False` —
  it never fires `on_chain_start`/`on_chain_error`, so by default an
  exception raised inside the routing function itself (e.g. a `KeyError`
  when a router's return value doesn't match a registered destination) is
  invisible to any callback-based tool. `LangGraphTracer` patches around
  this specific gap (best-effort — it touches a LangGraph internal module,
  `langgraph._internal._runnable.RunnableCallable`, and degrades silently
  to "not captured" if that internal shape changes on a future LangGraph
  version) and records a `branch:dispatch` span with `status=ERROR` when
  this happens. This does **not** mean every `trace=False` component in
  LangGraph is now traced — only the conditional-edge dispatch case
  specifically; other internal `trace=False` components (channel writes,
  `ToolNode`'s own dispatch, etc.) remain outside agent-trace's callback
  coverage.

---

## 7. Error classification and LangGraph-internal control-flow signals

- **`error.origin`/`error.known_pattern`:** every span that closes `ERROR`
  is tagged with `error.origin` (`"provider"` for an LLM-SDK exception,
  `"chain"` for LangGraph/LangChain framework code, `"application"`
  otherwise) and, when the exception message matches a previously
  root-caused failure signature (e.g. LangGraph's
  `ErrorCode.INVALID_CHAT_HISTORY`), `error.known_pattern`. `agent-trace
  show <run_id>` prints an "Error classification" summary using these
  attributes so you don't have to read raw `trace.json` to spot the pattern.

- **`Command`/`ParentCommand`/`GraphInterrupt` are not application errors:**
  LangGraph raises these internally to implement multi-agent handoff jumps
  (`Command(graph=Command.PARENT, ...)`) and `interrupt()` pauses — not real
  failures. `LangGraphTracer` special-cases them: the span closes `OK` with
  `langgraph.handoff=true` (a handoff) or `langgraph.interrupted=true` (a
  pause) instead of `ERROR`, so they don't drown out genuine failures when
  scanning a trace. See `examples/06-langgraph-handoff-parallel-tools/` for
  a worked example.

- **`CANCELLED` is distinct from `ERROR`:** a span ended by
  `asyncio.CancelledError` closes with `status=CANCELLED`, not `ERROR` — a
  cancelled run and a genuine crash are different things when you're
  diagnosing why a checkpoint didn't get written.

## 8. Per-superstep state-merge diagnostics (parallel `Command.PARENT` routing)

`agent_trace.integrations.langgraph_state_diff.wrap_checkpointer()` wraps
any `BaseCheckpointSaver` to detect when N>1 parallel tasks in the same
Pregel superstep propose a write to the same channel but fewer than N
survive in the persisted checkpoint — the shape behind
[langgraph#7129](https://github.com/langchain-ai/langgraph/issues/7129).
When it happens, a `checkpoint:superstep_merge` span records the exact
count: how many were proposed, how many survived, and which task IDs lost
their update. See `examples/05-parallel-command-parent-routing/` for a
worked example and the module's own docstring for the heuristic's scope
(it diffs proposed-vs-persisted channel values; it does not re-implement
Pregel's channel-merge algorithm).

---

## 9. Troubleshooting

### Spans not appearing in the trace

**Cause:** `tracer.span(...)` calls are outside a `tracer.start_trace(...)` block.
Spans created without an active trace are detached — they are created but not
registered to any trace, so they do not appear in `trace.json`.

**Fix:** Ensure the graph invoke is inside `with tracer.start_trace(...)`.

### LLM calls not intercepted (live calls during replay)

**Symptom:** During replay, you see network activity or `NetworkGuardError`.

**Most common causes:**

1. The `httpx.Client` or `openai.OpenAI()` client was instantiated *before*
   the `replay(...)` context was entered. The patch only applies to clients
   created inside the context.

2. Your LangGraph nodes use `httpx.AsyncClient`. Async client patching is not
   yet implemented.

3. The OpenAI SDK version you are using bypasses `httpx.Client.__init__` and
   creates a transport directly. Check with `pip show openai` and compare to
   the versions that were tested in `pyproject.toml`.

**Fix for (1):** Move all SDK client construction inside the replay block, or
use a factory function that creates a fresh client each time.
