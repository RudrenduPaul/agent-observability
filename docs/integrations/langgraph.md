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
  agent-trace patches both `httpx.Client` (sync) and `httpx.AsyncClient`
  (async) — see `src/agent_trace/interceptor/httpx_hook.py`'s
  `RecordingTransport`/`AsyncRecordingTransport` and
  `Tracer._patch_httpx()`, which installs the patch on both classes at
  request-dispatch time (`_transport_for_url`), not just at client
  construction time. Async graphs are recorded and replayed the same as
  sync ones; no special configuration is needed.

- **Streaming:** By default, LLM streaming calls (SSE responses) are
  recorded as a single buffered response — the recorded body is the full
  concatenated stream, and replay returns it as one chunk rather than a
  live SSE stream. `RecordingTransport`/`AsyncRecordingTransport`
  (`src/agent_trace/interceptor/httpx_hook.py`) also support a
  non-buffering, pass-through `stream=True` mode that tees the real
  streamed response to both the caller and the fixture as it arrives (see
  `_TeeSyncByteStream`/`_TeeAsyncByteStream`), but that mode is only
  reachable today by constructing the transport directly — the high-level
  `Tracer.start_trace(record=True)` path does not yet expose a `stream=`
  kwarg to opt into it.

- **Conditional edges:** LangGraph's conditional routing is driven by the
  return value of your routing function, which depends on node outputs. Since
  node outputs are deterministic during replay, conditional edges route the
  same way they did during recording.

- **Conditional-edge routing dispatch (`trace=False`):** LangGraph
  deliberately builds a conditional edge's routing dispatch
  (`add_conditional_edges`, and the `should_continue`/
  `post_model_hook_router` edges `create_react_agent` inserts internally)
  as an internal component with `trace=False` — it never fires
  `on_chain_start`/`on_chain_end`/`on_chain_error`, so by default neither a
  successful routing decision nor an exception raised inside the routing
  function itself (e.g. a `KeyError` when a router's return value doesn't
  match a registered destination) is visible to any callback-based tool.
  `LangGraphTracer` patches around this gap (best-effort — it touches a
  LangGraph internal module, `langgraph._internal._runnable.RunnableCallable`,
  and degrades silently to "not captured" if that internal shape changes on
  a future LangGraph version) and records a `branch:dispatch` span for
  **every** dispatch, success or failure — `branch.router_name` (the
  underlying router function's name), `branch.registered_destinations`, and
  `status=ERROR` plus the usual exception/classification attributes when the
  router itself raises. `ToolNode`'s `InjectedState`/`InjectedStore`
  argument-injection step (`_inject_tool_args`) is instrumented the same
  way: a `tool_inject:<name>` span per tool call records `tool.injection_ran`
  and `tool.injected_arg_keys`, since injection happens before the tool's
  own `on_tool_start` callback fires and so has no span of its own to
  attach to. `agent_trace.integrations.langgraph.find_tool_params_shaped_like_state()`
  additionally flags, at `LangGraphTracer(graph=...)` construction time, any
  tool parameter that shares a name with a real state field but was never
  annotated `InjectedState`/`InjectedStore` — the model-facing schema shape
  behind [langgraph#3266](https://github.com/langchain-ai/langgraph/issues/3266).
  See `examples/08-langgraph-react-agent-tool-arg-injection/` for a worked
  example. This does **not** mean every `trace=False` component in
  LangGraph is now traced — only conditional-edge routing dispatch and
  `ToolNode` argument injection specifically; other internal `trace=False`
  components (channel writes, etc.) remain outside agent-trace's callback
  coverage.

- **Pregel-internal state/channel-routing errors:** agent-trace's current
  architecture (the HTTP interceptor plus LangChain/LangGraph callback
  spans) cannot capture a failure that happens entirely *inside* LangGraph's
  own pregel scheduler and never surfaces as an exception or a network call
  — e.g. a state write silently dropped because a channel (such as the `ui`
  channel used by `push_ui_message`) isn't registered on the compiled
  graph's schema, which LangGraph itself only reports as a `logging.warning`
  call (`"wrote to unknown channel X, ignoring it"`,
  [langgraph#5464](https://github.com/langchain-ai/langgraph/issues/5464)).
  Neither the callback layer nor the HTTP interceptor ever sees this class
  of bug, regardless of how complete the rest of a trace otherwise is.
  `agent_trace.interceptor.logging_hook.capture_logging()` narrows this gap
  for the specific case where LangGraph *does* at least log a warning (by
  attaching a `logging.Handler` to the `langgraph` logger namespace for the
  duration of a `with` block) — but there is currently no general
  instrumentation of pregel's own internal scheduling/channel-routing
  decisions, so a silent channel-routing failure that produces neither a
  log line nor an exception remains an outright blind spot. If you're
  evaluating agent-trace specifically for this failure class, know that it
  gets zero automatic value from agent-trace's callback/HTTP capture today.

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

1. The OpenAI SDK version you are using bypasses `httpx.Client`/
   `httpx.AsyncClient` entirely and creates a transport directly. Check with
   `pip show openai` and compare to the versions that were tested in
   `pyproject.toml`.

2. Streaming was recorded in the default buffered mode and your node code
   depends on incremental chunk-arrival timing rather than just the final
   concatenated content — see the streaming note above.

Both `httpx.Client` (sync) and `httpx.AsyncClient` (async) are patched, and
the patch is applied at request-dispatch time (`_transport_for_url`), not
just at client-construction time — so a client instantiated before
`replay(...)`/`start_trace(record=True)` is entered is still intercepted,
including LangGraph nodes that use `httpx.AsyncClient` via `graph.ainvoke(...)`.

---

## 10. Wiring agent-trace into `langgraph dev` / LangGraph Studio

Every example above (and the quick-start in section 2) assumes a
user-owned script calling `graph.invoke()` directly, inside a
`with tracer.start_trace(...)` block you write yourself. `langgraph dev`/
LangGraph Studio inverts that: `langgraph_api` imports your `graph.py` and
calls your `make_graph()`-style factory function (the one `langgraph.json`'s
`graphs` config points at) exactly **once**, at server startup — before any
`.invoke()` call exists anywhere for you to wrap. A bug in that
construction phase itself (MCP client setup, tool loading, config parsing —
[langgraph#4798](https://github.com/langchain-ai/langgraph/issues/4798)) is
therefore unreachable by `LangGraphTracer` no matter how carefully you wire
it into your own invoke() call, because there is no code of yours in the
loop to wire it into.

See `examples/09-langgraph-dev-cli/` for a complete, runnable reference
implementation (no `langgraph-cli` install required to run the example
itself). Summary of the three pieces it demonstrates:

### `AGENT_TRACE_AUTO_RECORD` — process-wide activation, no `with` block

Set before the process that imports `agent_trace` starts — directly, or via
the `agent-trace run` CLI wrapper — this activates recording on the global
`tracer` singleton the moment `agent_trace` is first imported, with no
`with tracer.start_trace(...)` block required anywhere in your code:

```bash
# Directly:
AGENT_TRACE_AUTO_RECORD=1 langgraph dev

# Or via the CLI wrapper (also sets a run_id and prints where it's recording to):
agent-trace run -- langgraph dev
```

Recording then stays active for the server's entire remaining lifetime
(until the process exits, or `tracer.stop_auto_record()` is called
explicitly) — one trace/fixture pair accumulates every HTTP exchange and
LangGraph span for the process's whole life, not one logical run per
served request. See `Tracer.start_auto_record()`'s docstring in
`src/agent_trace/__init__.py` for the full API (env vars honored:
`AGENT_TRACE_AUTO_RECORD`, `AGENT_TRACE_RUN_ID`,
`AGENT_TRACE_AUTO_RECORD_NAME`, `AGENT_TRACE_TRACE_DIR`).

### `agent-trace run` — subprocess-wrapping CLI command

```bash
agent-trace run -- langgraph dev
agent-trace run --run-id my-run -- langgraph dev --port 2024
```

Execs the given command with `AGENT_TRACE_AUTO_RECORD=1` (plus
`AGENT_TRACE_RUN_ID`/`AGENT_TRACE_AUTO_RECORD_NAME`/`AGENT_TRACE_TRACE_DIR`)
already set in its environment, and relays the child process's exit code.
This is the recommended way to launch `langgraph dev` under agent-trace —
it needs no code changes in your `graph.py` at all, only this one
command-line change to how you start the dev server.

### `instrument_graph_factory()` — pre-invocation (construction-phase) hook

```python
from agent_trace import tracer
from agent_trace.integrations.langgraph import instrument_graph_factory

@instrument_graph_factory(tracer)
def make_graph(config: dict) -> CompiledStateGraph:
    mcp_client = MultiServerMCPClient(...)   # now captured
    tools = mcp_client.get_tools()
    return build_graph(tools)
```

Decorates the factory function itself so tracing/recording is active while
the graph is being *built*, not only while it is later invoked. If a trace
is already active (e.g. `AGENT_TRACE_AUTO_RECORD` fired above), the factory
call becomes a nested `graph-construction` span; if not, it gets its own
scoped `start_trace(record=True)` for the duration of the call, so
construction-phase HTTP calls are captured either way instead of being an
outright blind spot. Combine with
`graph.with_config(callbacks=[LangGraphTracer(tracer=tracer,
trace=tracer.active_trace)])` inside the factory (only meaningful when
`tracer.active_trace` is not None — i.e. `AGENT_TRACE_AUTO_RECORD` was set)
to also trace every subsequent `.invoke()`/`.stream()` call the framework
makes against the compiled graph, not just construction — see
`examples/09-langgraph-dev-cli/graph.py` for the complete pattern.
