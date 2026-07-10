# Example 09 — Wiring agent-trace into `langgraph dev` / LangGraph Studio (#4798)

Every other example in this repo assumes a user-owned script calling
`graph.invoke()` directly, inside a `with tracer.start_trace(...)` block the
developer writes themselves. `langgraph dev`/LangGraph Studio inverts that:
the framework imports your `graph.py` and calls your `make_graph()`-style
factory function exactly **once**, at server startup — before any
`.invoke()` call exists anywhere for you to wrap. This example is the
reference implementation for that deployment shape, and closes the gap
behind [langgraph#4798](https://github.com/langchain-ai/langgraph/issues/4798)
(an MCP tool-loading failure that happens entirely inside the construction
phase, where the pre-existing `LangGraphTracer` had no hook point at all).

## Prerequisites

```bash
pip install agent-trace[langgraph]
```

`langgraph-cli`/`langgraph-api` are **not** required to run this example —
`simulate_dev_server.py` reproduces the exact process lifecycle
(`make_graph()` called once, then invoked many times) without them. See
"Running this for real" below for the actual `langgraph dev` commands.

## The three pieces

### 1. `graph.py` — the `make_graph()` entry point

`langgraph.json`'s `graphs` config points at `graph:make_graph`, exactly
like a real LangGraph Platform deployment. Two things make it observable:

```python
@instrument_graph_factory(tracer)
def make_graph(config: dict | None = None) -> Any:
    tools = _load_tools_from_mcp_server()   # captured when recording is active
    ...
    graph = builder.compile()
    if tracer.active_trace is not None:
        graph = graph.with_config(
            callbacks=[LangGraphTracer(tracer=tracer, trace=tracer.active_trace)]
        )
    return graph
```

- **`@instrument_graph_factory(tracer)`** activates recording for the
  duration of the factory call itself — so MCP client setup / tool loading
  / config parsing is captured, whether or not anything else was already
  recording.
- **`graph.with_config(callbacks=[LangGraphTracer(...)])`** binds a
  callback to the *compiled* graph, so every future `.invoke()`/`.stream()`
  call is traced automatically — regardless of who calls it later
  (`langgraph dev`'s own request-handling code, never your code). This only
  produces real span data when a trace is already active at the moment
  `make_graph()` runs (see next section) — otherwise `make_graph()` still
  returns a perfectly working, uninstrumented graph.

### 2. `AGENT_TRACE_AUTO_RECORD` — the activation mechanism

`instrument_graph_factory`'s "was a trace already active?" check is decided
entirely by *when* recording activates relative to `graph.py`'s import.
`AGENT_TRACE_AUTO_RECORD=1`, read once when `agent_trace` is first
imported, is what makes recording active **before** `langgraph dev` ever
imports your `graph.py` — with no code of yours in the loop:

```bash
# Directly:
AGENT_TRACE_AUTO_RECORD=1 langgraph dev

# Or via the CLI wrapper (sets the env var for you, plus a run_id):
agent-trace run -- langgraph dev
```

Either way, by the time `langgraph_api` imports `graph.py` and calls
`make_graph()`, `tracer.active_trace` is already the persistent auto-record
trace — so construction lands as a nested span, and the `with_config(...)`
callback binding actually has a real trace to attach to. Recording then
stays active for the server's entire remaining lifetime (until the process
exits, or `tracer.stop_auto_record()` is called) — see
`Tracer.start_auto_record()`'s docstring in `src/agent_trace/__init__.py`
for the coarser-grained tradeoff this implies (one trace per process
lifetime, not one per served run).

### 3. `simulate_dev_server.py` — runnable, no `langgraph-cli` needed

```bash
python examples/09-langgraph-dev-cli/simulate_dev_server.py
```

Reproduces the exact `langgraph dev` process lifecycle — `make_graph()`
called once, then `.invoke()` called multiple times against the same
compiled graph — twice, so you can see the difference recording makes:

- **Scenario A (wired correctly):** `tracer.start_auto_record()` is called
  before `graph.py` is imported (the in-process equivalent of
  `AGENT_TRACE_AUTO_RECORD=1` being set before the real server starts).
  Construction lands as a nested span, and **both** simulated `.invoke()`
  calls land in the same trace as real spans.
- **Scenario B (today's status quo — nothing set):** construction still
  gets its own short-lived scoped trace (one HTTP exchange captured), but
  that trace closes the instant `make_graph()` returns — so the returned
  graph has no callback bound, and every subsequent `.invoke()` call
  produces zero spans anywhere.

Confirmed live output (abridged):

```
=== Scenario A: AGENT_TRACE_AUTO_RECORD active before import ===
  invoke #0: (using tools: search, book_flight) ok, done.
  invoke #1: (using tools: search, book_flight) ok, done.
  Result (ONE trace covers construction + both invokes):
    spans:      5  ['graph-construction', 'node:LangGraph', 'node:call_model', 'node:LangGraph', 'node:call_model']
    exchanges:  1

=== Scenario B: no AGENT_TRACE_AUTO_RECORD, no start_trace() ===
  Construction-phase capture (its own short-lived scoped trace):
    spans:      0  []
    exchanges:  1
  invoke #0 (uninstrumented): (using tools: search, book_flight) ok, done.
  -> No LangGraphTracer was bound ... zero spans anywhere.
```

## Running this for real (actual `langgraph dev`)

```bash
pip install langgraph-cli[inmem]
cd examples/09-langgraph-dev-cli

# Either of these — both set AGENT_TRACE_AUTO_RECORD before langgraph dev
# ever imports graph.py:
AGENT_TRACE_AUTO_RECORD=1 langgraph dev
# or:
agent-trace run -- langgraph dev
```

Then hit the server (LangGraph Studio, or the REST API it exposes) to
trigger a run. `agent-trace run` prints the `run_id`/`trace_dir` it's
recording into on startup — inspect the result the same way as any other
recording:

```bash
agent-trace show <run_id>
agent-trace replay <run_id>
```

## Why this matters

Before this example (and the `AGENT_TRACE_AUTO_RECORD`/
`instrument_graph_factory`/`agent-trace run` mechanisms it demonstrates),
there was no documented pattern for wiring agent-trace into a
`langgraph dev`/Studio deployment at all — every doc and example assumed a
user-owned invocation. A construction-phase bug like #4798 (an MCP
tool-loading failure with no `.invoke()` call in sight) was invisible to
agent-trace no matter how carefully you wired `LangGraphTracer` into your
own code, because there was no code of yours in the loop to wire it into.
