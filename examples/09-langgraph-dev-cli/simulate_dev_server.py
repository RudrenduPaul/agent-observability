"""
Simulates the `langgraph dev`/LangGraph Studio process lifecycle against
`graph.py` — issue #4798 — without needing `langgraph-cli`/`langgraph-api`
installed or a real MCP server running. See this directory's README.md for
the full explanation and the real (non-simulated) commands to run against
an actual `langgraph dev` server.

What "the langgraph dev lifecycle" means here, concretely: the framework
imports your `graph.py` and calls your `make_graph()`-style factory exactly
**once**, at server startup — before any `.invoke()` call exists — then
reuses that one compiled graph object to serve every subsequent run for the
life of the process. This script reproduces exactly that shape twice, so
you can see the difference recording makes:

Scenario A — wired correctly (what this example recommends):
  AGENT_TRACE_AUTO_RECORD is active *before* `graph.py` is imported (the
  real equivalent: `agent-trace run -- langgraph dev`, or setting the env
  var yourself before running `langgraph dev`). Construction-phase work
  (the simulated MCP tool-listing call) lands as a nested span under the
  persistent auto-record trace, `make_graph()` binds a `LangGraphTracer`
  onto the compiled graph because `tracer.active_trace` is already set, and
  every subsequent `.invoke()` call — simulating separate served runs — is
  captured into that same trace too.

Scenario B — today's status quo (no code changes, nothing set):
  No trace is active when `make_graph()` runs. `instrument_graph_factory`
  still activates its own *scoped* trace for the duration of the factory
  call alone (so the construction-phase HTTP call is still captured,
  in its own tiny run) — but that trace closes the moment `make_graph()`
  returns, so the compiled graph has no callback bound, and every
  subsequent `.invoke()` call produces zero spans anywhere. This is
  the exact gap issue #4798 describes.

Run:
    python examples/09-langgraph-dev-cli/simulate_dev_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from agent_trace import tracer
from agent_trace._replay.fixture import Fixture


def _print_run_summary(run_dir: Path, label: str) -> None:
    trace_json = run_dir / "trace.json"
    fixture_db = run_dir / "fixture.db"

    span_count = 0
    span_names: list[str] = []
    if trace_json.exists():
        import json

        data = json.loads(trace_json.read_text(encoding="utf-8"))
        span_names = [s["name"] for s in data.get("spans", [])]
        span_count = len(span_names)

    exchange_count = 0
    if fixture_db.exists():
        with Fixture(fixture_db) as fixture:
            exchange_count = fixture.exchange_count()

    print(f"  {label}")
    print(f"    run_dir:    {run_dir}")
    print(f"    spans:      {span_count}  {span_names}")
    print(f"    exchanges:  {exchange_count}")


def scenario_a_wired_correctly() -> None:
    print("=== Scenario A: AGENT_TRACE_AUTO_RECORD active before import ===")
    run_dir = tracer.start_auto_record(
        name="langgraph-dev-simulated", run_id="dev-scenario-a"
    )
    try:
        # The real equivalent of this import is langgraph_api importing
        # your graph.py exactly once, at server startup, because
        # AGENT_TRACE_AUTO_RECORD was already set before the process
        # (or `langgraph dev` itself, via `agent-trace run`) started.
        from graph import make_graph

        graph: Any = make_graph({})

        for i in range(2):
            result = graph.invoke({"messages": [f"request {i}"]})
            print(f"  invoke #{i}: {result['messages'][-1]}")
    finally:
        tracer.stop_auto_record()

    _print_run_summary(
        run_dir, "Result (ONE trace covers construction + both invokes):"
    )
    print()


def scenario_b_status_quo() -> None:
    print("=== Scenario B: no AGENT_TRACE_AUTO_RECORD, no start_trace() ===")
    # graph.make_graph is already imported (Python caches modules) — call it
    # again directly to reproduce "make_graph() called with nothing active",
    # exactly as if this were a *separate* process that never set the env
    # var. instrument_graph_factory() still activates a scoped trace for the
    # construction call itself (see its docstring), but that trace closes
    # the instant make_graph() returns.
    from graph import make_graph

    graph: Any = make_graph({})

    # Find the scoped construction-only run instrument_graph_factory created.
    trace_dir = tracer._trace_dir
    construction_run_dirs = sorted(
        (d for d in trace_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
        key=lambda d: d.stat().st_mtime,
    )
    if construction_run_dirs:
        _print_run_summary(
            construction_run_dirs[-1],
            "Construction-phase capture (its own short-lived scoped trace):",
        )

    result = graph.invoke({"messages": ["request 0"]})
    print(f"  invoke #0 (uninstrumented): {result['messages'][-1]}")
    print(
        "  -> No LangGraphTracer was bound (tracer.active_trace was already "
        "None again by the time make_graph() returned), so this invoke() "
        "produced zero spans anywhere — the exact gap issue #4798 describes."
    )
    print()


def main() -> None:
    scenario_a_wired_correctly()
    scenario_b_status_quo()
    print(
        "Takeaway: AGENT_TRACE_AUTO_RECORD (directly, or via "
        "`agent-trace run -- langgraph dev`) is what turns 'construction-phase "
        "HTTP calls captured, invocations blind' into 'the whole process, "
        "construction through every served run, is one observable trace.'"
    )


if __name__ == "__main__":
    main()
