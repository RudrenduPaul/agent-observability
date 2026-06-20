"""
agent-trace record/replay demo — no API key required.

Runs a 12-step pure-Python LangGraph pipeline that fails deterministically at
step 7.  Records the span tree to a local fixture, then replays it and shows
the same failure reproduced without any network calls.

This script is the source for the README demo; record a GIF by running:

    asciinema rec demo.cast
    python demos/record_replay_demo.py
    asciinema play demo.cast

Run directly:

    uv run --extra langgraph python demos/record_replay_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    sys.exit(
        "langgraph is not installed.\n"
        "Run: uv add agent-trace[langgraph] or pip install agent-trace[langgraph]"
    )

from agent_trace import Tracer, replay
from agent_trace.integrations.langgraph import LangGraphTracer

FAIL_AT_STEP = 7
TOTAL_STEPS = 12

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


class PipelineState(TypedDict):
    completed_steps: list[str]
    current_step: int


def _make_step(n: int):
    def step(state: PipelineState) -> PipelineState:
        if n == FAIL_AT_STEP:
            raise RuntimeError(
                f"Step {n}: upstream dependency returned null — "
                "cannot continue pipeline"
            )
        print(f"  {GREEN}✓{RESET} step_{n:02d}  completed")
        return {
            "completed_steps": state["completed_steps"] + [f"step_{n:02d}"],
            "current_step": n,
        }

    step.__name__ = f"step_{n:02d}"
    return step


def build_graph():
    g = StateGraph(PipelineState)
    for i in range(1, TOTAL_STEPS + 1):
        g.add_node(f"step_{i:02d}", _make_step(i))
    g.set_entry_point("step_01")
    for i in range(1, TOTAL_STEPS):
        g.add_edge(f"step_{i:02d}", f"step_{(i + 1):02d}")
    g.add_edge(f"step_{TOTAL_STEPS:02d}", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    trace_dir = Path(tempfile.mkdtemp(prefix="agent-trace-demo-"))
    run_id = "pipeline-run-001"

    # -------------------------------------------------------------------
    # RECORD PASS
    # -------------------------------------------------------------------
    print(f"\n{BOLD}{CYAN}=== RECORD MODE ==={RESET}")
    print(f"Running {TOTAL_STEPS}-step pipeline  (will fail at step {FAIL_AT_STEP})\n")

    t = Tracer(trace_dir=trace_dir)
    try:
        with t.start_trace("pipeline-demo", record=True, run_id=run_id) as rec_trace:
            cb = LangGraphTracer(tracer=t, trace=rec_trace)
            graph.invoke(
                {"completed_steps": [], "current_step": 0},
                config={"callbacks": [cb]},
            )
    except RuntimeError as exc:
        print(f"\n  {RED}✗ step_{FAIL_AT_STEP:02d}  {exc}{RESET}")

    spans = rec_trace.spans
    fixture_db = trace_dir / run_id / "fixture.db"

    print(f"\n{BOLD}Recorded:{RESET}")
    print(f"  {len(spans)} spans captured")
    print(f"  fixture → {fixture_db}")
    print(f"  {len([s for s in spans if 'step_0' in s.name or 'step_' in s.name])} node spans")
    error_spans = [s for s in spans if s.status.value == "ERROR"]
    print(f"  {len(error_spans)} error span(s)")

    # -------------------------------------------------------------------
    # REPLAY PASS
    # -------------------------------------------------------------------
    print(f"\n{BOLD}{YELLOW}=== REPLAY MODE ==={RESET}")
    print("(No network calls — all state served from local fixture)\n")

    try:
        with replay(run_id, trace_dir=trace_dir):
            graph.invoke(
                {"completed_steps": [], "current_step": 0},
            )
    except RuntimeError as exc:
        print(f"\n  {RED}✗ step_{FAIL_AT_STEP:02d}  {exc}{RESET}")

    print(f"\n{BOLD}{GREEN}Replay complete — same failure reproduced offline.{RESET}")
    print(f"Fixture path: {fixture_db}")
    print()


if __name__ == "__main__":
    main()
