"""
Parallel-tool Command.PARENT routing — capture (and gap).

This example demonstrates the exact failure class behind issue #7129:
parallel tool calls that each return ``Command(graph=Command.PARENT,
update={...})`` to hand a value up to the parent graph. It's split into two
parts:

Part A — the capture gap (a real, live graph run, no API key needed):
    Three tool nodes are fanned out in parallel via `Send`, each returning
    a `Command(graph=Command.PARENT, ...)`. `LangGraphTracer`'s tool spans
    (`on_tool_start`/`on_tool_end`, agent_trace/integrations/langgraph.py)
    capture only name + OK status for each — the actual Command *payload*
    (what was proposed, and to which channel) is invisible in the span
    tree. This part uses three distinct channels (no LangGraph-level
    conflict) precisely so the run completes cleanly and the *capture* gap
    — not a crash — is what's on display.

Part B — the new diagnostic (agent_trace.integrations.langgraph_state_diff):
    Simulates the exact silent-drop shape #7129 hit — three parallel tasks
    proposing a write to the *same* channel in one superstep, where only one
    survives — via direct checkpointer calls (`put_writes`/`put`), the same
    calls Pregel's own scheduler makes internally regardless of which
    LangGraph mechanism (parallel node writes, Command.PARENT routing, ...)
    produced them. Reproducing the exact multi-agent/version-specific
    trigger conditions for #7129 inside a minimal, no-API-key example isn't
    reliable across LangGraph versions (a live run of the natural "two
    parallel writes to one plain channel" shape in the currently installed
    LangGraph version raises a loud `InvalidUpdateError` rather than
    silently dropping — see the module docstring in langgraph_state_diff.py
    for why the checkpointer boundary is still the right place to watch
    regardless) — so Part B demonstrates the diagnostic's *mechanism*
    directly against the real checkpointer contract instead.

Run:
    python examples/05-parallel-command-parent-routing/example.py

No API key required — every "model" decision here is a plain Python stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, Send
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.langgraph import LangGraphTracer
from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

# ---------------------------------------------------------------------------
# Part A — capture gap: a real, live parallel Command.PARENT graph run
# ---------------------------------------------------------------------------


class ParentState(TypedDict, total=False):
    result_alpha: str
    result_beta: str
    result_gamma: str


class ChildState(TypedDict):
    tool_name: str


def run_tool(state: ChildState) -> Command:
    """Every parallel tool call hands its result up to the parent graph via
    Command(graph=Command.PARENT, ...) — the exact mechanism #7129's three
    parallel tool calls used. Each writes to its *own* channel here so the
    run completes cleanly (see the module docstring for why)."""
    name = state["tool_name"]
    return Command(
        graph=Command.PARENT, update={f"result_{name}": f"value-from-{name}"}
    )


def dispatch(state: ParentState) -> list[Send]:
    return [
        Send("run_tool", {"tool_name": name}) for name in ("alpha", "beta", "gamma")
    ]


def build_graph(checkpointer: object) -> object:
    child_builder = StateGraph(ChildState)
    child_builder.add_node("run_tool", run_tool)
    child_builder.set_entry_point("run_tool")
    child_graph = child_builder.compile()

    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("run_tool", child_graph)
    parent_builder.add_node("collect", lambda s: {})
    parent_builder.add_conditional_edges(START, dispatch, ["run_tool"])
    parent_builder.add_edge("run_tool", "collect")
    parent_builder.add_edge("collect", END)
    return parent_builder.compile(checkpointer=checkpointer)


def run_part_a() -> None:
    print("=" * 78)
    print("Part A — capture gap: 3 parallel Command.PARENT tool calls, live run")
    print("=" * 78)

    t = Tracer(trace_dir=Path.home() / ".agent-trace" / "runs")
    checkpointer = wrap_checkpointer(InMemorySaver(), tracer=t)
    graph = build_graph(checkpointer)

    with t.start_trace("parallel-command-parent-routing") as trace:
        cb = LangGraphTracer(tracer=t, trace=trace)
        config = {"configurable": {"thread_id": "demo-thread"}, "callbacks": [cb]}
        result = graph.invoke({}, config=config)

    print(f"\nFinal parent state: {result}")
    print("\n--- Span tree ---")
    StdoutExporter().export(trace)

    tool_spans = [s for s in trace.spans if s.name.startswith("node:run_tool")]
    print(f"\n{len(tool_spans)} tool-dispatch span(s) captured.")
    print(
        "Notice: each span shows only a name + OK status. The actual "
        "Command(graph=Command.PARENT, update={...}) payload each task "
        "proposed — the exact thing you'd need to see to root-cause a "
        "dropped update — is not on any span today."
    )
    print(f"\nRun ID: {trace.run_id}")


# ---------------------------------------------------------------------------
# Part B — the new diagnostic, exercised against the real checkpointer
# contract with the exact silent-drop shape from #7129
# ---------------------------------------------------------------------------


def run_part_b() -> None:
    print("\n" + "=" * 78)
    print("Part B — the new per-superstep state-merge diagnostic")
    print("=" * 78)

    t = Tracer(trace_dir=Path.home() / ".agent-trace" / "runs")
    inner = InMemorySaver()

    with t.start_trace("parallel-command-parent-merge-diagnostic") as trace:
        wrapped = wrap_checkpointer(inner, tracer=t)

        config = {
            "configurable": {
                "thread_id": "demo-drop-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "parent-checkpoint-0",
            }
        }

        # Three parallel tasks each propose a write to the SAME channel —
        # the exact shape behind #7129 (2 of 3 parallel Command.PARENT
        # updates silently discarded in one superstep).
        wrapped.put_writes(
            config, [("result", "value-from-alpha")], task_id="tool_call_alpha"
        )
        wrapped.put_writes(
            config, [("result", "value-from-beta")], task_id="tool_call_beta"
        )
        wrapped.put_writes(
            config, [("result", "value-from-gamma")], task_id="tool_call_gamma"
        )

        # Only "beta"'s value actually landed in the finalized checkpoint —
        # LangGraph's own internal merge kept one, dropped two, silently.
        checkpoint = {
            "id": "cp-after-superstep",
            "channel_values": {"result": "value-from-beta"},
        }
        wrapped.put(config, checkpoint, metadata={}, new_versions={})

    print("\n--- Span tree ---")
    StdoutExporter().export(trace)

    merge_spans = [s for s in trace.spans if s.name == "checkpoint:superstep_merge"]
    if not merge_spans:
        print("\n(no superstep_merge span — nothing was dropped)")
        return

    print(f"\n{len(merge_spans)} superstep-merge span(s) captured:")
    for span in merge_spans:
        for event in span.events:
            attrs = event.attributes
            print(
                f"  channel={attrs['channel']!r}  "
                f"proposed={attrs['proposed_count']}  "
                f"survived={attrs['survived_count']}  "
                f"dropped={attrs['dropped_count']}  "
                f"dropped_task_ids={attrs['dropped_task_ids']!r}"
            )
    print(
        "\nThis is exactly the fact a developer needs to root-cause #7129: "
        "3 proposed, only 1 survived, and which 2 task IDs lost their "
        "update — captured automatically instead of requiring the "
        "developer to already suspect LangGraph's internal Pregel/"
        "Command.PARENT merge semantics before they could even look for it."
    )


def main() -> None:
    run_part_a()
    run_part_b()


if __name__ == "__main__":
    main()
