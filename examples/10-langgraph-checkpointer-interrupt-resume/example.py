"""
Checkpointer + interrupt + resume — capture, and the #4217 diagnostic.

This example fills the gap flagged against both #531 and #4217: every other
example under ``examples/`` is either linear/non-checkpointed
(``02-langgraph-failure-replay``) or exercises checkpointer *writes* without
an interrupt/resume step (``05-parallel-command-parent-routing``). None of
them compile a graph with a real checkpointer, pause it mid-run via
``interrupt_before``, and resume it — the exact failure class behind #531
(a dangling ``tool_call_id`` surviving a checkpointer resume) and #4217
(a silent no-op resume caused by an external ``update_state()`` call that
omits ``as_node``).

It's split into two parts:

Part A — the happy path (record + resume), showing what agent-trace
captures across an interrupt/resume boundary: `LangGraphTracer` for the
node/LLM/tool span tree, `TracingCheckpointSaver` for the checkpointer
serde-boundary spans, and `traced_update_state()` recording the pregel
scheduler's post-write task list.

Part B — the #4217 bug, reproduced against a real, currently-installed
LangGraph via a two-tool ``create_react_agent``-shaped graph (no API key —
every "LLM decision" is a plain Python stand-in, exactly like
``05-parallel-command-parent-routing``): calling ``update_state()`` to
inject a `ToolMessage` *without* ``as_node`` leaves the pregel scheduler's
``next`` task list empty (`checkpoint.zero_tasks_scheduled=True`) and the
resumed run silently does nothing further — vs. supplying
``as_node="tools"``, which schedules the `agent` node correctly and the run
continues. `traced_update_state()` (`agent_trace.integrations.
langgraph_checkpoint`) captures this distinction on a
``checkpoint:update_state`` span; `agent-trace show`/`agent-trace replay`
surface it as an explicit "Zero tasks scheduled after a state update"
warning (`src/agent_trace/_cli.py::_print_zero_task_updates`) instead of the
silent, evidence-free "nothing happened" #4217's reporter originally hit.

Run:
    python examples/10-langgraph-checkpointer-interrupt-resume/example.py

No API key required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.types import Command, interrupt
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace._cli import _print_zero_task_updates
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.langgraph import LangGraphTracer
from agent_trace.integrations.langgraph_checkpoint import (
    TracingCheckpointSaver,
    traced_update_state,
)

TRACE_DIR = Path.home() / ".agent-trace" / "runs"


# ---------------------------------------------------------------------------
# Part A — interrupt + resume, happy path
# ---------------------------------------------------------------------------


class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    approved: bool


def propose_action(state: ApprovalState) -> ApprovalState:
    """Stand-in for an LLM proposing a sensitive action. Pauses for a human
    decision via interrupt() before the graph is allowed to continue —
    the same checkpointer/interrupt shape #531's original report hit."""
    decision = interrupt(
        {"question": "Approve sending the report to the customer?"}
    )
    return {"approved": bool(decision), "messages": []}


def send_report(state: ApprovalState) -> ApprovalState:
    return {"messages": [AIMessage(content="Report sent.")]}


def build_approval_graph(checkpointer: object):
    builder = StateGraph(ApprovalState)
    builder.add_node("propose", propose_action)
    builder.add_node("send", send_report)
    builder.add_edge(START, "propose")
    builder.add_edge("propose", "send")
    builder.add_edge("send", END)
    return builder.compile(checkpointer=checkpointer)


def run_part_a() -> None:
    print("=" * 78)
    print("Part A — checkpointer + interrupt() + resume (happy path)")
    print("=" * 78)

    t = Tracer(trace_dir=TRACE_DIR)
    checkpointer = TracingCheckpointSaver(InMemorySaver(), tracer=t)
    graph = build_approval_graph(checkpointer)

    with t.start_trace("checkpointer-interrupt-resume") as trace:
        cb = LangGraphTracer(tracer=t, trace=trace)
        config = {"configurable": {"thread_id": "approval-1"}, "callbacks": [cb]}

        result = graph.invoke(
            {"messages": [HumanMessage(content="Ready to send?")], "approved": False},
            config=config,
        )
        print(f"\nAfter first invoke — interrupted: {'__interrupt__' in result}")
        print(f"Pending next node(s): {graph.get_state(config).next}")

        # A human approves out-of-band; resume the graph from the checkpoint.
        result = graph.invoke(Command(resume=True), config=config)
        print(f"\nAfter resume — final messages: "
              f"{[m.content for m in result['messages']]}")

    print("\n--- Span tree ---")
    StdoutExporter().export(trace)
    print(f"\nRun ID: {trace.run_id}")
    print(
        "Notice the 'checkpoint:serde:*' spans (from TracingCheckpointSaver) "
        "recording every dumps_typed/loads_typed call across the interrupt/"
        "resume boundary, and the 'node:propose'/'node:send' spans showing "
        "langgraph.interrupted=true on the paused run."
    )
    return trace.run_id


# ---------------------------------------------------------------------------
# Part B — the #4217 bug: update_state() without as_node silently drops the
# resume. Reproduced against the currently-installed LangGraph with a
# two-tool create_react_agent-shaped graph (fake "LLM" node, no API key).
# ---------------------------------------------------------------------------


@tool
def get_data_externally() -> str:
    """Fetch external data (stand-in for a real API call)."""
    return "external-data-123"


@tool
def process_data(data: str) -> str:
    """Process previously fetched data."""
    return f"processed:{data}"


def build_react_shaped_graph(checkpointer: object):
    """Mirrors the exact structure behind #4217: an `agent` node that emits
    tool_calls, a `tools` ToolNode, and interrupt_before=["tools"] so a
    caller can inspect/inject a tool result before it runs — the same shape
    `create_react_agent(..., interrupt_before=["tools"])` produces."""
    call_count = {"n": 0}

    def agent_node(state: dict) -> dict:
        call_count["n"] += 1
        msgs = state["messages"]
        if call_count["n"] == 1:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_data_externally", "args": {}, "id": "call_1"}
                        ],
                    )
                ]
            }
        last = msgs[-1]
        if isinstance(last, ToolMessage) and last.name == "get_data_externally":
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "process_data",
                                "args": {"data": last.content},
                                "id": "call_2",
                            }
                        ],
                    )
                ]
            }
        return {"messages": [AIMessage(content="done")]}

    class S(TypedDict):
        messages: Annotated[list, add_messages]

    builder = StateGraph(S)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode([get_data_externally, process_data]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])


def run_part_b() -> None:
    print()
    print("=" * 78)
    print("Part B — the #4217 bug: update_state() without as_node")
    print("=" * 78)

    t = Tracer(trace_dir=TRACE_DIR)

    # --- Broken path: as_node omitted -----------------------------------
    checkpointer_broken = InMemorySaver()
    graph_broken = build_react_shaped_graph(checkpointer_broken)
    config_broken = {"configurable": {"thread_id": "broken-resume"}}

    with t.start_trace("update-state-without-as-node") as trace_broken:
        cb = LangGraphTracer(tracer=t, trace=trace_broken)
        config_broken["callbacks"] = [cb]
        graph_broken.invoke(
            {"messages": [HumanMessage(content="go")]}, config=config_broken
        )

        tool_msg = ToolMessage(
            content="external-data-123",
            tool_call_id="call_1",
            name="get_data_externally",
        )
        # The exact call from #4217's report: no as_node supplied.
        new_config = traced_update_state(
            t, graph_broken, config_broken, {"messages": [tool_msg]}
        )
        print(
            f"\n[broken] next tasks after update_state (no as_node): "
            f"{graph_broken.get_state(new_config).next}"
        )
        result = graph_broken.invoke(None, config=config_broken)
        print(
            f"[broken] resume produced "
            f"{len(result['messages'])} message(s) — no further LLM/tool "
            f"call happened (the silent no-op #4217 hit)."
        )

    # --- Fixed path: as_node="tools" supplied -----------------------------
    checkpointer_fixed = InMemorySaver()
    graph_fixed = build_react_shaped_graph(checkpointer_fixed)
    config_fixed = {"configurable": {"thread_id": "fixed-resume"}}

    with t.start_trace("update-state-with-as-node") as trace_fixed:
        cb = LangGraphTracer(tracer=t, trace=trace_fixed)
        config_fixed["callbacks"] = [cb]
        graph_fixed.invoke(
            {"messages": [HumanMessage(content="go")]}, config=config_fixed
        )

        tool_msg = ToolMessage(
            content="external-data-123",
            tool_call_id="call_1",
            name="get_data_externally",
        )
        new_config = traced_update_state(
            t, graph_fixed, config_fixed, {"messages": [tool_msg]}, as_node="tools"
        )
        print(
            f"\n[fixed] next tasks after update_state (as_node='tools'): "
            f"{graph_fixed.get_state(new_config).next}"
        )
        result = graph_fixed.invoke(None, config=config_fixed)
        print(
            f"[fixed] resume produced {len(result['messages'])} message(s) "
            f"— the agent re-invoked and called process_data as expected."
        )

    print(f"\nBroken run ID: {trace_broken.run_id}")
    print(f"Fixed run ID:  {trace_fixed.run_id}")

    print()
    print("--- agent-trace's automated #4217 diagnostic ---")
    print(f"$ agent-trace show {trace_broken.run_id}")
    _print_zero_task_updates(trace_broken.to_dict()["spans"])
    print(
        f"\n$ agent-trace show {trace_fixed.run_id}  (nothing printed — no "
        "zero-task update on the fixed run)"
    )
    _print_zero_task_updates(trace_fixed.to_dict()["spans"])
    print(
        "\nThe warning fires only for the broken run's checkpoint:"
        "update_state span — the exact automated signal #4217's reporter "
        "had to get from a LangGraph maintainer by hand (see "
        "src/agent_trace/_cli.py::_print_zero_task_updates, wired into both "
        "`agent-trace show <run_id>` and `agent-trace replay <run_id>`)."
    )


def main() -> None:
    run_part_a()
    run_part_b()


if __name__ == "__main__":
    main()
