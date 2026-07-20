"""
graph.invoke(stream_mode=...) vs graph.stream(...) delivery timing — issue #4653.

Reproduces the exact distinction behind
https://github.com/langchain-ai/langgraph/issues/4653 (and its "same issue"
commenters JasonChen280/Kigstn): ``graph.invoke(state,
stream_mode="messages")`` fully collects the entire run's output internally
before returning ANYTHING to the caller — LangGraph's own
``Pregel.invoke()`` drains its own ``self.stream(...)`` generator completely
in a loop before returning the final value — while ``graph.stream(state,
stream_mode="messages")`` yields each chunk to the caller progressively, as
soon as the corresponding node finishes. A tool that prints/reacts the
moment a node completes (e.g. a UI showing "flight booked!" as soon as
that step is done) behaves completely differently depending on which one is
used, even though every span in a callback-only trace looks identical
either way — the callback layer alone cannot see this delivery-timing
difference, which is exactly the stream-yield timestamp capture this
example also demonstrates.

No API key required — both nodes are plain Python stand-ins with a small,
deliberate `time.sleep()` so the timing difference is observable without
needing a slow LLM call to make it visible.

What this example demonstrates about agent-trace's capture:

1. `graph.invoke(...)` produces a normal callback-driven trace (node/chain
   spans) but NO `graph:stream` span at all — there is nothing in a
   callback-only trace that could ever show *when* the caller actually
   received output, because invoke() doesn't hand anything back until the
   whole run is done.
2. Wrapping `graph.stream(...)` in `traced_stream()` (the fix implemented
   in `src/agent_trace/integrations/langgraph.py`) opens a dedicated
   `graph:stream` span carrying one `stream_yield` SpanEvent per chunk,
   timestamped on the same clock as every other span, at the *exact* moment
   each chunk reaches the caller's own `for` loop — not when the underlying
   node finished internally.
3. Printing a running wall-clock timestamp inline, next to each printed
   chunk, makes the two delivery modes visibly different: invoke()'s
   "chunks" (there's only one — the final return value) all arrive at once,
   after both nodes have already finished; stream()'s chunks arrive
   one-at-a-time, each right after its node completes.

Run:
    python examples/07-langgraph-invoke-vs-stream-timing/example.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace.integrations.langgraph import traced_stream

_NODE_DELAY_S = 0.3


class State(TypedDict):
    steps: list[str]


def slow_step_one(state: State) -> State:
    time.sleep(_NODE_DELAY_S)
    return {"steps": [*state["steps"], "step_one_done"]}


def slow_step_two(state: State) -> State:
    time.sleep(_NODE_DELAY_S)
    return {"steps": [*state["steps"], "step_two_done"]}


def build_graph() -> Any:
    builder = StateGraph(State)
    builder.add_node("step_one", slow_step_one)
    builder.add_node("step_two", slow_step_two)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    return builder.compile()


def run_invoke(graph: Any) -> None:
    print("\n--- graph.invoke(stream_mode='updates') ---")
    start = time.monotonic()
    result = graph.invoke({"steps": []}, stream_mode="updates")
    elapsed = time.monotonic() - start
    print(
        f"  [{elapsed:6.3f}s] caller received the ENTIRE result at once, "
        f"after both nodes had already finished: {result}"
    )
    print(
        "  (nothing was observable to the caller before this single "
        "moment — invoke() blocks until the whole run completes)"
    )


def run_stream(graph: Any, tracer: Tracer) -> None:
    print("\n--- graph.stream(stream_mode='updates'), wrapped in traced_stream() ---")
    start = time.monotonic()
    raw_stream = graph.stream({"steps": []}, stream_mode="updates")
    for chunk in traced_stream(tracer, raw_stream):
        elapsed = time.monotonic() - start
        print(f"  [{elapsed:6.3f}s] caller received a chunk progressively: {chunk}")


def main() -> None:
    graph = build_graph()
    tracer = Tracer(trace_dir=Path.home() / ".agent-trace" / "runs")

    with tracer.start_trace("invoke-mode") as invoke_trace:
        run_invoke(graph)

    with tracer.start_trace("stream-mode") as stream_trace:
        run_stream(graph, tracer)

    print("\n--- What each trace's span tree shows ---")
    invoke_stream_spans = [s for s in invoke_trace.spans if s.name == "graph:stream"]
    stream_stream_spans = [s for s in stream_trace.spans if s.name == "graph:stream"]
    print(
        f"invoke-mode trace:  {len(invoke_stream_spans)} 'graph:stream' span(s) "
        "(none — invoke() never yields progressively, so there's nothing for "
        "traced_stream() to wrap)"
    )
    print(
        f"stream-mode trace:  {len(stream_stream_spans)} 'graph:stream' span(s), "
        f"carrying {len(stream_stream_spans[0].events) if stream_stream_spans else 0} "
        "stream_yield event(s) — one per chunk, timestamped at the moment the "
        "caller's own for-loop actually received it"
    )

    if stream_stream_spans:
        print("\nstream_yield event timestamps (relative to span start, seconds):")
        span = stream_stream_spans[0]
        for event in span.events:
            if event.name == "stream_yield":
                rel = event.timestamp - span.start_time
                print(f"  index={event.attributes['stream.index']}  +{rel:.3f}s")

    print(f"\nRun IDs: invoke={invoke_trace.run_id}  stream={stream_trace.run_id}")


if __name__ == "__main__":
    main()
