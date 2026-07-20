"""
`make_graph()`-style entry point for `langgraph dev`/LangGraph Studio —
example 09. See ../../docs/integrations/langgraph.md, section 10, and this
directory's README.md for the full explanation.

`langgraph.json` (next to this file) points LangGraph Platform's
`graphs` config at ``graph:make_graph``. When you run `langgraph dev` (or
LangGraph Studio serves this directory), `langgraph_api` imports this module
and calls `make_graph()` exactly **once**, at server startup — before any
`.invoke()`/`.ainvoke()` call exists anywhere. Every subsequent run served
by that process reuses the same compiled graph object.

That inverted control flow (the framework owns the entry point, not you) is
exactly why issue #4798's construction-phase failures (MCP client setup,
tool loading, config parsing — all of which normally happen inside a
`make_graph()`-style factory) were invisible to `LangGraphTracer`, which
only ever attaches at invoke()/stream() time: there is no `graph.invoke()`
call anywhere for a developer to wrap in `with tracer.start_trace(...)`.

Two mechanisms close that gap, both demonstrated here:

1. ``@instrument_graph_factory(tracer)`` on `make_graph()` itself —
   activates recording for the duration of the factory call, capturing
   construction-phase HTTP traffic (the simulated MCP tool-listing call
   below) either way. Whether that's a brand-new scoped trace or a span
   nested under an already-active one depends on whether recording was
   already active when `make_graph()` was called — see the module
   docstring in `agent_trace.integrations.langgraph.instrument_graph_factory`.
2. Binding a `LangGraphTracer` callback onto the *compiled* graph itself via
   ``graph.with_config(callbacks=[...])`` — so every future
   `.invoke()`/`.stream()` call is traced automatically, regardless of who
   calls it (`langgraph dev`'s own request-handling code, not you). This
   only produces real span data if a trace is active for the life of the
   process — i.e. only when `AGENT_TRACE_AUTO_RECORD=1` was set (directly,
   or via `agent-trace run -- langgraph dev`) *before* this module was
   imported, so `tracer.active_trace` is already the persistent
   auto-record trace by the time `make_graph()` runs. Without that,
   `make_graph()` still returns a working graph — it just has no LangGraph
   callback attached, exactly like today's status quo.
"""

from __future__ import annotations

import sys
from typing import Any, TypedDict

import httpx

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import tracer
from agent_trace.integrations.langgraph import (
    LangGraphTracer,
    instrument_graph_factory,
)


class State(TypedDict):
    messages: list[str]


def _load_tools_from_mcp_server() -> list[str]:
    """Stand-in for an MCP `streamable_http`/SSE tool-listing call typically
    made during graph construction (a real replacement would be something
    like ``MultiServerMCPClient(...).get_tools()``) — issue #4798's exact
    failure class. Self-contained (a local ``httpx.MockTransport``, no real
    network or MCP server needed) so this example is deterministic and
    needs no external services, while still exercising the real
    RecordingTransport code path: when recording is active, this HTTP call
    is captured exactly like a real MCP tool-listing request would be.
    """

    def _fake_mcp_server(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tools": ["search", "book_flight"]})

    client = httpx.Client(transport=httpx.MockTransport(_fake_mcp_server))
    with client:
        response = client.get("https://mcp.example.com/tools")
    return list(response.json()["tools"])


def _build_call_model_node(tools: list[str]) -> Any:
    """Returns a node function closing over the tools loaded at
    construction time — a real node here would call a chat model using
    those tools; this is a plain Python stand-in so the example needs no
    API key."""

    def _call_model(state: State) -> State:
        tool_names = ", ".join(tools)
        reply = f"(using tools: {tool_names}) ok, done."
        return {"messages": [*state["messages"], reply]}

    return _call_model


@instrument_graph_factory(tracer)
def make_graph(config: dict[str, Any] | None = None) -> Any:
    """The construction-phase entry point `langgraph.json` points at.

    Called once, by `langgraph dev`/LangGraph Studio, at server startup.
    """
    tools = _load_tools_from_mcp_server()  # captured when recording is active

    builder = StateGraph(State)
    builder.add_node("call_model", _build_call_model_node(tools))
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    graph = builder.compile()

    # Attach LangGraphTracer to every future invoke()/stream() call on this
    # compiled graph, regardless of who calls it — only meaningful when a
    # trace is already active (AGENT_TRACE_AUTO_RECORD=1 was set before this
    # module was imported); a no-op-safe skip otherwise.
    if tracer.active_trace is not None:
        graph = graph.with_config(
            callbacks=[LangGraphTracer(tracer=tracer, trace=tracer.active_trace)]
        )

    # Stash the loaded tool names so simulate_dev_server.py can show they
    # really did make it through graph construction.
    graph._agent_trace_example_tools_loaded = tools  # type: ignore[attr-defined]
    return graph
