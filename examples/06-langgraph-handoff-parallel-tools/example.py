"""
Multi-agent handoff + parallel-tool-call race — issue #5277.

Reproduces the exact failure class behind
https://github.com/langchain-ai/langgraph/issues/5277: a `flight_assistant`
agent returns two parallel tool calls in one turn — `book_flight` (a normal
tool) and `transfer_to_hotel_assistant` (a handoff tool returning
`Command(graph=Command.PARENT, ...)`, LangGraph's own documented multi-agent
handoff pattern). The handoff's parent-graph jump reaches `hotel_assistant`
before `book_flight`'s ToolMessage has been merged into shared state, so
`hotel_assistant`'s own `call_model` step builds a message history with an
`AIMessage` whose `book_flight` tool call has no matching `ToolMessage`.
LangGraph's own `_validate_chat_history()` rejects it locally:

    ValueError: Found AIMessages with tool_calls that do not have a
    corresponding ToolMessage. [...] (ErrorCode.INVALID_CHAT_HISTORY)

No API key required — both "models" are plain Python stand-ins (a real
OpenAI call was never needed to hit this bug either; the crash is 100% free
and deterministic, confirmed by the original investigation of #5277).

What this example demonstrates about agent-trace's capture:

1. Zero HTTP exchanges are recorded before the crash — the failure happens
   entirely client-side, so `src/agent_trace/interceptor/httpx_hook.py` (the
   HTTP interceptor) has nothing to see here. The evidence comes entirely
   from `LangGraphTracer`'s callback-layer capture
   (`src/agent_trace/integrations/langgraph.py`), not HTTP replay.
2. Every `ParentCommand` control-flow signal raised by the handoff jump
   (LangGraph's own internal mechanism, not an application error) closes its
   span OK with `langgraph.handoff=true` — not ERROR — so it doesn't drown
   out the real failure (the fix for the exact "3 spans mislabeled ERROR for
   a successful handoff" issue this investigation originally found).
3. Every span carrying the genuine `ValueError` gets tagged
   `error.origin=application` and `error.known_pattern=
   langgraph_invalid_chat_history` automatically — turning "read the raw
   trace.json and recognize the pattern yourself" (what the reporter did
   unaided in the GitHub thread) into a one-line classification already on
   the span.

Run:
    python examples/06-langgraph-handoff-parallel-tools/example.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import create_react_agent
    from langgraph.types import Command
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-trace[langgraph]")

from agent_trace import Tracer
from agent_trace._replay.fixture import Fixture
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.langgraph import LangGraphTracer

# create_react_agent emits a deprecation warning on current LangGraph
# versions (moved to langchain.agents) — irrelevant noise for this example.
warnings.filterwarnings("ignore", message=".*create_react_agent.*")

# ---------------------------------------------------------------------------
# Fake chat models — no API key, fully deterministic
# ---------------------------------------------------------------------------


class FlightFakeChatModel(BaseChatModel):
    """Always returns the reporter's exact parallel-tool-call shape: a
    normal tool call (book_flight) alongside a handoff tool call
    (transfer_to_hotel_assistant) in one AIMessage."""

    @property
    def _llm_type(self) -> str:
        return "fake-flight"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FlightFakeChatModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "book_flight",
                    "args": {"destination": "SFO"},
                    "id": "call_book_flight_1",
                },
                {
                    "name": "transfer_to_hotel_assistant",
                    "args": {},
                    "id": "call_transfer_1",
                },
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class HotelFakeChatModel(BaseChatModel):
    """The receiving agent's model — never actually reached with a valid
    history in this reproduction; create_react_agent's own
    _validate_chat_history() rejects the incoming messages first."""

    @property
    def _llm_type(self) -> str:
        return "fake-hotel"

    def bind_tools(self, tools: Any, **kwargs: Any) -> HotelFakeChatModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        reply = AIMessage(content="Hotel booked.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


# ---------------------------------------------------------------------------
# flight_assistant — hand-rolled (not create_react_agent) for full control
# over the exact race shape: book_flight's result merges into THIS
# subgraph's state while the handoff's Command(graph=Command.PARENT, ...)
# carries the full accumulated trajectory (the tool-calling AIMessage) plus
# only its own ToolMessage — LangGraph's own documented multi-agent handoff
# pattern (see the "Multi-agent" how-to in LangGraph's docs).
# ---------------------------------------------------------------------------


def flight_call_model(state: MessagesState) -> dict[str, Any]:
    response = FlightFakeChatModel().invoke(state["messages"])
    return {"messages": [response]}


def flight_tools(state: MessagesState) -> list[Any]:
    last = state["messages"][-1]
    book_call = next(tc for tc in last.tool_calls if tc["name"] == "book_flight")
    transfer_call = next(
        tc for tc in last.tool_calls if tc["name"] == "transfer_to_hotel_assistant"
    )
    book_result = ToolMessage(
        content="Flight booked to SFO", tool_call_id=book_call["id"]
    )
    transfer_result = ToolMessage(
        content="Transferred to hotel_assistant", tool_call_id=transfer_call["id"]
    )
    return [
        # book_flight's result updates *this* (flight_assistant) subgraph's
        # own state...
        {"messages": [book_result]},
        # ...while the handoff jumps straight to the parent graph, carrying
        # the full trajectory so far (the tool-calling AIMessage) plus only
        # ITS OWN ToolMessage. book_flight's ToolMessage above is a separate,
        # parallel write that hasn't landed in this update.
        Command(
            graph=Command.PARENT,
            goto="hotel_assistant",
            update={"messages": state["messages"] + [transfer_result]},
        ),
    ]


def build_graph() -> Any:
    flight_builder = StateGraph(MessagesState)
    flight_builder.add_node("call_model", flight_call_model)
    flight_builder.add_node("tools", flight_tools)
    flight_builder.add_edge(START, "call_model")
    flight_builder.add_edge("call_model", "tools")
    flight_graph = flight_builder.compile()

    hotel_assistant = create_react_agent(
        model=HotelFakeChatModel(), tools=[], name="hotel_assistant"
    )

    parent_builder = StateGraph(MessagesState)
    parent_builder.add_node("flight_assistant", flight_graph)
    parent_builder.add_node("hotel_assistant", hotel_assistant)
    parent_builder.add_edge(START, "flight_assistant")
    parent_builder.add_edge("hotel_assistant", END)
    return parent_builder.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main() -> None:
    graph = build_graph()

    t = Tracer(trace_dir=Path.home() / ".agent-trace" / "runs")
    with t.start_trace("langgraph-handoff-parallel-tools-race", record=True) as trace:
        cb = LangGraphTracer(tracer=t, trace=trace)
        try:
            graph.invoke(
                {"messages": [HumanMessage(content="Book me a flight and a hotel")]},
                config={"callbacks": [cb]},
            )
            print("(no crash — unexpected; the race did not manifest this run)")
        except ValueError as exc:
            print(f"Crashed as expected: {type(exc).__name__}: {str(exc)[:120]}...")

    run_dir = Path.home() / ".agent-trace" / "runs" / trace.run_id
    with Fixture(run_dir / "fixture.db") as fixture:
        exchange_count = fixture.exchange_count()

    print(f"\nHTTP exchanges recorded before the crash: {exchange_count}")
    print("(zero — the failure is entirely client-side; nothing ever reached the wire)")

    print("\n--- Span tree ---")
    StdoutExporter().export(trace)

    handoff_spans = [
        s for s in trace.spans if s.attributes.get("langgraph.handoff") is True
    ]
    error_spans = [s for s in trace.spans if s.status.value == "ERROR"]

    print(
        f"\n{len(handoff_spans)} span(s) correctly closed OK for the "
        "ParentCommand handoff:"
    )
    for s in handoff_spans:
        signal = s.attributes.get("langgraph.control_flow_signal")
        print(f"  {s.name}  (langgraph.control_flow_signal={signal})")

    print(f"\n{len(error_spans)} span(s) closed ERROR for the genuine failure:")
    for s in error_spans:
        origin = s.attributes.get("error.origin", "?")
        pattern = s.attributes.get("error.known_pattern", "")
        print(f"  {s.name}  origin={origin}  known_pattern={pattern}")

    print(f"\nRun ID: {trace.run_id}")
    print(f"Trace saved to: {run_dir / 'trace.json'}")


if __name__ == "__main__":
    main()
