"""
A tool's own logic raising an exception — not an HTTP-level failure (#30708).

`examples/02-langgraph-failure-replay` only exercises HTTP-level failures
(`response.raise_for_status()`) and one generic `RuntimeError`. No existing
example shows a *tool's own code* raising and agent-trace capturing it —
exactly the gap behind issue #30708: a `create_react_agent` tool raising
`ZeroDivisionError` with no visible `on_tool_error` event in the installed
`langgraph`/`langchain-core` version's `astream_events` output.

This example builds a `create_react_agent` graph with a `divide` tool and a
fake model that always calls it with `b=0`, using `ToolNode(...,
handle_tool_errors=False)` so the exception actually propagates instead of
being swallowed into a `ToolMessage` (LangGraph's default
`handle_tool_errors=True` behavior) — reproducing #30708's exact "the tool
raised, and nothing downstream told me" shape.

It shows `LangGraphTracer.on_tool_error` (`src/agent_trace/integrations/
langgraph.py`) closing the tool's span `ERROR` with the exception captured
via `Span.record_exception` (`src/agent_trace/core/span.py`) — independent
of whatever event types the installed `langchain-core`/`langgraph`
version's own `astream_events` implementation happens to support, since
this capture path is agent-trace's own callback handler, not a consumer of
`astream_events`.

Run:
    python examples/15-langgraph-tool-execution-error/example.py

No API key required.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode, create_react_agent
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace._cli import _print_errors_only
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.langgraph import LangGraphTracer

TRACE_DIR = Path.home() / ".agent-trace" / "runs"


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b  # raises ZeroDivisionError when b == 0 — #30708's exact bug


class FakeChatModel(BaseChatModel):
    """Stand-in for a real provider (no API key/network needed). Always
    proposes calling divide(10, 0), triggering the tool's own
    ZeroDivisionError."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "divide", "args": {"a": 10, "b": 0}, "id": "call_1"}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


def main() -> None:
    model = FakeChatModel()
    # handle_tool_errors=False: don't swallow the exception into a
    # ToolMessage (LangGraph's default) — let it actually propagate, the
    # shape #30708's reporter hit.
    tool_node = ToolNode([divide], handle_tool_errors=False)
    graph = create_react_agent(model, tools=tool_node)

    t = Tracer(trace_dir=TRACE_DIR)
    with t.start_trace("tool-execution-error") as trace:
        cb = LangGraphTracer(tracer=t, trace=trace)
        try:
            graph.invoke(
                {"messages": [("user", "divide 10 by 0")]}, config={"callbacks": [cb]}
            )
        except ZeroDivisionError as exc:
            print(f"Graph raised (expected): {type(exc).__name__}: {exc}\n")

    print("--- What LangGraphTracer.on_tool_error captured ---")
    _print_errors_only(trace.to_dict()["spans"])

    print("\n--- Full span tree ---")
    StdoutExporter().export(trace)

    tool_span = next(s for s in trace.spans if s.name == "tool:divide")
    print(f"\ntool:divide span status: {tool_span.status.value}")
    print(f"Run ID: {trace.run_id}")
    print(
        "\nThis capture happened via LangGraphTracer's own on_tool_error "
        "callback — independent of whatever event types the installed "
        "langchain-core/langgraph version's astream_events implementation "
        "supports."
    )


if __name__ == "__main__":
    main()
