"""
create_react_agent + a non-OpenAI provider's finish_reason anomaly (#6574).

OpenAI's `finish_reason` vocabulary (`stop`, `tool_calls`, `length`, ...) is
well understood, but other providers use their own conventions — Gemini in
particular can return `finish_reason="MALFORMED_FUNCTION_CALL"` alongside an
AIMessage that still *looks* like a normal tool call: the model proposed
calling a function, but Gemini's own validation flagged the generated
arguments as malformed. A developer who only checks
`response.tool_calls` (as most ReAct-agent loops do) never sees this — the
tool call gets dispatched exactly as if it were a clean, validated one.

This example builds a `create_react_agent` graph against a fake
provider (`FakeGeminiChatModel`, no API key/network needed) that
reproduces this exact anomaly on its first turn, and shows what
`LangGraphTracer`'s `on_llm_end` capture (`_extract_finish_reason`/
`_record_llm_end_data` in `src/agent_trace/integrations/langgraph.py`)
does with it: `llm.finish_reason` and `llm.has_tool_calls` land on the LLM
span together, so a developer inspecting the trace — or running
`agent-trace inspect <run_id>` — can see that a *dispatched* tool call
co-occurred with a non-`stop`/non-`tool_calls` finish reason, the signal
this issue asked for.

Run:
    python examples/11-langgraph-react-agent-non-openai-finish-reason/example.py

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
    from langgraph.prebuilt import create_react_agent
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.langgraph import LangGraphTracer

TRACE_DIR = Path.home() / ".agent-trace" / "runs"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


class FakeGeminiChatModel(BaseChatModel):
    """Stand-in for `ChatGoogleGenerativeAI` (no API key/network needed).

    First turn: emits a tool call but sets
    `generation_info.finish_reason="MALFORMED_FUNCTION_CALL"` — Gemini's
    real, documented anomaly where the proposed function-call arguments
    failed the provider's own validation, yet LangChain still surfaces a
    normal-looking `AIMessage` with `tool_calls` populated.

    Second turn (after the tool result comes back): finishes cleanly with
    `finish_reason="STOP"`, for contrast.
    """

    call_count: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.call_count += 1
        if self.call_count == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "Boston"},
                        "id": "call_1",
                    }
                ],
            )
            generation_info = {"finish_reason": "MALFORMED_FUNCTION_CALL"}
        else:
            msg = AIMessage(content="The weather in Boston is sunny.")
            generation_info = {"finish_reason": "STOP"}
        return ChatResult(
            generations=[ChatGeneration(message=msg, generation_info=generation_info)]
        )

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002 — required by create_react_agent
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-gemini-chat-model"


def main() -> None:
    model = FakeGeminiChatModel()
    graph = create_react_agent(model, tools=[get_weather])

    t = Tracer(trace_dir=TRACE_DIR)
    with t.start_trace("react-agent-non-openai-finish-reason") as trace:
        cb = LangGraphTracer(tracer=t, trace=trace)
        graph.invoke(
            {"messages": [("user", "what's the weather in Boston?")]},
            config={"callbacks": [cb]},
        )

    print("--- Span tree ---")
    StdoutExporter().export(trace)

    print("\n--- LLM spans: finish_reason + has_tool_calls ---")
    for span in trace.spans:
        if span.name.startswith("llm:"):
            finish_reason = span.attributes.get("llm.finish_reason")
            has_tool_calls = span.attributes.get("llm.has_tool_calls")
            flag = (
                "  <-- tool call dispatched despite a non-stop/non-tool_calls "
                "finish_reason"
                if has_tool_calls and finish_reason not in ("stop", "tool_calls", "STOP")
                else ""
            )
            print(f"  {span.name}: finish_reason={finish_reason!r} "
                  f"has_tool_calls={has_tool_calls}{flag}")

    print(f"\nRun ID: {trace.run_id}")
    print(
        "\nRun `agent-trace inspect "
        f"{trace.run_id}` to see this alongside every other pattern check — "
        "though the anomaly here is on the *span* (LangChain callback data), "
        "not the raw HTTP body, since FakeGeminiChatModel makes no HTTP "
        "call; a real ChatGoogleGenerativeAI run would additionally have "
        "the raw response captured in fixture.db."
    )


if __name__ == "__main__":
    main()
