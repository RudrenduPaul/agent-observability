"""
`create_react_agent(..., response_format=...)` + tools failure mode (#4940).

Combining structured output (`response_format=SomeSchema`) with tool calling
is a recurring failure class on Anthropic/Bedrock models: if the model's
final turn doesn't cleanly close out every pending `tool_use` block before
LangGraph asks it to emit the structured `response_format` payload, the
provider rejects the request with a 400 — e.g. Anthropic's real error text,
`"tool_use ids were found without tool_result blocks immediately after"`.

This example reproduces that shape end-to-end with a fake `BaseChatModel`
(no API key needed) that makes one real HTTP call — through a real
`httpx.Client`, patched by agent-trace's `RecordingTransport` exactly like a
live Anthropic call would be — to a mock transport that returns Anthropic's
actual 400 error body. It shows both halves of what this issue asked for:

1. The **raw fixture capture** of the 400 (`fixture.db`, reachable via
   `Fixture.all_exchanges()` or `agent-trace inspect <run_id>`).
2. What the **LangGraph integration span** shows for the same failure —
   which, since the HTTP-error-response-body-on-span fix shipped
   (`Span.record_exception`, `src/agent_trace/core/span.py`), is no longer
   a generic "400 Bad Request" one-liner: the actual Anthropic error body is
   attached directly to the ERROR span, visible in `agent-trace show`
   without ever touching `fixture.db` by hand.

Run:
    python examples/12-langgraph-structured-response-tool-conflict/example.py

No API key required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import httpx
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from pydantic import BaseModel
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace._cli import _print_errors_only
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.langgraph import LangGraphTracer

TRACE_DIR = Path.home() / ".agent-trace" / "runs"

# Anthropic's real error body for this exact failure class.
ANTHROPIC_400_BODY = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": (
            "messages.1: `tool_use` ids were found without `tool_result` "
            "blocks immediately after: toolu_01. Each `tool_use` block must "
            "have a corresponding `tool_result` block in the next message."
        ),
    },
}


class WeatherReport(BaseModel):
    city: str
    forecast: str


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


def _mock_anthropic_400(request: "httpx.Request") -> "httpx.Response":
    return httpx.Response(400, json=ANTHROPIC_400_BODY, request=request)


class FakeAnthropicChatModel(BaseChatModel):
    """Stand-in for `ChatAnthropic` (no API key/network needed).

    First turn: proposes a tool call, as normal. Second turn (the
    structured-`response_format` turn LangGraph triggers after the tool
    result comes back): makes a real HTTP call — through
    `httpx.Client`, patched by agent-trace's `RecordingTransport` the same
    way a live SDK call would be — to a mock transport returning
    Anthropic's actual 400 body for this failure class, then raises via
    `response.raise_for_status()`.
    """

    call_count: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.call_count += 1
        if self.call_count == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Boston"}, "id": "call_1"}
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        client = httpx.Client(transport=httpx.MockTransport(_mock_anthropic_400))
        response = client.post(
            "https://api.anthropic.com/v1/messages", json={"stand-in": "request"}
        )
        response.raise_for_status()  # pragma: no cover — always raises above
        return ChatResult(generations=[])  # pragma: no cover — unreachable

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-anthropic-chat-model"


def main() -> None:
    model = FakeAnthropicChatModel()
    graph = create_react_agent(model, tools=[get_weather], response_format=WeatherReport)

    t = Tracer(trace_dir=TRACE_DIR)
    with t.start_trace("structured-response-tool-conflict", record=True) as trace:
        cb = LangGraphTracer(tracer=t, trace=trace)
        try:
            graph.invoke(
                {"messages": [("user", "what's the weather in Boston?")]},
                config={"callbacks": [cb]},
            )
        except Exception as exc:
            print(f"Graph raised (expected): {type(exc).__name__}: {exc}\n")

    print("--- 1. Raw fixture capture (fixture.db) ---")
    from agent_trace._replay.fixture import Fixture

    with Fixture(TRACE_DIR / trace.run_id / "fixture.db") as fixture:
        for exchange in fixture.all_exchanges():
            if exchange["response_status"] == 400:
                print(f"  {exchange['method']} {exchange['url']} -> 400")
                print(f"  body: {exchange['response_body']}")

    print("\n--- 2. What the LangGraph integration span shows today ---")
    spans_as_dicts = trace.to_dict()["spans"]
    _print_errors_only(spans_as_dicts)

    print("\n--- Full span tree ---")
    StdoutExporter().export(trace)

    print(f"\nRun ID: {trace.run_id}")
    print(
        "\nNotice the ERROR spans' inline '! HTTP 400: ...' line — that's "
        "Anthropic's actual rejection text, attached directly to the span "
        "by Span.record_exception's exception.http_response_body capture, "
        "not just a generic 'Client error 400 Bad Request' message."
    )


if __name__ == "__main__":
    main()
