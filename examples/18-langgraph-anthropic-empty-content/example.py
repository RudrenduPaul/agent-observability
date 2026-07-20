"""
Anthropic empty-content message replay 400 (#3168).

Anthropic's Messages API rejects any request where an `assistant` message
has empty `content` (`""` or `[]`) unless that message is the *last* one in
the array — the real rejection text is `"messages.N: all messages must have
non-empty content"`. LangGraph agents can produce exactly this shape: a
`create_react_agent` tool-calling turn that returns an `AIMessage` with only
`tool_calls` and no text content, immediately followed by another empty
`AIMessage` (e.g. a provider hiccup, a truncated retry, or a node that
re-invokes the model without new input) — leaving a *non-final* assistant
message with empty content once the next turn's request is built.

This example reproduces that shape end-to-end with a fake `BaseChatModel`
(no API key needed) that makes a real HTTP call — through a real
`httpx.Client`, patched by agent-trace's `RecordingTransport` exactly like a
live Anthropic call would be — to a mock transport returning Anthropic's
actual 400 error body for this failure class. It shows both halves of what
this issue asked for:

1. The **raw fixture capture** of the 400 (`fixture.db`, reachable via
   `Fixture.all_exchanges()` or `agent-trace inspect <run_id>`).
2. `agent_trace._inspect.check_empty_content_not_final` **firing on the
   captured fixture** — the automated flag that catches this shape before
   it ever reaches Anthropic, by walking the request body's `messages` array
   for a non-final assistant message with empty content.

Run:
    python examples/18-langgraph-anthropic-empty-content/example.py

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
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace._cli import _print_errors_only
from agent_trace._inspect import check_empty_content_not_final
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.langgraph import LangGraphTracer

TRACE_DIR = Path.home() / ".agent-trace" / "runs"

# Anthropic's real error body for this exact failure class.
ANTHROPIC_400_BODY = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "messages.1: all messages must have non-empty content",
    },
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


def _mock_anthropic_400(request: "httpx.Request") -> "httpx.Response":
    return httpx.Response(400, json=ANTHROPIC_400_BODY, request=request)


class FakeAnthropicChatModel(BaseChatModel):
    """Stand-in for `ChatAnthropic` (no API key/network needed).

    First turn: proposes a tool call with *empty* text content (a normal,
    valid shape when it's the final message of that request). Second turn
    (once the tool result comes back): the first turn's empty-content
    `AIMessage` is no longer the final message in the growing conversation
    — it now sits in the middle of the `messages` array agent-trace
    captures on the request. This model then makes a real HTTP call —
    through `httpx.Client`, patched by agent-trace's `RecordingTransport`
    the same way a live SDK call would be — to a mock transport returning
    Anthropic's actual 400 body for this exact non-final-empty-content
    shape.
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
            "https://api.anthropic.com/v1/messages",
            json={
                "messages": [
                    {"role": "user", "content": "what's the weather in Boston?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_1", "name": "get_weather"}],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny in Boston"},
                ]
            },
        )
        response.raise_for_status()  # pragma: no cover — always raises above
        return ChatResult(generations=[])  # pragma: no cover — unreachable

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-anthropic-chat-model"


def main() -> None:
    model = FakeAnthropicChatModel()
    graph = create_react_agent(model, tools=[get_weather])

    t = Tracer(trace_dir=TRACE_DIR)
    with t.start_trace("anthropic-empty-content-replay", record=True) as trace:
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

    fixture_path = TRACE_DIR / trace.run_id / "fixture.db"
    with Fixture(fixture_path) as fixture:
        exchanges = fixture.all_exchanges()
        for exchange in exchanges:
            if exchange["response_status"] == 400:
                print(f"  {exchange['method']} {exchange['url']} -> 400")
                print(f"  body: {exchange['response_body']}")

    print("\n--- 2. What the LangGraph integration span shows today ---")
    spans_as_dicts = trace.to_dict()["spans"]
    _print_errors_only(spans_as_dicts)

    print("\n--- 3. check_empty_content_not_final over the captured fixture ---")
    flags = check_empty_content_not_final(exchanges)
    for flag in flags:
        print(f"  FLAGGED: {flag['detail']}")
    assert flags, "expected check_empty_content_not_final to flag the captured exchange"

    print("\n--- Full span tree ---")
    StdoutExporter().export(trace)

    print(f"\nRun ID: {trace.run_id}")
    print(
        "\nThe check caught the exact non-final empty-content assistant message "
        "that triggers Anthropic's real 400 — before ever needing to hand-read "
        "the raw request body."
    )


if __name__ == "__main__":
    main()
