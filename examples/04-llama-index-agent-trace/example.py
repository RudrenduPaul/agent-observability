"""
llama_index framework integration example.

Demonstrates LlamaIndexTracer: installs onto llama_index's global Dispatcher,
captures a span for every instrumented llama_index call (LLM chat/complete,
tool calls, ...) with correct parent/child nesting, and enriches those spans
with the chat-history / tool-call data llama_index's own instrumentation
events carry (LLMChatStartEvent, LLMChatEndEvent, AgentToolCallEvent, ...).

No API key or network access required — this uses llama_index's own MockLLM
and a plain Python FunctionTool, so the example is fully reproducible offline.

Run:
    pip install agent-trace[llama-index]
    python examples/04-llama-index-agent-trace/example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.llms import MockLLM
    from llama_index.core.tools import FunctionTool
except ImportError:
    sys.exit(
        "llama-index-core is not installed.\nRun: pip install agent-trace[llama-index]"
    )

from agent_trace import tracer
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.llama_index import LlamaIndexTracer


def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    # A real implementation would call a weather API. Hardcoded so this
    # example needs no network access or credentials.
    return f"It is sunny and 72F in {city}."


def run_agent_like_flow(question: str) -> str:
    """Simulate a minimal ReAct-style loop: LLM call -> tool call -> LLM call.

    Wrapped in a LlamaIndexTracer context so every llama_index-instrumented
    call along the way (MockLLM.chat/.complete, FunctionTool.call) becomes an
    agent-trace span, nested under the llama_index dispatcher's own
    parent/child structure.
    """
    with tracer.start_trace("llama_index_agent_flow", record=True) as trace:
        with LlamaIndexTracer(tracer=tracer, trace=trace):
            llm = MockLLM()
            weather_tool = FunctionTool.from_defaults(fn=get_weather)

            # Step 1: "decide" to call the weather tool (MockLLM just echoes,
            # but this is enough to exercise the real chat -> complete span
            # nesting and the LLMChatStartEvent/LLMChatEndEvent enrichment).
            llm.chat([ChatMessage(role="user", content=question)])

            # Step 2: call the tool the agent "decided" on.
            tool_result = weather_tool.call(city="San Francisco")

            # Step 3: feed the tool result back to the LLM for a final answer.
            final = llm.chat(
                [
                    ChatMessage(role="user", content=question),
                    ChatMessage(role="tool", content=str(tool_result.raw_output)),
                ]
            )

        run_id = trace.run_id

    trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
    loaded_trace = Trace.from_dict(json.loads(trace_path.read_text()))

    print("\n--- Span tree ---")
    StdoutExporter().export(loaded_trace)

    print("\n--- Selected span attributes ---")
    for span in loaded_trace.spans:
        interesting = {
            k: v
            for k, v in span.attributes.items()
            if k.startswith(("llm.", "agent.", "llama_index.class"))
        }
        if interesting:
            print(f"{span.name}: {interesting}")

    print(f"\nTrace saved to: {trace_path.parent}")
    return str(final.message.content)


def main() -> None:
    question = "What's the weather in San Francisco?"
    print(f"Question: {question}")
    answer = run_agent_like_flow(question)
    print(f"\n--- Final answer ---\n{answer}")


if __name__ == "__main__":
    main()
