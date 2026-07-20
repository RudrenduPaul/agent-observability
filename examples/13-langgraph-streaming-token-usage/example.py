"""
Streaming token usage: the LangGraphTracer callback gap (#3911).

Issue #3911 reports LangChain's own `get_openai_callback()` returning zero
tokens under `stream_mode="messages"` — because `get_openai_callback` only
reads `response.llm_output["token_usage"]`, and in streaming configurations
many providers/langchain-core versions never populate `llm_output` at all;
usage instead arrives on the final aggregated chunk's own
`AIMessageChunk.usage_metadata` field.

This example reproduces that exact shape with a fake streaming
`BaseChatModel` (no API key needed): `llm_output` is never populated (the
`_generate`/non-streaming path isn't used at all here), but the final
streamed chunk carries `usage_metadata`. It shows two things:

1. What a naive `llm_output`-only reader — the same field
   `get_openai_callback` and agent-trace's `on_llm_end` *used* to read
   exclusively — would report: zero/missing usage, reproducing #3911's
   exact symptom.
2. What `LangGraphTracer.on_llm_end` reports today: correct token counts,
   via the `response.generations[0][0].message.usage_metadata` fallback
   added to `_extract_token_usage()`
   (`src/agent_trace/integrations/langgraph.py`) — the wire-level truth
   surfaced at the span level, no raw fixture-body inspection required.

Run:
    python examples/13-langgraph-streaming-token-usage/example.py

No API key required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, UsageMetadata
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-observability-trace-cli[langgraph]")

from agent_trace import Tracer
from agent_trace.integrations.langgraph import LangGraphTracer

TRACE_DIR = Path.home() / ".agent-trace" / "runs"


class FakeStreamingChatModel(BaseChatModel):
    """Stand-in for `ChatOpenAI(streaming=True)` (no API key/network
    needed). `llm_output` is never populated — mirrors the real-world shape
    #3911 hit, where streaming responses carry usage only on the final
    chunk's `usage_metadata`, not on `llm_output["token_usage"]`."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Non-streaming fallback path — not exercised by .stream() below,
        # included only so this remains a complete BaseChatModel.
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hi"))])

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        words = ["The ", "answer ", "is ", "42."]
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            usage = (
                UsageMetadata(input_tokens=14, output_tokens=6, total_tokens=20)
                if is_last
                else None
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=word, usage_metadata=usage)
            )

    @property
    def _llm_type(self) -> str:
        return "fake-streaming-chat-model"


def main() -> None:
    model = FakeStreamingChatModel()

    t = Tracer(trace_dir=TRACE_DIR)
    with t.start_trace("streaming-token-usage") as trace:
        cb = LangGraphTracer(tracer=t, trace=trace)
        chunks = list(model.stream("What is the answer?", config={"callbacks": [cb]}))

    streamed_content = "".join(c.content for c in chunks)
    print(f"Streamed content: {streamed_content!r}")

    llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))

    print("\n--- 1. What a naive llm_output-only reader sees (#3911's bug) ---")
    print(
        "get_openai_callback() and any code reading only "
        "`response.llm_output['token_usage']` sees NOTHING here — "
        "llm_output is never populated in this streaming shape. Usage is "
        "zero/missing, exactly like #3911's report."
    )

    print("\n--- 2. What LangGraphTracer.on_llm_end captures today ---")
    print(f"  llm.streamed          = {llm_span.attributes.get('llm.streamed')}")
    print(f"  llm.stream_token_count= {llm_span.attributes.get('llm.stream_token_count')}")
    print(f"  llm.usage.prompt_tokens     = {llm_span.attributes.get('llm.usage.prompt_tokens')}")
    print(f"  llm.usage.completion_tokens = {llm_span.attributes.get('llm.usage.completion_tokens')}")
    print(f"  llm.usage.total_tokens      = {llm_span.attributes.get('llm.usage.total_tokens')}")

    print(
        "\nThe usage.* attributes above came from "
        "response.generations[0][0].message.usage_metadata — the "
        "_extract_token_usage() fallback in "
        "src/agent_trace/integrations/langgraph.py — not from llm_output, "
        "which stayed empty for this entire run. A create_react_agent graph "
        "built on a real streaming ChatOpenAI/AzureChatOpenAI model goes "
        "through the identical on_chat_model_start/on_llm_end callback "
        "path, so this same fallback applies inside a full graph run too."
    )
    print(f"\nRun ID: {trace.run_id}")


if __name__ == "__main__":
    main()
