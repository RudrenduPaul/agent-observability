"""
Google GenAI thinking-config example.
Run: uv run python examples/04-google-genai-thinking-config/example.py

Prerequisites:
    pip install agent-observability-trace-cli[google-genai]

Demonstrates what agent_trace.integrations.google_genai adds on top of the
generic httpx interceptor: instead of leaving a Gemini "thinking" call as
opaque JSON in fixture.db, GoogleGenAITracer/instrument_client put
`thinkingConfig`/`includeThoughts`/`thinkingBudget` and the resulting
thoughts-token count directly onto the span as queryable attributes — which
is exactly the field a bug like "thinking budget silently ignored" (or
crewAI's GeminiCompletion never setting `includeThoughts` in the first
place) needs surfaced without hand-reading a raw request body.

No live Gemini API key is required: the underlying
``client.models.generate_content`` call is stubbed with a local function that
returns a real ``google.genai.types.GenerateContentResponse`` shape, so the
example runs offline like 01-basic-trace and 03-ci-pipeline while still
exercising the real SDK's pydantic types end-to-end.

Two call sites are shown side by side:
  1. The raw SDK path (google.genai.Client) via instrument_client()
  2. The LangChain path (ChatGoogleGenerativeAI) via GoogleGenAITracer,
     comparing a bare invoke() against the same call routed through an
     LCEL chain — this is the invocation_context distinction issue #31767
     needed to debug "why does the chain path behave differently".
"""

from __future__ import annotations

import json
import sys
import types as pytypes
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit(
        "google-genai is not installed.\nRun: pip install agent-observability-trace-cli[google-genai]"
    )

from agent_trace import Tracer
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.google_genai import GoogleGenAITracer, instrument_client


def _stub_generate_content(
    *, model: str, contents: object, config: object = None, **kwargs: object
) -> types.GenerateContentResponse:
    """Stand-in for the real network call — returns a real response shape
    (including a non-trivial thoughts_token_count) with zero API cost."""
    return types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=42,
            candidates_token_count=18,
            thoughts_token_count=256,
            total_token_count=316,
        )
    )


def run_raw_sdk_example(tracer: Tracer) -> str:
    """Instrument a raw google.genai.Client and make one generate_content call
    with an explicit thinking config."""
    client = genai.Client(api_key="demo-key-not-used")
    # Swap in the offline stub before instrumenting, so the example never
    # touches the network (see module docstring).
    client.models.generate_content = _stub_generate_content

    with tracer.start_trace("gemini_sdk_thinking_demo") as trace:
        instrument_client(client, tracer=tracer, trace=trace)
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Explain quantum entanglement in one paragraph.",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=1024,
                )
            ),
        )
        run_id = trace.run_id

    return run_id


def run_langchain_example(tracer: Tracer) -> str:
    """Compare a bare ChatGoogleGenerativeAI.invoke() against the same
    request routed through an LCEL chain, using GoogleGenAITracer for both.

    ``_generate`` (the method LangChain calls internally to reach the wire)
    is monkeypatched to a canned ``ChatResult`` so this half of the example
    also runs offline — everything upstream of it (serialization, the
    callback manager, ``on_chat_model_start``/``on_llm_end`` firing with
    ``parent_run_id`` set or unset) is the real langchain-google-genai code
    path, so the resulting spans reflect real behaviour, not a hand-rolled
    stand-in.
    """
    try:
        from langchain_core.messages import AIMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print(
            "\n(skipping LangChain half of the demo — install "
            "langchain-google-genai to see it)"
        )
        return ""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key="demo-key-not-used",
        thinking_budget=1024,
        include_thoughts=True,
    )

    def _stub_generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(
            content="Entangled particles share a single quantum state ...",
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 44,
                "total_tokens": 64,
                "output_token_details": {"reasoning": 44 - 12},
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    llm._generate = pytypes.MethodType(_stub_generate, llm)

    with tracer.start_trace("gemini_langchain_thinking_demo") as trace:
        cb = GoogleGenAITracer(tracer=tracer, trace=trace)

        # 1. Bare invocation -> google_genai.invocation_context = "direct_invocation"
        llm.invoke(
            "Explain quantum entanglement in one paragraph.", config={"callbacks": [cb]}
        )

        # 2. Same model routed through an LCEL chain ->
        #    google_genai.invocation_context = "lcel_chain"
        prompt = ChatPromptTemplate.from_template("{q}")
        chain = prompt | llm | StrOutputParser()
        chain.invoke(
            {"q": "Explain quantum entanglement in one paragraph."},
            config={"callbacks": [cb]},
        )

        run_id = trace.run_id

    return run_id


def _print_trace(run_id: str) -> None:
    trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
    loaded_trace = Trace.from_dict(json.loads(trace_path.read_text()))
    StdoutExporter().export(loaded_trace)
    for span in loaded_trace.spans:
        thinking_attrs = {
            k: v for k, v in span.attributes.items() if k.startswith("google_genai.")
        }
        if thinking_attrs:
            print(f"  {span.name} -> {thinking_attrs}")
    print(f"  (trace saved to {trace_path.parent})\n")


def main() -> None:
    tracer = Tracer()

    print("=== Raw google.genai.Client (instrument_client) ===")
    raw_run_id = run_raw_sdk_example(tracer)
    _print_trace(raw_run_id)

    print("=== langchain_google_genai.ChatGoogleGenerativeAI (GoogleGenAITracer) ===")
    lc_run_id = run_langchain_example(tracer)
    if lc_run_id:
        _print_trace(lc_run_id)


if __name__ == "__main__":
    main()
