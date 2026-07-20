"""
Integration tests for the Google GenAI integration.

Require the real langchain-core / langchain-google-genai / google-genai
packages, but make NO live network calls:

  - GoogleGenAITracer tests use a real ``ChatGoogleGenerativeAI`` instance
    (constructed with a dummy API key) driven directly through langchain-core's
    callback manager on a pure-Python fake chat model / real prompt objects, so
    the serialized-kwargs shape agent_trace parses is the real thing produced
    by langchain-google-genai — not a hand-rolled stand-in.
  - instrument_client tests use a real ``google.genai.Client`` whose
    ``models.generate_content`` bound method is swapped for a local fake
    *before* instrumentation (so instrument_client wraps the fake, never
    touching the network), while still exercising the real
    ``types.GenerateContentConfig``/``types.ThinkingConfig``/
    ``types.GenerateContentResponse`` pydantic models end-to-end.

Run with: uv run pytest tests/integration/ -m integration

Requirements: pip install agent-observability-trace-cli[google-genai]
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "langchain_google_genai", reason="langchain-google-genai not installed"
)
pytest.importorskip("google.genai", reason="google-genai not installed")


@pytest.mark.integration
class TestGoogleGenAITracerIntegration:
    def test_bare_invocation_captures_thinking_fields_and_direct_context(
        self, tmp_path: Path
    ) -> None:
        """A bare ChatGoogleGenerativeAI.invoke() must produce a span carrying
        the real thinking-config fields and invocation_context='direct_invocation'.
        """
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_google_genai import ChatGoogleGenerativeAI

        from agent_trace import Tracer
        from agent_trace.integrations.google_genai import GoogleGenAITracer

        # Real ChatGoogleGenerativeAI, never actually called — we only need its
        # real serialization shape, which we replay through a FakeListChatModel
        # driven by langchain-core's own callback manager so this test makes no
        # network calls while still exercising real to_json() output.
        real_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key="fake-key-for-serialization-only",
            thinking_budget=1024,
            include_thoughts=True,
        )
        serialized = real_llm.to_json()
        assert "thinking_budget" in serialized["kwargs"], (
            "langchain-google-genai's to_json() no longer flattens thinking_budget "
            "into kwargs — GoogleGenAITracer's serialized-kwargs parsing needs updating"
        )

        fake_llm = FakeListChatModel(responses=["hello from gemini"])

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("gg-bare-invoke") as trace:
            cb = GoogleGenAITracer(tracer=t, trace=trace)
            # Drive the real serialized payload through the actual on_chat_model_start
            # path exactly as langchain-core's CallbackManager would.
            import uuid

            run_id = uuid.uuid4()
            cb.on_chat_model_start(serialized, [["hello"]], run_id=run_id)

        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.attributes["llm.model"] == "gemini-2.5-flash"
        assert llm_span.attributes["google_genai.thinking_budget"] == 1024
        assert llm_span.attributes["google_genai.include_thoughts"] is True
        assert (
            llm_span.attributes["google_genai.invocation_context"]
            == "direct_invocation"
        )

    def test_lcel_chain_invocation_sets_chain_context(self, tmp_path: Path) -> None:
        """A real LCEL chain (prompt | llm | parser) must fire on_chat_model_start
        with a non-None parent_run_id, which GoogleGenAITracer must translate to
        invocation_context='lcel_chain'.
        """
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        from agent_trace import Tracer
        from agent_trace.integrations.google_genai import GoogleGenAITracer

        fake_llm = FakeListChatModel(responses=["hello from gemini"])
        prompt = ChatPromptTemplate.from_template("{q}")
        chain = prompt | fake_llm | StrOutputParser()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("gg-lcel-chain") as trace:
            cb = GoogleGenAITracer(tracer=t, trace=trace)
            result = chain.invoke({"q": "hello"}, config={"callbacks": [cb]})

        assert result == "hello from gemini"
        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.attributes["google_genai.invocation_context"] == "lcel_chain"
        # The LLM span must be nested under the chain's root span, not a sibling.
        chain_spans = [s for s in trace.spans if s.name.startswith("chain:")]
        assert any(s.span_id == llm_span.parent_id for s in chain_spans)

    def test_direct_invocation_is_not_nested_under_a_chain(
        self, tmp_path: Path
    ) -> None:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from agent_trace import Tracer
        from agent_trace.integrations.google_genai import GoogleGenAITracer

        fake_llm = FakeListChatModel(responses=["hello"])

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("gg-direct") as trace:
            cb = GoogleGenAITracer(tracer=t, trace=trace)
            fake_llm.invoke("hello", config={"callbacks": [cb]})

        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.parent_id is None
        assert (
            llm_span.attributes["google_genai.invocation_context"]
            == "direct_invocation"
        )


@pytest.mark.integration
class TestInstrumentClientIntegration:
    def test_generate_content_span_uses_real_sdk_types(self, tmp_path: Path) -> None:
        """instrument_client must correctly read thinking-config and usage
        fields off the REAL google.genai.types objects, not just fakes."""
        from google import genai
        from google.genai import types

        from agent_trace import Tracer
        from agent_trace.integrations.google_genai import instrument_client

        client = genai.Client(api_key="fake-key-never-used")

        def _fake_generate_content(*, model, contents, config=None, **kwargs):
            return types.GenerateContentResponse(
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=12,
                    candidates_token_count=7,
                    thoughts_token_count=4,
                    total_token_count=23,
                )
            )

        # Replace BEFORE instrumenting so instrument_client wraps our fake
        # instead of hitting the network.
        client.models.generate_content = _fake_generate_content

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("gg-sdk-real-types") as trace:
            instrument_client(client, tracer=t, trace=trace)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="hi",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True, thinking_budget=2048
                    )
                ),
            )

        assert response.usage_metadata.total_token_count == 23
        assert len(trace.spans) == 1
        span = trace.spans[0]
        assert span.attributes["google_genai.include_thoughts"] is True
        assert span.attributes["google_genai.thinking_budget"] == 2048
        assert span.attributes["llm.usage.prompt_tokens"] == 12
        assert span.attributes["llm.usage.completion_tokens"] == 7
        assert span.attributes["llm.usage.total_tokens"] == 23
        assert span.attributes["google_genai.usage.thoughts_tokens"] == 4

    def test_uninstrument_restores_real_client(self, tmp_path: Path) -> None:
        from google import genai

        from agent_trace import Tracer
        from agent_trace.integrations.google_genai import (
            instrument_client,
            uninstrument_client,
        )

        client = genai.Client(api_key="fake-key-never-used")

        def _fake_generate_content(*, model, contents, config=None, **kwargs):
            return "stub-response"

        client.models.generate_content = _fake_generate_content

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("gg-sdk-uninstrument") as trace:
            instrument_client(client, tracer=t, trace=trace)
            assert client.models.generate_content is not _fake_generate_content
            uninstrument_client(client)
            assert client.models.generate_content is _fake_generate_content
