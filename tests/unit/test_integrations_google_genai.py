"""
Unit tests for agent_trace.integrations.google_genai.

Neither langchain_core nor google-genai are installed test dependencies —
these tests inject fake modules into sys.modules (GoogleGenAITracer tests,
mirroring test_integrations_langgraph.py) or fake objects (instrument_client
tests) so they run in CI without the real optional packages.  Real-package
integration coverage lives in tests/integration/test_google_genai.py.
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from agent_trace import SpanStatus, Tracer

# ---------------------------------------------------------------------------
# Fake langchain_core fixture (module-level injection) — mirrors
# test_integrations_langgraph.py's _make_fake_langchain_core().
# ---------------------------------------------------------------------------


def _make_fake_langchain_core() -> dict[str, ModuleType]:
    class FakeBaseCallbackHandler:
        def __init__(self) -> None:
            pass

    fake_callbacks = types.ModuleType("langchain_core.callbacks")
    fake_callbacks.BaseCallbackHandler = FakeBaseCallbackHandler  # type: ignore[attr-defined]

    fake_lc = types.ModuleType("langchain_core")
    fake_lc.callbacks = fake_callbacks  # type: ignore[attr-defined]

    return {
        "langchain_core": fake_lc,
        "langchain_core.callbacks": fake_callbacks,
    }


@pytest.fixture()
def patched_langchain(monkeypatch):
    fakes = _make_fake_langchain_core()
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)

    import agent_trace.integrations.google_genai as gg_module

    original = gg_module._GoogleGenAITracerClass
    gg_module._GoogleGenAITracerClass = None

    yield fakes

    gg_module._GoogleGenAITracerClass = original


@pytest.fixture()
def tracer_and_trace(tmp_path: Path, patched_langchain):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("gg-unit-test") as trace:
        yield t, trace


def _make_handler(t, trace):
    from agent_trace.integrations.google_genai import GoogleGenAITracer

    return GoogleGenAITracer(tracer=t, trace=trace)


def _run_id() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Initialisation — same __new__ / __init__ wiring pattern as LangGraphTracer
# ---------------------------------------------------------------------------


class TestGoogleGenAITracerInit:
    def test_tracer_attribute_is_set(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        assert handler._tracer is t

    def test_trace_attribute_is_set(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        assert handler._trace is trace

    def test_spans_dict_starts_empty(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        assert handler._spans == {}

    def test_two_instances_have_independent_span_dicts(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h1 = _make_handler(t, trace)
        h2 = _make_handler(t, trace)
        assert h1._spans is not h2._spans


# ---------------------------------------------------------------------------
# on_chat_model_start — thinking-config capture + invocation context
# ---------------------------------------------------------------------------


class TestChatModelStartThinkingFields:
    def test_flat_thinking_budget_captured(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash", "thinking_budget": 1024}},
            [["hi"]],
            run_id=run_id,
        )
        span = handler._spans[str(run_id)]
        assert span.attributes["google_genai.thinking_budget"] == 1024

    def test_flat_include_thoughts_captured(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash", "include_thoughts": True}},
            [["hi"]],
            run_id=run_id,
        )
        span = handler._spans[str(run_id)]
        assert span.attributes["google_genai.include_thoughts"] is True

    def test_flat_thinking_level_captured(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-3-pro", "thinking_level": "low"}},
            [["hi"]],
            run_id=run_id,
        )
        span = handler._spans[str(run_id)]
        assert span.attributes["google_genai.thinking_level"] == "low"

    def test_nested_thinking_config_dict_captured(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {
                "kwargs": {
                    "model": "gemini-2.5-flash",
                    "thinking_config": {
                        "include_thoughts": True,
                        "thinking_budget": 512,
                    },
                }
            },
            [["hi"]],
            run_id=run_id,
        )
        span = handler._spans[str(run_id)]
        assert span.attributes["google_genai.include_thoughts"] is True
        assert span.attributes["google_genai.thinking_budget"] == 512

    def test_no_thinking_fields_when_absent(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [["hi"]], run_id=run_id
        )
        span = handler._spans[str(run_id)]
        assert "google_genai.thinking_budget" not in span.attributes
        assert "google_genai.include_thoughts" not in span.attributes

    def test_model_name_used_in_span_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [["hi"]], run_id=run_id
        )
        span = handler._spans[str(run_id)]
        assert "gemini-2.5-flash" in span.name
        assert span.attributes["llm.model"] == "gemini-2.5-flash"


class TestInvocationContext:
    def test_bare_invocation_is_direct(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}},
            [["hi"]],
            run_id=run_id,
            parent_run_id=None,
        )
        span = handler._spans[str(run_id)]
        assert span.attributes["google_genai.invocation_context"] == "direct_invocation"

    def test_lcel_chain_invocation_is_lcel_chain(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        chain_run_id = _run_id()
        handler.on_chain_start({"name": "my_chain"}, {}, run_id=chain_run_id)

        llm_run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}},
            [["hi"]],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )
        span = handler._spans[str(llm_run_id)]
        assert span.attributes["google_genai.invocation_context"] == "lcel_chain"

    def test_lcel_chain_llm_span_is_child_of_chain_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        chain_run_id = _run_id()
        handler.on_chain_start({"name": "my_chain"}, {}, run_id=chain_run_id)
        chain_span = handler._spans[str(chain_run_id)]

        llm_run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}},
            [["hi"]],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )
        llm_span = handler._spans[str(llm_run_id)]
        assert llm_span.parent_id == chain_span.span_id


# ---------------------------------------------------------------------------
# on_llm_end — token usage, including Gemini's thoughts-token count
# ---------------------------------------------------------------------------


class TestLLMEndUsage:
    def test_usage_metadata_prompt_and_completion_tokens(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [["hi"]], run_id=run_id
        )
        span_ref = handler._spans[str(run_id)]

        message = MagicMock()
        message.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 8,
            "total_tokens": 18,
        }
        generation = MagicMock()
        generation.message = message
        response = MagicMock()
        response.generations = [[generation]]

        handler.on_llm_end(response, run_id=run_id)

        assert span_ref.attributes["llm.usage.prompt_tokens"] == 10
        assert span_ref.attributes["llm.usage.completion_tokens"] == 8
        assert span_ref.attributes["llm.usage.total_tokens"] == 18

    def test_usage_metadata_thoughts_tokens_captured(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [["hi"]], run_id=run_id
        )
        span_ref = handler._spans[str(run_id)]

        message = MagicMock()
        message.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_token_details": {"reasoning": 12},
        }
        generation = MagicMock()
        generation.message = message
        response = MagicMock()
        response.generations = [[generation]]

        handler.on_llm_end(response, run_id=run_id)

        assert span_ref.attributes["google_genai.usage.thoughts_tokens"] == 12

    def test_falls_back_to_llm_output_token_usage(self, tracer_and_trace):
        """Non-Gemini providers (no usage_metadata on the message) still work."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gpt-4o"}}, [["hi"]], run_id=run_id
        )
        span_ref = handler._spans[str(run_id)]

        response = MagicMock()
        response.generations = []
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            }
        }

        handler.on_llm_end(response, run_id=run_id)

        assert span_ref.attributes["llm.usage.prompt_tokens"] == 4
        assert span_ref.attributes["llm.usage.completion_tokens"] == 2
        assert span_ref.attributes["llm.usage.total_tokens"] == 6

    def test_on_llm_end_closes_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [["hi"]], run_id=run_id
        )
        span_ref = handler._spans[str(run_id)]

        response = MagicMock()
        response.generations = []
        response.llm_output = {}
        handler.on_llm_end(response, run_id=run_id)

        assert span_ref.end_time is not None
        assert span_ref.status == SpanStatus.OK
        assert str(run_id) not in handler._spans

    def test_on_llm_end_tolerates_missing_usage(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [["hi"]], run_id=run_id
        )
        response = MagicMock()
        response.generations = [[MagicMock(message=MagicMock(usage_metadata=None))]]
        response.llm_output = {}
        handler.on_llm_end(response, run_id=run_id)  # must not raise


class TestLLMError:
    def test_on_llm_error_marks_span_error(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [["hi"]], run_id=run_id
        )
        span_ref = handler._spans[str(run_id)]
        handler.on_llm_error(RuntimeError("boom"), run_id=run_id)
        assert str(run_id) not in handler._spans
        assert span_ref.status == SpanStatus.ERROR


# ---------------------------------------------------------------------------
# Chain / tool passthrough (generic, standalone usage without LangGraphTracer)
# ---------------------------------------------------------------------------


class TestChainAndToolPassthrough:
    def test_chain_start_end_round_trip(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "seq"}, {}, run_id=run_id)
        assert str(run_id) in handler._spans
        handler.on_chain_end({}, run_id=run_id)
        assert str(run_id) not in handler._spans

    def test_tool_start_end_round_trip(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "query", run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.attributes["tool.name"] == "search"
        handler.on_tool_end("result", run_id=run_id)
        assert str(run_id) not in handler._spans

    def test_tool_error_marks_span_error(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "query", run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_tool_error(ValueError("nope"), run_id=run_id)
        assert span_ref.status == SpanStatus.ERROR


# ---------------------------------------------------------------------------
# instrument_client / uninstrument_client — raw google.genai.Client SDK
#
# These tests fake out the "google.genai" module rather than requiring the
# real package, exercising instrument_client's patch/restore logic and
# attribute extraction against plain fake objects with the field names
# confirmed via introspection of the real google-genai 2.10.0 SDK
# (types.GenerateContentConfig.thinking_config,
# types.ThinkingConfig.{include_thoughts,thinking_budget,thinking_level},
# types.GenerateContentResponse.usage_metadata,
# types.GenerateContentResponseUsageMetadata.{prompt_token_count,
# candidates_token_count,total_token_count,thoughts_token_count}).
# ---------------------------------------------------------------------------


class _FakeThinkingConfig:
    def __init__(
        self, include_thoughts=None, thinking_budget=None, thinking_level=None
    ):
        self.include_thoughts = include_thoughts
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level


class _FakeGenerateContentConfig:
    def __init__(self, thinking_config=None):
        self.thinking_config = thinking_config


class _FakeUsageMetadata:
    def __init__(
        self,
        prompt_token_count=None,
        candidates_token_count=None,
        total_token_count=None,
        thoughts_token_count=None,
    ):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.thoughts_token_count = thoughts_token_count


class _FakeGenerateContentResponse:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata


class _FakeModels:
    """Stands in for google.genai.models.Models — a stable per-Client instance."""

    def __init__(self):
        self.generate_content_calls = []
        self.generate_content_stream_calls = []

    def generate_content(self, *, model, contents, config=None, **kwargs):
        self.generate_content_calls.append((model, contents, config))
        return _FakeGenerateContentResponse(
            usage_metadata=_FakeUsageMetadata(
                prompt_token_count=5,
                candidates_token_count=3,
                total_token_count=8,
            )
        )

    def generate_content_stream(self, *, model, contents, config=None, **kwargs):
        self.generate_content_stream_calls.append((model, contents, config))
        yield _FakeGenerateContentResponse(
            usage_metadata=_FakeUsageMetadata(
                prompt_token_count=5,
                candidates_token_count=1,
                total_token_count=6,
            )
        )
        yield _FakeGenerateContentResponse(
            usage_metadata=_FakeUsageMetadata(
                prompt_token_count=5,
                candidates_token_count=3,
                total_token_count=8,
            )
        )


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


@pytest.fixture()
def fake_google_genai(monkeypatch):
    """Inject a minimal fake google.genai package so _require_google_genai()
    (a plain ``import google.genai``) succeeds without the real dependency."""
    fake_genai = types.ModuleType("google.genai")
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    yield


@pytest.fixture()
def client_tracer_and_trace(tmp_path: Path, fake_google_genai):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("gg-sdk-unit-test") as trace:
        yield t, trace


class TestInstrumentClient:
    def test_generate_content_creates_span(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        instrument_client(client, tracer=t, trace=trace)

        client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert len(trace.spans) == 1
        assert "gemini-2.5-flash" in trace.spans[0].name
        assert trace.spans[0].status == SpanStatus.OK

    def test_generate_content_captures_thinking_config(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        instrument_client(client, tracer=t, trace=trace)

        config = _FakeGenerateContentConfig(
            thinking_config=_FakeThinkingConfig(
                include_thoughts=True, thinking_budget=1024
            )
        )
        client.models.generate_content(
            model="gemini-2.5-flash", contents="hi", config=config
        )

        span = trace.spans[0]
        assert span.attributes["google_genai.include_thoughts"] is True
        assert span.attributes["google_genai.thinking_budget"] == 1024

    def test_generate_content_captures_usage(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        instrument_client(client, tracer=t, trace=trace)

        client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        span = trace.spans[0]
        assert span.attributes["llm.usage.prompt_tokens"] == 5
        assert span.attributes["llm.usage.completion_tokens"] == 3
        assert span.attributes["llm.usage.total_tokens"] == 8

    def test_original_call_still_happens(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        instrument_client(client, tracer=t, trace=trace)

        client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert len(client.models.generate_content_calls) == 1

    def test_exception_marks_span_error_and_propagates(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()

        def _boom(*, model, contents, config=None, **kwargs):
            raise RuntimeError("api down")

        client.models.generate_content = _boom
        instrument_client(client, tracer=t, trace=trace)

        with pytest.raises(RuntimeError, match="api down"):
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")

        assert trace.spans[0].status == SpanStatus.ERROR

    def test_instrument_is_idempotent(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        instrument_client(client, tracer=t, trace=trace)
        patched_once = client.models.generate_content
        instrument_client(client, tracer=t, trace=trace)
        assert client.models.generate_content is patched_once

    def test_uninstrument_restores_original(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import (
            instrument_client,
            uninstrument_client,
        )

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        original_func = client.models.generate_content.__func__
        instrument_client(client, tracer=t, trace=trace)
        assert client.models.generate_content.__func__ is not original_func

        uninstrument_client(client)
        assert client.models.generate_content.__func__ is original_func

    def test_uninstrument_without_instrument_is_noop(self, client_tracer_and_trace):
        from agent_trace.integrations.google_genai import uninstrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        uninstrument_client(client)  # must not raise

    def test_generate_content_stream_creates_span_with_final_usage(
        self, client_tracer_and_trace
    ):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()
        instrument_client(client, tracer=t, trace=trace)

        chunks = list(
            client.models.generate_content_stream(
                model="gemini-2.5-flash", contents="hi"
            )
        )

        assert len(chunks) == 2
        assert len(trace.spans) == 1
        span = trace.spans[0]
        assert span.status == SpanStatus.OK
        # Final chunk's cumulative usage should win.
        assert span.attributes["llm.usage.completion_tokens"] == 3
        assert span.attributes["llm.usage.total_tokens"] == 8

    def test_generate_content_stream_error_marks_span_error(
        self, client_tracer_and_trace
    ):
        from agent_trace.integrations.google_genai import instrument_client

        t, trace = client_tracer_and_trace
        client = _FakeClient()

        def _boom_stream(*, model, contents, config=None, **kwargs):
            yield _FakeGenerateContentResponse()
            raise RuntimeError("stream broke")

        client.models.generate_content_stream = _boom_stream
        instrument_client(client, tracer=t, trace=trace)

        with pytest.raises(RuntimeError, match="stream broke"):
            list(
                client.models.generate_content_stream(
                    model="gemini-2.5-flash", contents="hi"
                )
            )

        assert trace.spans[0].status == SpanStatus.ERROR


class TestRequireGoogleGenAIMissing:
    def test_instrument_client_raises_clear_error_without_package(
        self, tmp_path: Path, monkeypatch
    ):
        """instrument_client must raise a clear ImportError when google-genai
        is not installed (simulated by removing it from sys.modules and
        blocking the import)."""
        from agent_trace.integrations import google_genai as gg_module

        monkeypatch.delitem(sys.modules, "google.genai", raising=False)
        monkeypatch.delitem(sys.modules, "google", raising=False)

        real_import = __import__

        def _blocked_import(name, *args, **kwargs):
            if name == "google" or name.startswith("google.genai"):
                raise ImportError("no google-genai installed")
            return real_import(name, *args, **kwargs)

        import builtins

        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("gg-missing-pkg") as trace, pytest.raises(ImportError):
            gg_module.instrument_client(_FakeClient(), tracer=t, trace=trace)
