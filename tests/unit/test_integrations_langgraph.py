"""
Unit tests for agent_trace.integrations.langgraph.LangGraphTracer.

langchain_core is NOT an installed test dependency — these tests mock the
BaseCallbackHandler base class so they can run in CI without the real package.
"""

from __future__ import annotations

import sys
import threading
import types
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from agent_trace import Tracer

# ---------------------------------------------------------------------------
# Fake langchain_core fixture (module-level injection)
# ---------------------------------------------------------------------------


def _make_fake_langchain_core() -> dict[str, ModuleType]:
    """Return a minimal sys.modules patch that satisfies langchain imports."""

    class FakeBaseCallbackHandler:
        """Stub that carries the same interface agent_trace actually calls."""

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
    """Inject a fake langchain_core into sys.modules for the duration of a test."""
    fakes = _make_fake_langchain_core()
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Force _get_tracer_class to rebuild (it caches the concrete class once).
    import agent_trace.integrations.langgraph as lg_module

    original = lg_module._LangGraphTracerClass
    lg_module._LangGraphTracerClass = None

    # _install_runtime_capture_patch() also caches its outcome globally
    # (success or failure) the first time it ever runs. If *this* test is
    # the first thing in the whole process to trigger it, the fake
    # langchain_core above makes the (real) langgraph._internal._runnable
    # module's own `from langchain_core.runnables import ...` fail — not
    # because the real patch is broken, but because langchain_core is a
    # stub here. Save/restore the flag around the fake-module window so
    # that false negative doesn't permanently poison the patch for the rest
    # of the test session (e.g. the later, real-langchain_core integration
    # tests).
    original_runtime_patch_installed = lg_module._runtime_patch_installed

    yield fakes

    # Restore the cached class so other tests (real langchain, integration) are
    # not affected by the reset.
    lg_module._LangGraphTracerClass = original
    lg_module._runtime_patch_installed = original_runtime_patch_installed


@pytest.fixture()
def tracer_and_trace(tmp_path: Path, patched_langchain):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("lg-unit-test") as trace:
        yield t, trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(t, trace):
    from agent_trace.integrations.langgraph import LangGraphTracer

    return LangGraphTracer(tracer=t, trace=trace)


def _run_id() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Initialisation — tests for the __new__ / __init__ wiring bug
# ---------------------------------------------------------------------------


class TestLangGraphTracerInit:
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

    def test_lock_is_a_lock(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        assert isinstance(handler._lock, threading.Lock)

    def test_two_instances_have_independent_span_dicts(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h1 = _make_handler(t, trace)
        h2 = _make_handler(t, trace)
        assert h1._spans is not h2._spans

    def test_two_instances_have_independent_locks(self, tracer_and_trace):
        t, trace = tracer_and_trace
        h1 = _make_handler(t, trace)
        h2 = _make_handler(t, trace)
        assert h1._lock is not h2._lock


# ---------------------------------------------------------------------------
# Callback round-trips — span lifecycle
# ---------------------------------------------------------------------------


class TestLangGraphCallbacks:
    def test_chain_start_registers_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        assert str(run_id) in handler._spans

    def test_chain_end_removes_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        handler.on_chain_end({}, run_id=run_id)
        assert str(run_id) not in handler._spans

    def test_chain_error_marks_span_error(self, tracer_and_trace):
        from agent_trace.core.span import SpanStatus

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "failing"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(ValueError("boom"), run_id=run_id)
        assert str(run_id) not in handler._spans
        assert span_ref.status == SpanStatus.ERROR

    def test_llm_start_registers_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start({"kwargs": {"model_name": "gpt-4"}}, ["hi"], run_id=run_id)
        assert str(run_id) in handler._spans

    def test_llm_end_closes_span_and_records_tokens(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start({"kwargs": {}}, ["hi"], run_id=run_id)
        span_ref = handler._spans[str(run_id)]

        response = MagicMock()
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        }
        handler.on_llm_end(response, run_id=run_id)

        assert str(run_id) not in handler._spans
        assert span_ref.attributes.get("llm.usage.total_tokens") == 15

    def test_llm_end_bad_usage_does_not_raise(self, tracer_and_trace):
        """on_llm_end must not propagate exceptions from malformed llm_output."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start({"kwargs": {}}, ["hi"], run_id=run_id)
        bad_response = MagicMock()
        bad_response.llm_output = None
        handler.on_llm_end(bad_response, run_id=run_id)  # must not raise
        assert str(run_id) not in handler._spans

    def test_tool_start_and_end_round_trip(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "query", run_id=run_id)
        assert str(run_id) in handler._spans
        handler.on_tool_end("result", run_id=run_id)
        assert str(run_id) not in handler._spans

    def test_tool_error_marks_span_error(self, tracer_and_trace):
        from agent_trace.core.span import SpanStatus

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "query", run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_tool_error(RuntimeError("tool broke"), run_id=run_id)
        assert str(run_id) not in handler._spans
        assert span_ref.status == SpanStatus.ERROR

    def test_parent_child_span_wiring(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        parent_id = _run_id()
        child_id = _run_id()

        handler.on_chain_start({"name": "parent"}, {}, run_id=parent_id)
        parent_span = handler._spans[str(parent_id)]

        handler.on_tool_start(
            {"name": "child_tool"},
            "q",
            run_id=child_id,
            parent_run_id=parent_id,
        )
        child_span = handler._spans[str(child_id)]

        assert child_span.parent_id == parent_span.span_id

    def test_chat_model_start_registers_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id=run_id)
        assert str(run_id) in handler._spans
        span = handler._spans[str(run_id)]
        assert span.name == "llm:ChatOpenAI"

    def test_chat_model_start_kwargs_model_name_in_span_name(self, tracer_and_trace):
        """Model name from serialized.kwargs.model_name appears in span name."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model_name": "gpt-4o"}}, [[]], run_id=run_id
        )
        span = handler._spans[str(run_id)]
        assert span.name == "llm:gpt-4o"
        assert span.attributes.get("llm.model") == "gpt-4o"

    def test_unknown_run_id_in_end_is_noop(self, tracer_and_trace):
        """Closing a span that was never opened must not raise."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        phantom_id = _run_id()
        handler.on_chain_end({}, run_id=phantom_id)  # must not raise
        handler.on_tool_end("x", run_id=phantom_id)  # must not raise


# ---------------------------------------------------------------------------
# Previously-discarded data — now captured onto spans
# ---------------------------------------------------------------------------


class TestChainInputsOutputsMetadata:
    """on_chain_start/on_chain_end: persist inputs/outputs/metadata."""

    def test_chain_start_captures_inputs(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {"x": 1, "y": "hi"}, run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.attributes.get("chain.inputs") == '{"x": 1, "y": "hi"}'

    def test_chain_start_empty_inputs_sets_no_attribute(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        span = handler._spans[str(run_id)]
        assert "chain.inputs" not in span.attributes

    def test_chain_end_captures_outputs(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_end({"result": "done"}, run_id=run_id)
        assert span_ref.attributes.get("chain.outputs") == '{"result": "done"}'

    def test_chain_start_captures_metadata(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start(
            {"name": "my_node"},
            {},
            run_id=run_id,
            metadata={"user_key": "abc"},
        )
        span = handler._spans[str(run_id)]
        assert span.attributes.get("chain.metadata") == '{"user_key": "abc"}'

    def test_chain_end_unknown_run_id_is_noop(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        handler.on_chain_end({"result": "done"}, run_id=_run_id())  # must not raise


class TestChatModelMessagesCapture:
    """on_chat_model_start: persist the full messages list, not just the model name."""

    def test_chat_model_start_captures_messages(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[{"type": "human", "content": "hello"}]],
            run_id=run_id,
        )
        span = handler._spans[str(run_id)]
        assert "hello" in span.attributes.get("llm.messages", "")

    def test_chat_model_start_serializes_basemessage_like_objects(
        self, tracer_and_trace
    ):
        """Objects exposing model_dump() (BaseMessage's pydantic-v2 shape)
        must be serialized via model_dump(), not str()."""

        class FakeMessage:
            def __init__(self, content: str) -> None:
                self.content = content

            def model_dump(self) -> dict:
                return {"type": "human", "content": self.content}

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"}, [[FakeMessage("via model_dump")]], run_id=run_id
        )
        span = handler._spans[str(run_id)]
        assert "via model_dump" in span.attributes.get("llm.messages", "")

    def test_chat_model_start_captures_metadata(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[]],
            run_id=run_id,
            metadata={"run_id": "corr-123"},
        )
        span = handler._spans[str(run_id)]
        assert "corr-123" in span.attributes.get("llm.metadata", "")


class TestLlmEndResponseCapture:
    """on_llm_end: response content, response_metadata/generation_info,
    finish_reason, tool-call presence, and the usage_metadata fallback."""

    def test_llm_end_captures_content(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "gpt-4"}, [[]], run_id=run_id)

        message = MagicMock()
        message.content = "hello from the model"
        message.tool_calls = []
        message.response_metadata = {}
        gen = MagicMock()
        gen.message = message
        gen.generation_info = {}
        response = MagicMock()
        response.llm_output = {}
        response.generations = [[gen]]

        handler.on_llm_end(response, run_id=run_id)
        assert trace.spans[-1].attributes.get("llm.content") == "hello from the model"

    def test_llm_end_captures_finish_reason(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "gemini"}, [[]], run_id=run_id)

        message = MagicMock()
        message.content = ""
        message.tool_calls = []
        message.response_metadata = {"finish_reason": "MALFORMED_FUNCTION_CALL"}
        gen = MagicMock()
        gen.message = message
        gen.generation_info = {}
        response = MagicMock()
        response.llm_output = {}
        response.generations = [[gen]]

        handler.on_llm_end(response, run_id=run_id)
        assert (
            trace.spans[-1].attributes.get("llm.finish_reason")
            == "MALFORMED_FUNCTION_CALL"
        )
        assert "MALFORMED_FUNCTION_CALL" in trace.spans[-1].attributes.get(
            "llm.response_metadata", ""
        )

    def test_llm_end_captures_has_tool_calls_true(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "gpt-4"}, [[]], run_id=run_id)

        message = MagicMock()
        message.content = ""
        message.tool_calls = [{"name": "search", "args": {}, "id": "1"}]
        message.response_metadata = {}
        gen = MagicMock()
        gen.message = message
        gen.generation_info = {}
        response = MagicMock()
        response.llm_output = {}
        response.generations = [[gen]]

        handler.on_llm_end(response, run_id=run_id)
        assert trace.spans[-1].attributes.get("llm.has_tool_calls") is True

    def test_llm_end_usage_metadata_fallback(self, tracer_and_trace):
        """When llm_output carries no usage, fall back to
        response.generations[0][0].message.usage_metadata."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "gpt-4"}, [[]], run_id=run_id)

        message = MagicMock()
        message.content = ""
        message.tool_calls = []
        message.response_metadata = {}
        message.usage_metadata = {
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
        }
        gen = MagicMock()
        gen.message = message
        gen.generation_info = {}
        response = MagicMock()
        response.llm_output = {}  # no token_usage/usage here
        response.generations = [[gen]]

        handler.on_llm_end(response, run_id=run_id)
        span = trace.spans[-1]
        assert span.attributes.get("llm.usage.prompt_tokens") == 7
        assert span.attributes.get("llm.usage.completion_tokens") == 3
        assert span.attributes.get("llm.usage.total_tokens") == 10

    def test_llm_end_llm_output_usage_takes_priority_over_fallback(
        self, tracer_and_trace
    ):
        """If llm_output already carries usage, don't overwrite with the
        usage_metadata fallback."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "gpt-4"}, [[]], run_id=run_id)

        message = MagicMock()
        message.content = ""
        message.tool_calls = []
        message.response_metadata = {}
        message.usage_metadata = {
            "input_tokens": 999,
            "output_tokens": 999,
            "total_tokens": 999,
        }
        gen = MagicMock()
        gen.message = message
        gen.generation_info = {}
        response = MagicMock()
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            }
        }
        response.generations = [[gen]]

        handler.on_llm_end(response, run_id=run_id)
        span = trace.spans[-1]
        assert span.attributes.get("llm.usage.total_tokens") == 3

    def test_llm_end_malformed_response_does_not_raise(self, tracer_and_trace):
        """A response missing the expected shape entirely must not crash
        on_llm_end (defensive serialization / attribute-error guards)."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "gpt-4"}, [[]], run_id=run_id)
        response = MagicMock()
        response.llm_output = None
        response.generations = None
        handler.on_llm_end(response, run_id=run_id)  # must not raise
        assert str(run_id) not in handler._spans


class TestToolInputOutputMetadataCapture:
    """on_tool_start/on_tool_end: raw input, output text, metadata,
    thread name, and event-loop state."""

    def test_tool_start_captures_input_str(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "raw query text", run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.attributes.get("tool.input") == "raw query text"

    def test_tool_end_captures_output(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "q", run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_tool_end("the tool's result text", run_id=run_id)
        assert span_ref.attributes.get("tool.output") == "the tool's result text"

    def test_tool_start_captures_metadata(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start(
            {"name": "search"},
            "q",
            run_id=run_id,
            metadata={"user_key": "abc"},
        )
        span = handler._spans[str(run_id)]
        assert span.attributes.get("tool.metadata") == '{"user_key": "abc"}'

    def test_tool_start_captures_thread_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "q", run_id=run_id)
        span = handler._spans[str(run_id)]
        assert (
            span.attributes.get("tool.thread_name") == threading.current_thread().name
        )

    def test_tool_start_captures_no_event_loop_in_sync_context(self, tracer_and_trace):
        """Pytest test functions run synchronously — no event loop running."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "q", run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.attributes.get("tool.has_event_loop") is False

    def test_tool_end_none_output_sets_no_attribute(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "q", run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_tool_end(None, run_id=run_id)
        assert "tool.output" not in span_ref.attributes


class TestLlmStartMetadataCapture:
    def test_llm_start_captures_metadata(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start(
            {"kwargs": {}},
            ["hi"],
            run_id=run_id,
            metadata={"user_key": "abc"},
        )
        span = handler._spans[str(run_id)]
        assert span.attributes.get("llm.metadata") == '{"user_key": "abc"}'


class TestRuntimeContextCapture:
    """chain.runtime — captured via the ContextVar the RunnableCallable
    monkeypatch (_install_runtime_capture_patch) populates."""

    def test_chain_start_captures_runtime_when_context_var_set(self, tracer_and_trace):
        from agent_trace.integrations.langgraph import _current_runtime

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()

        class FakeRuntime:
            def __repr__(self) -> str:
                return "FakeRuntime(context=None)"

        token = _current_runtime.set(FakeRuntime())
        try:
            handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        finally:
            _current_runtime.reset(token)

        span = handler._spans[str(run_id)]
        assert "FakeRuntime" in span.attributes.get("chain.runtime", "")

    def test_chain_start_no_runtime_attribute_when_unset(self, tracer_and_trace):
        from agent_trace.integrations.langgraph import _current_runtime

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()

        token = _current_runtime.set(None)
        try:
            handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        finally:
            _current_runtime.reset(token)

        span = handler._spans[str(run_id)]
        assert "chain.runtime" not in span.attributes


class TestExceptionClassification:
    """error.origin + error.known_pattern — applied to any span closing ERROR
    via _close_span_with_exception (on_chain_error/on_llm_error/on_tool_error
    all funnel through it)."""

    def test_classify_origin_provider(self):
        from agent_trace.integrations.langgraph import _classify_exception_origin

        class FakeError(Exception):
            pass

        FakeError.__module__ = "openai._exceptions"
        assert _classify_exception_origin(FakeError("boom")) == "provider"

    def test_classify_origin_chain(self):
        from agent_trace.integrations.langgraph import _classify_exception_origin

        class FakeError(Exception):
            pass

        FakeError.__module__ = "langgraph.errors"
        assert _classify_exception_origin(FakeError("boom")) == "chain"

    def test_classify_origin_application_default(self):
        from agent_trace.integrations.langgraph import _classify_exception_origin

        class FakeError(Exception):
            pass

        FakeError.__module__ = "my_app.agents"
        assert _classify_exception_origin(FakeError("boom")) == "application"

    def test_match_known_error_signature_invalid_chat_history(self):
        from agent_trace.integrations.langgraph import _match_known_error_signature

        msg = "ErrorCode.INVALID_CHAT_HISTORY: messages must alternate roles"
        assert (
            _match_known_error_signature(msg) == "langgraph_invalid_chat_history"
        )

    def test_match_known_error_signature_invalid_tool_selection(self):
        from agent_trace.integrations.langgraph import _match_known_error_signature

        msg = "Selected invalid tool(s): frobulate. Available: search, math."
        assert (
            _match_known_error_signature(msg) == "middleware_invalid_tool_selection"
        )

    def test_match_known_error_signature_no_match(self):
        from agent_trace.integrations.langgraph import _match_known_error_signature

        assert _match_known_error_signature("some unrelated failure") is None

    def test_match_known_error_signature_empty_message(self):
        from agent_trace.integrations.langgraph import _match_known_error_signature

        assert _match_known_error_signature("") is None

    def test_chain_error_sets_origin_attribute(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(ValueError("plain application bug"), run_id=run_id)
        assert span_ref.attributes.get("error.origin") == "application"

    def test_chain_error_sets_known_pattern_when_matched(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(
            ValueError("Selected invalid tool(s): foo"), run_id=run_id
        )
        assert (
            span_ref.attributes.get("error.known_pattern")
            == "middleware_invalid_tool_selection"
        )

    def test_chain_error_no_known_pattern_attribute_when_unmatched(
        self, tracer_and_trace
    ):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(ValueError("totally unrelated"), run_id=run_id)
        assert "error.known_pattern" not in span_ref.attributes

    def test_tool_error_also_classified(self, tracer_and_trace):
        """_close_span_with_exception is shared — tool spans get the same
        classification as chain spans."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "t"}, "in", run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_tool_error(RuntimeError("tool broke"), run_id=run_id)
        assert span_ref.attributes.get("error.origin") == "application"


class TestControlFlowSignalHandling:
    """Command/ParentCommand handoff jumps and GraphInterrupt pauses must
    close OK with an informational attribute, not ERROR — verified against
    the real langgraph package's exception types in
    tests/integration/test_langgraph.py. These unit tests inject a fake type
    into the module-level cache so the behavior is exercised even when the
    real langgraph package's types aren't the ones under test."""

    @pytest.fixture()
    def fake_control_flow_type(self, monkeypatch):
        import agent_trace.integrations.langgraph as lg_module

        class FakeParentCommand(BaseException):
            pass

        monkeypatch.setattr(
            lg_module, "_control_flow_exception_types", (FakeParentCommand,)
        )
        return FakeParentCommand

    def test_control_flow_signal_closes_span_ok(
        self, tracer_and_trace, fake_control_flow_type
    ):
        from agent_trace.core.span import SpanStatus

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "handoff"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(fake_control_flow_type("jump"), run_id=run_id)
        assert span_ref.status == SpanStatus.OK

    def test_control_flow_signal_sets_handoff_attribute(
        self, tracer_and_trace, fake_control_flow_type
    ):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "handoff"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(fake_control_flow_type("jump"), run_id=run_id)
        assert span_ref.attributes.get("langgraph.handoff") is True
        assert (
            span_ref.attributes.get("langgraph.control_flow_signal")
            == "FakeParentCommand"
        )

    def test_graph_interrupt_sets_interrupted_attribute_not_handoff(
        self, tracer_and_trace, monkeypatch
    ):
        import agent_trace.integrations.langgraph as lg_module

        class GraphInterrupt(BaseException):
            pass

        monkeypatch.setattr(
            lg_module, "_control_flow_exception_types", (GraphInterrupt,)
        )

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "pause"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(GraphInterrupt(), run_id=run_id)
        assert span_ref.attributes.get("langgraph.interrupted") is True
        assert "langgraph.handoff" not in span_ref.attributes

    def test_control_flow_signal_does_not_set_error_origin(
        self, tracer_and_trace, fake_control_flow_type
    ):
        """Control-flow signals are not classified as errors at all — they
        never reach the error.origin/known_pattern classification path."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "handoff"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(fake_control_flow_type("jump"), run_id=run_id)
        assert "error.origin" not in span_ref.attributes

    def test_genuine_error_still_marked_error_when_not_control_flow(
        self, tracer_and_trace, fake_control_flow_type
    ):
        """A ValueError (not the injected fake control-flow type) is
        unaffected by the fake-type patch and still closes ERROR."""
        from agent_trace.core.span import SpanStatus

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(ValueError("real bug"), run_id=run_id)
        assert span_ref.status == SpanStatus.ERROR


class TestCancelledStatus:
    """asyncio.CancelledError must close a span CANCELLED, not ERROR."""

    def test_chain_error_cancelled_error_sets_cancelled_status(
        self, tracer_and_trace
    ):
        import asyncio

        from agent_trace.core.span import SpanStatus

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(asyncio.CancelledError(), run_id=run_id)
        assert span_ref.status == SpanStatus.CANCELLED

    def test_cancelled_status_distinct_from_error_status(self, tracer_and_trace):
        import asyncio

        from agent_trace.core.span import SpanStatus

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)

        run_id_cancelled = _run_id()
        handler.on_chain_start({"name": "n1"}, {}, run_id=run_id_cancelled)
        span_cancelled = handler._spans[str(run_id_cancelled)]
        handler.on_chain_error(asyncio.CancelledError(), run_id=run_id_cancelled)

        run_id_error = _run_id()
        handler.on_chain_start({"name": "n2"}, {}, run_id=run_id_error)
        span_error = handler._spans[str(run_id_error)]
        handler.on_chain_error(RuntimeError("real failure"), run_id=run_id_error)

        assert span_cancelled.status == SpanStatus.CANCELLED
        assert span_error.status == SpanStatus.ERROR
        assert span_cancelled.status != span_error.status

    def test_cancelled_error_still_records_exception_event(self, tracer_and_trace):
        import asyncio

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(asyncio.CancelledError(), run_id=run_id)
        exception_events = [e for e in span_ref.events if e.name == "exception"]
        assert len(exception_events) == 1
        assert exception_events[0].attributes["exception.type"] == "CancelledError"


class TestSerializationRobustness:
    """Regression coverage for the recursive-Mock hang: _deep_serialize must
    terminate quickly even against objects whose model_dump()/dict()/
    attribute access always yields a brand-new child object (the exact shape
    of an unconfigured unittest.mock.MagicMock)."""

    def test_to_attr_string_terminates_on_self_generating_mock(self):
        import time

        from agent_trace.integrations.langgraph import _to_attr_string

        start = time.monotonic()
        result = _to_attr_string(MagicMock())
        elapsed = time.monotonic() - start

        assert isinstance(result, str)
        assert elapsed < 5.0, f"serialization took {elapsed:.2f}s — recursion regressed"

    def test_to_attr_string_bounds_deeply_nested_dicts(self):
        nested: dict = {"v": 0}
        cursor = nested
        for i in range(1, 50):
            cursor["next"] = {"v": i}
            cursor = cursor["next"]

        from agent_trace.integrations.langgraph import _to_attr_string

        result = _to_attr_string(nested)
        assert isinstance(result, str)
        assert len(result) < 10_000

    def test_to_attr_string_handles_circular_reference(self):
        from agent_trace.integrations.langgraph import _to_attr_string

        circular: dict = {"a": 1}
        circular["self"] = circular

        result = _to_attr_string(circular)
        assert isinstance(result, str)
        assert "circular-reference" in result

    def test_llm_end_with_recursive_mock_response_does_not_hang(self, tracer_and_trace):
        """End-to-end regression test: on_llm_end must not hang when given a
        bare MagicMock response (no explicit .generations/.message set up)."""
        import time

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "gpt-4"}, [[]], run_id=run_id)

        response = MagicMock()
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        }
        # response.generations is left as an auto-generated MagicMock
        # attribute (not explicitly configured) — this is exactly the shape
        # that previously caused unbounded recursion in the serializer.

        start = time.monotonic()
        handler.on_llm_end(response, run_id=run_id)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"on_llm_end took {elapsed:.2f}s — recursion regressed"
        assert str(run_id) not in handler._spans


# ---------------------------------------------------------------------------
# LangGraph 1.x compatibility — serialized=None, name in kwargs
# ---------------------------------------------------------------------------


class TestLangGraph1xSerializedNone:
    """LangGraph 1.x passes serialized=None; name arrives in **kwargs.

    These tests guard against the AttributeError crash introduced in that
    API change and verify that node/tool names are resolved from kwargs.
    """

    def test_chain_start_serialized_none_does_not_crash(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        # LangGraph 1.x calling pattern: serialized=None, name="node_a" in kwargs
        handler.on_chain_start(None, {}, run_id=run_id, name="node_a")  # type: ignore[arg-type]
        assert str(run_id) in handler._spans

    def test_chain_start_serialized_none_uses_kwargs_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start(None, {}, run_id=run_id, name="node_a")  # type: ignore[arg-type]
        span = handler._spans[str(run_id)]
        assert span.name == "node:node_a"
        assert span.attributes.get("langgraph.node") == "node_a"

    def test_chain_start_serialized_none_fallback_to_chain(self, tracer_and_trace):
        """No name anywhere → fall back to 'chain'."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start(None, {}, run_id=run_id)  # type: ignore[arg-type]
        span = handler._spans[str(run_id)]
        assert span.name == "node:chain"

    def test_llm_start_serialized_none_does_not_crash(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start(None, ["prompt"], run_id=run_id, name="gpt-4o")  # type: ignore[arg-type]
        assert str(run_id) in handler._spans

    def test_llm_start_serialized_none_uses_kwargs_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start(None, ["prompt"], run_id=run_id, name="gpt-4o")  # type: ignore[arg-type]
        span = handler._spans[str(run_id)]
        assert span.name == "llm:gpt-4o"

    def test_chat_model_start_serialized_none_does_not_crash(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(None, [[]], run_id=run_id, name="ChatOpenAI")  # type: ignore[arg-type]
        assert str(run_id) in handler._spans

    def test_chat_model_start_serialized_none_uses_kwargs_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(None, [[]], run_id=run_id, name="ChatOpenAI")  # type: ignore[arg-type]
        span = handler._spans[str(run_id)]
        assert span.name == "llm:ChatOpenAI"
        assert span.attributes.get("llm.model") == "ChatOpenAI"

    def test_tool_start_serialized_none_does_not_crash(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start(None, "query", run_id=run_id, name="web_search")  # type: ignore[arg-type]
        assert str(run_id) in handler._spans

    def test_tool_start_serialized_none_uses_kwargs_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start(None, "query", run_id=run_id, name="web_search")  # type: ignore[arg-type]
        span = handler._spans[str(run_id)]
        assert span.name == "tool:web_search"
        assert span.attributes.get("tool.name") == "web_search"


# ---------------------------------------------------------------------------
# Streaming callback hooks — on_llm_new_token
# ---------------------------------------------------------------------------


class TestStreamingCallbackHooks:
    """on_llm_new_token: per-token/per-delta streaming chunks land on the
    span instead of being silently discarded."""

    def test_new_token_records_span_event(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id=run_id)
        handler.on_llm_new_token("hel", run_id=run_id)
        handler.on_llm_new_token("lo", run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.attributes.get("llm.streamed") is True
        assert span.attributes.get("llm.stream_token_count") == 2
        stream_events = [e for e in span.events if e.name == "llm_stream_delta"]
        assert [e.attributes["token"] for e in stream_events] == ["hel", "lo"]
        assert [e.attributes["stream.index"] for e in stream_events] == [0, 1]

    def test_new_token_unknown_run_id_is_noop(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        handler.on_llm_new_token("tok", run_id=_run_id())  # must not raise

    def test_new_token_captures_tool_call_chunks(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id=run_id)

        class FakeMessage:
            def __init__(self) -> None:
                self.tool_call_chunks = [
                    {"name": "get_weather", "args": '{"city"', "index": 0}
                ]

        class FakeChunk:
            def __init__(self) -> None:
                self.message = FakeMessage()

        handler.on_llm_new_token("", chunk=FakeChunk(), run_id=run_id)
        span = handler._spans[str(run_id)]
        events = [e for e in span.events if e.name == "llm_stream_delta"]
        assert "get_weather" in events[0].attributes["tool_call_chunks"]

    def test_stream_token_count_keeps_counting_past_event_cap(self, tracer_and_trace):
        """SpanEvents stop being appended past the cap, but the running
        count attribute keeps counting every token."""
        import agent_trace.integrations.langgraph as lg_module

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id=run_id)
        total = lg_module._MAX_STREAM_EVENTS_PER_SPAN + 5
        for _ in range(total):
            handler.on_llm_new_token("x", run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.attributes.get("llm.stream_token_count") == total
        stream_events = [e for e in span.events if e.name == "llm_stream_delta"]
        assert len(stream_events) == lg_module._MAX_STREAM_EVENTS_PER_SPAN

    def test_stream_token_counter_cleared_on_span_close(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id=run_id)
        handler.on_llm_new_token("hi", run_id=run_id)
        handler.on_llm_end(MagicMock(generations=[]), run_id=run_id)
        assert str(run_id) not in handler._stream_token_counts


# ---------------------------------------------------------------------------
# Node-level declared tags — captured at graph-construction time
# ---------------------------------------------------------------------------


class TestDeclaredNodeTagsCapture:
    """on_chain_start: a compiled graph's node-level *declared* tags (from
    .with_config(tags=[...]) at construction time) land on the span when a
    graph= is supplied to LangGraphTracer — distinct from the runtime tags
    callback kwarg, which never carries them."""

    def _fake_graph(self, node_name: str, tags: list[str] | None):
        class FakeBound:
            def __init__(self) -> None:
                self.config = {"tags": tags} if tags else {}

        class FakeNode:
            def __init__(self) -> None:
                self.bound = FakeBound()

        class FakeGraph:
            def __init__(self) -> None:
                self.nodes = {node_name: FakeNode()}

        return FakeGraph()

    def test_declared_tags_captured_when_graph_supplied(self, tracer_and_trace):
        from agent_trace.integrations.langgraph import LangGraphTracer

        t, trace = tracer_and_trace
        graph = self._fake_graph("my_node", ["nostream"])
        handler = LangGraphTracer(tracer=t, trace=trace, graph=graph)
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.attributes.get("langgraph.declared_tags") == "nostream"

    def test_no_declared_tags_sets_no_attribute(self, tracer_and_trace):
        from agent_trace.integrations.langgraph import LangGraphTracer

        t, trace = tracer_and_trace
        graph = self._fake_graph("my_node", None)
        handler = LangGraphTracer(tracer=t, trace=trace, graph=graph)
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        span = handler._spans[str(run_id)]
        assert "langgraph.declared_tags" not in span.attributes

    def test_no_graph_supplied_sets_no_attribute(self, tracer_and_trace):
        """Default (graph=None) behavior is unchanged — no lookup attempted."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)  # no graph= passed
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)
        span = handler._spans[str(run_id)]
        assert "langgraph.declared_tags" not in span.attributes

    def test_malformed_graph_object_does_not_crash(self, tracer_and_trace):
        """A graph= whose shape doesn't match expectations degrades to 'no
        declared tags', never an exception into the caller's callback."""
        from agent_trace.integrations.langgraph import LangGraphTracer

        t, trace = tracer_and_trace
        handler = LangGraphTracer(tracer=t, trace=trace, graph=object())
        run_id = _run_id()
        handler.on_chain_start({"name": "my_node"}, {}, run_id=run_id)  # must not raise
        span = handler._spans[str(run_id)]
        assert "langgraph.declared_tags" not in span.attributes


# ---------------------------------------------------------------------------
# traced_stream / traced_astream
# ---------------------------------------------------------------------------


class TestTracedStream:
    def test_yields_every_item_unchanged(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_stream

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("stream-test") as trace:
            items = list(traced_stream(t, iter(["a", "b", "c"])))
        assert items == ["a", "b", "c"]

    def test_records_stream_yield_events_with_index(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_stream

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("stream-test") as trace:
            list(traced_stream(t, iter(["a", "b"])))
        span = next(s for s in trace.spans if s.name == "graph:stream")
        events = [e for e in span.events if e.name == "stream_yield"]
        assert [e.attributes["stream.index"] for e in events] == [0, 1]
        assert span.attributes.get("stream.chunk_count") == 2
        assert span.status.value == "OK"

    def test_span_name_is_customizable(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_stream

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("stream-test") as trace:
            list(traced_stream(t, iter(["a"]), span_name="graph:stream:custom"))
        assert any(s.name == "graph:stream:custom" for s in trace.spans)

    def test_exception_in_source_stream_closes_span_error(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_stream

        def bad_stream():
            yield "a"
            raise RuntimeError("boom")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("stream-test") as trace, pytest.raises(RuntimeError):
            list(traced_stream(t, bad_stream()))
        span = next(s for s in trace.spans if s.name == "graph:stream")
        assert span.status.value == "ERROR"
        assert any(e.name == "exception" for e in span.events)

    def test_early_abandonment_closes_span_cancelled(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_stream

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("stream-test") as trace:
            gen = traced_stream(t, iter(["a", "b", "c"]))
            next(gen)
            gen.close()
        span = next(s for s in trace.spans if s.name == "graph:stream")
        assert span.status.value == "CANCELLED"

    def test_chunk_content_is_captured(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_stream

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("stream-test") as trace:
            list(traced_stream(t, iter([{"messages": ["hi"]}])))
        span = next(s for s in trace.spans if s.name == "graph:stream")
        event = next(e for e in span.events if e.name == "stream_yield")
        assert "hi" in event.attributes["stream.chunk"]


class TestTracedAstream:
    async def test_yields_every_item_unchanged(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_astream

        async def source():
            for x in ["a", "b"]:
                yield x

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("astream-test") as trace:
            items = [x async for x in traced_astream(t, source())]
        assert items == ["a", "b"]
        span = next(s for s in trace.spans if s.name == "graph:astream")
        assert span.attributes.get("stream.chunk_count") == 2
        assert span.status.value == "OK"

    async def test_exception_in_source_stream_closes_span_error(self, tmp_path):
        from agent_trace.integrations.langgraph import traced_astream

        async def bad_source():
            yield "a"
            raise RuntimeError("boom")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("astream-test") as trace, pytest.raises(RuntimeError):
            async for _ in traced_astream(t, bad_source()):
                pass
        span = next(s for s in trace.spans if s.name == "graph:astream")
        assert span.status.value == "ERROR"


# ---------------------------------------------------------------------------
# derive_trace_id — deterministic trace_id from LangGraph thread_id/checkpoint
# identity (issue #7417). Pure function, no langchain_core/langgraph needed.
# ---------------------------------------------------------------------------


class TestDeriveTraceId:
    def test_deterministic_for_same_thread_id(self):
        from agent_trace.integrations.langgraph import derive_trace_id

        assert derive_trace_id("thread-1") == derive_trace_id("thread-1")

    def test_different_thread_ids_produce_different_ids(self):
        from agent_trace.integrations.langgraph import derive_trace_id

        assert derive_trace_id("thread-1") != derive_trace_id("thread-2")

    def test_checkpoint_id_changes_the_result(self):
        from agent_trace.integrations.langgraph import derive_trace_id

        assert derive_trace_id("t1", "cp1") == derive_trace_id("t1", "cp1")
        assert derive_trace_id("t1", "cp1") != derive_trace_id("t1", "cp2")
        assert derive_trace_id("t1") != derive_trace_id("t1", "cp1")

    def test_returns_32_char_hex_string(self):
        from agent_trace.integrations.langgraph import derive_trace_id

        value = derive_trace_id("thread-1")
        assert len(value) == 32
        int(value, 16)  # raises ValueError if not valid hex

    def test_wired_into_start_trace_trace_id_param(self, tmp_path: Path):
        """Tracer.start_trace(trace_id=...) actually uses the derived id."""
        from agent_trace.integrations.langgraph import derive_trace_id

        derived = derive_trace_id("thread-42")
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("derived-trace-id-test", trace_id=derived) as trace:
            assert trace.trace_id == derived


# ---------------------------------------------------------------------------
# long_span_threshold_secs — flags a span at close time once its measured
# open duration crosses a configurable threshold (issue #7417). Uses
# FixtureClock for deterministic elapsed-time control instead of real sleeps.
# ---------------------------------------------------------------------------


class TestLongRunningSpanThreshold:
    def test_span_under_threshold_not_flagged(self, tracer_and_trace):
        from agent_trace.core.clock import FixtureClock, restore_clock, set_clock
        from agent_trace.integrations.langgraph import LangGraphTracer

        t, trace = tracer_and_trace
        handler = LangGraphTracer(
            tracer=t, trace=trace, long_span_threshold_secs=180
        )
        clock = FixtureClock(initial=1_000.0)
        token = set_clock(clock)
        try:
            run_id = _run_id()
            handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
            span_ref = handler._spans[str(run_id)]
            clock.advance(1_010.0)  # 10s elapsed — well under 180s
            handler.on_chain_end({}, run_id=run_id)
        finally:
            restore_clock(token)
        assert "span.exceeded_long_running_threshold" not in span_ref.attributes

    def test_span_over_threshold_is_flagged(self, tracer_and_trace):
        from agent_trace.core.clock import FixtureClock, restore_clock, set_clock
        from agent_trace.integrations.langgraph import LangGraphTracer

        t, trace = tracer_and_trace
        handler = LangGraphTracer(
            tracer=t, trace=trace, long_span_threshold_secs=180
        )
        clock = FixtureClock(initial=1_000.0)
        token = set_clock(clock)
        try:
            run_id = _run_id()
            handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
            span_ref = handler._spans[str(run_id)]
            clock.advance(1_200.0)  # 200s elapsed — over the 180s threshold
            handler.on_chain_end({}, run_id=run_id)
        finally:
            restore_clock(token)
        assert span_ref.attributes.get("span.exceeded_long_running_threshold") is True
        assert span_ref.attributes.get("span.long_running_threshold_secs") == 180
        assert span_ref.attributes.get("span.duration_secs_at_close") == 200.0

    def test_disabled_by_default(self, tracer_and_trace):
        from agent_trace.core.clock import FixtureClock, restore_clock, set_clock

        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)  # no long_span_threshold_secs
        clock = FixtureClock(initial=1_000.0)
        token = set_clock(clock)
        try:
            run_id = _run_id()
            handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
            span_ref = handler._spans[str(run_id)]
            clock.advance(10_000.0)  # huge elapsed — still not flagged
            handler.on_chain_end({}, run_id=run_id)
        finally:
            restore_clock(token)
        assert "span.exceeded_long_running_threshold" not in span_ref.attributes

    def test_error_span_also_checked(self, tracer_and_trace):
        from agent_trace.core.clock import FixtureClock, restore_clock, set_clock
        from agent_trace.integrations.langgraph import LangGraphTracer

        t, trace = tracer_and_trace
        handler = LangGraphTracer(
            tracer=t, trace=trace, long_span_threshold_secs=180
        )
        clock = FixtureClock(initial=1_000.0)
        token = set_clock(clock)
        try:
            run_id = _run_id()
            handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
            span_ref = handler._spans[str(run_id)]
            clock.advance(1_200.0)
            handler.on_chain_error(ValueError("boom"), run_id=run_id)
        finally:
            restore_clock(token)
        assert span_ref.attributes.get("span.exceeded_long_running_threshold") is True
