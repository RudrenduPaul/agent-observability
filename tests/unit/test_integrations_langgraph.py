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

    yield fakes

    # Restore the cached class so other tests (real langchain, integration) are
    # not affected by the reset.
    lg_module._LangGraphTracerClass = original


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

    def test_unknown_run_id_in_end_is_noop(self, tracer_and_trace):
        """Closing a span that was never opened must not raise."""
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        phantom_id = _run_id()
        handler.on_chain_end({}, run_id=phantom_id)  # must not raise
        handler.on_tool_end("x", run_id=phantom_id)  # must not raise


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
