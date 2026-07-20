"""
Unit tests for agent_trace.integrations.langchain_core.LangChainTracer —
the generic, framework-agnostic BaseCallbackHandler for arbitrary LangChain
Runnable calls (not just LangGraph nodes).

langchain_core is NOT required to be installed for these tests to run —
they mock the BaseCallbackHandler base class, the same convention
test_integrations_langgraph.py uses.
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path
from types import ModuleType

import pytest

from agent_trace import SpanStatus, Tracer

# ---------------------------------------------------------------------------
# Fake langchain_core fixture (module-level injection)
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

    import agent_trace.integrations.langchain_core as lc_module

    original = lc_module._LangChainTracerClass
    lc_module._LangChainTracerClass = None

    yield fakes

    lc_module._LangChainTracerClass = original


@pytest.fixture()
def tracer_and_trace(tmp_path: Path, patched_langchain):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("lc-unit-test") as trace:
        yield t, trace


def _make_handler(t, trace):
    from agent_trace.integrations.langchain_core import LangChainTracer

    return LangChainTracer(tracer=t, trace=trace)


def _run_id() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# chain callbacks
# ---------------------------------------------------------------------------


class TestChainCallbacks:
    def test_chain_start_registers_span(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "MyChain"}, {}, run_id=run_id)
        assert str(run_id) in handler._spans
        span = handler._spans[str(run_id)]
        assert span.name == "chain:MyChain"

    def test_chain_start_uses_name_kwarg_when_serialized_missing(
        self, tracer_and_trace
    ):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start(None, {}, run_id=run_id, name="FromKwarg")
        span = handler._spans[str(run_id)]
        assert span.name == "chain:FromKwarg"

    def test_chain_start_records_inputs(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "c"}, {"query": "hi"}, run_id=run_id)
        span = handler._spans[str(run_id)]
        assert "hi" in span.attributes.get("chain.inputs", "")

    def test_chain_end_closes_span_ok_and_records_outputs(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "c"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_end({"result": 42}, run_id=run_id)
        assert str(run_id) not in handler._spans
        assert span_ref.status == SpanStatus.OK
        assert "42" in span_ref.attributes.get("chain.outputs", "")

    def test_chain_error_closes_span_error_and_records_exception(
        self, tracer_and_trace
    ):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chain_start({"name": "c"}, {}, run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_chain_error(IndexError("boom in _parse_ranking"), run_id=run_id)
        assert str(run_id) not in handler._spans
        assert span_ref.status == SpanStatus.ERROR
        assert any(
            e.name == "exception"
            and "boom in _parse_ranking" in e.attributes.get("exception.message", "")
            for e in span_ref.events
        )

    def test_nested_chain_parent_child_wiring(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        parent_id = _run_id()
        child_id = _run_id()
        handler.on_chain_start({"name": "parent"}, {}, run_id=parent_id)
        parent_span = handler._spans[str(parent_id)]
        handler.on_chain_start(
            {"name": "child"}, {}, run_id=child_id, parent_run_id=parent_id
        )
        child_span = handler._spans[str(child_id)]
        assert child_span.parent_id == parent_span.span_id


# ---------------------------------------------------------------------------
# LLM callbacks
# ---------------------------------------------------------------------------


class TestLlmCallbacks:
    def test_llm_start_registers_span_with_model_name(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start({"kwargs": {"model": "gpt-4o"}}, ["hi"], run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.name == "llm:gpt-4o"
        assert span.attributes.get("llm.model") == "gpt-4o"

    def test_chat_model_start_records_messages(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_chat_model_start(
            {"kwargs": {"model_name": "claude-3"}}, [["hello"]], run_id=run_id
        )
        span = handler._spans[str(run_id)]
        assert span.name == "llm:claude-3"
        assert "hello" in span.attributes.get("llm.messages", "")

    def test_llm_end_closes_span_ok(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start({"kwargs": {}}, ["hi"], run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_llm_end({}, run_id=run_id)
        assert str(run_id) not in handler._spans
        assert span_ref.status == SpanStatus.OK

    def test_llm_error_closes_span_error(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_llm_start({"kwargs": {}}, ["hi"], run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_llm_error(RuntimeError("rate limited"), run_id=run_id)
        assert span_ref.status == SpanStatus.ERROR


# ---------------------------------------------------------------------------
# tool callbacks
# ---------------------------------------------------------------------------


class TestToolCallbacks:
    def test_tool_start_and_end_round_trip(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "query text", run_id=run_id)
        span = handler._spans[str(run_id)]
        assert span.name == "tool:search"
        assert "query text" in span.attributes.get("tool.input", "")
        handler.on_tool_end("result text", run_id=run_id)
        assert str(run_id) not in handler._spans
        assert span.status == SpanStatus.OK
        assert "result text" in span.attributes.get("tool.output", "")

    def test_tool_error_closes_span_error(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        run_id = _run_id()
        handler.on_tool_start({"name": "search"}, "q", run_id=run_id)
        span_ref = handler._spans[str(run_id)]
        handler.on_tool_error(ValueError("bad input"), run_id=run_id)
        assert span_ref.status == SpanStatus.ERROR


# ---------------------------------------------------------------------------
# Misc robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_unknown_run_id_in_end_is_noop(self, tracer_and_trace):
        t, trace = tracer_and_trace
        handler = _make_handler(t, trace)
        phantom_id = _run_id()
        handler.on_chain_end({}, run_id=phantom_id)  # must not raise
        handler.on_tool_end("x", run_id=phantom_id)  # must not raise

    def test_missing_langchain_core_raises_clear_import_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "langchain_core":
                raise ImportError("no module named langchain_core")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        import agent_trace.integrations.langchain_core as lc_module

        original = lc_module._LangChainTracerClass
        lc_module._LangChainTracerClass = None
        try:
            with pytest.raises(ImportError, match="pip install agent-observability-trace-cli"):
                lc_module._get_tracer_class()
        finally:
            lc_module._LangChainTracerClass = original
