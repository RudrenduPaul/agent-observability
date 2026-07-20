"""
Integration tests for agent_trace.integrations.langchain_core.LangChainTracer
against a real (installed) langchain_core — no LangGraph anywhere in these
tests, deliberately: this integration exists specifically for plain
Runnable usage outside a LangGraph graph.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")


@pytest.mark.integration
class TestLangChainTracerIntegration:
    def test_bare_runnable_lambda_pipeline_produces_nested_spans(
        self, tmp_path: Path
    ) -> None:
        from langchain_core.runnables import RunnableLambda

        from agent_trace import Tracer
        from agent_trace.integrations.langchain_core import LangChainTracer

        def step_a(x: int) -> int:
            return x + 1

        def step_b(x: int) -> int:
            return x * 2

        chain = RunnableLambda(step_a) | RunnableLambda(step_b)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("plain-runnable") as trace:
            cb = LangChainTracer(tracer=t, trace=trace)
            result = chain.invoke(1, config={"callbacks": [cb]})

        assert result == 4  # (1 + 1) * 2
        names = [s.name for s in trace.spans]
        assert any(n.startswith("chain:") for n in names)

    def test_application_error_inside_bare_runnable_is_captured(
        self, tmp_path: Path
    ) -> None:
        """#31192's exact shape: an application-level exception raised
        inside a plain Runnable's own Python code (not during an HTTP
        call) — must land on an ERROR span with the exception captured,
        not vanish entirely."""
        from langchain_core.runnables import RunnableLambda

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langchain_core import LangChainTracer

        def step_a(x: int) -> int:
            return x + 1

        def _parse_ranking(x: int) -> int:
            raise IndexError("list index out of range")

        chain = RunnableLambda(step_a) | RunnableLambda(_parse_ranking)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("plain-runnable-error") as trace:
            cb = LangChainTracer(tracer=t, trace=trace)
            with pytest.raises(IndexError):
                chain.invoke(1, config={"callbacks": [cb]})

        error_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        assert error_spans, (
            f"expected at least one ERROR span, got: "
            f"{[(s.name, s.status) for s in trace.spans]}"
        )
        assert any(
            "list index out of range" in e.attributes.get("exception.message", "")
            for s in error_spans
            for e in s.events
            if e.name == "exception"
        )

    def test_all_spans_closed_after_clean_run(self, tmp_path: Path) -> None:
        from langchain_core.runnables import RunnableLambda

        from agent_trace import Tracer
        from agent_trace.integrations.langchain_core import LangChainTracer

        chain = RunnableLambda(lambda x: x + 1)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("clean-run") as trace:
            cb = LangChainTracer(tracer=t, trace=trace)
            chain.invoke(1, config={"callbacks": [cb]})

        assert trace.spans
        for span in trace.spans:
            assert span.end_time is not None, f"{span.name} was left open"

    def test_real_http_call_inside_runnable_recorded_and_captured_on_span(
        self, tmp_path: Path
    ) -> None:
        """A plain Runnable that makes a real HTTP call (via a
        RecordingTransport-patched httpx.Client) — verifies this
        integration composes correctly with the framework-agnostic
        interceptor layer, the actual #31192/#31227 population's shape."""
        import httpx
        from langchain_core.runnables import RunnableLambda

        from agent_trace import Tracer
        from agent_trace.integrations.langchain_core import LangChainTracer

        def call_api(x: str) -> str:
            client = httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"echo": x})
                )
            )
            response = client.get(f"https://api.example.com/echo/{x}")
            return response.text

        chain = RunnableLambda(call_api)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace(
            "runnable-http-call", record=True, run_id="lc-http-run"
        ) as trace:
            cb = LangChainTracer(tracer=t, trace=trace)
            chain.invoke("hello", config={"callbacks": [cb]})

        from agent_trace._replay.fixture import Fixture

        with Fixture(tmp_path / "lc-http-run" / "fixture.db") as fixture:
            exchanges = fixture.all_exchanges()
        assert len(exchanges) == 1
        assert exchanges[0]["url"] == "https://api.example.com/echo/hello"

    def test_classic_chain_memory_persistence_error_captured(
        self, tmp_path: Path
    ) -> None:
        """#6761: an exception raised during a classic (non-LangGraph)
        Chain's post-execution memory-persistence step (BaseMemory.
        save_context, called from Chain.prep_outputs) must land on an
        ERROR span, not vanish invisibly.

        The original #6761 investigation (against langchain==0.0.215) found
        prep_outputs() ran *outside* Chain.invoke()'s try/except, so
        on_chain_error never fired for this failure at all. Verified here
        against the current langchain-classic package (the officially
        maintained extraction of the legacy chains/memory API — modern
        `langchain` no longer ships `langchain.chains` at all) that this is
        no longer the case: invoke()'s try/except now wraps prep_outputs()
        too, so on_chain_error *does* fire — meaning LangChainTracer (this
        module) already closes this gap with no dedicated Chain/BaseMemory
        module needed. This test pins that fixed behavior as a regression
        guard.
        """
        pytest.importorskip(
            "langchain_classic", reason="langchain-classic not installed"
        )
        from typing import Any

        from langchain_classic.base_memory import BaseMemory
        from langchain_classic.chains.base import Chain

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langchain_core import LangChainTracer

        class _BrokenMemory(BaseMemory):
            @property
            def memory_variables(self) -> list[str]:
                return []

            def load_memory_variables(self, inputs: dict) -> dict:
                return {}

            def save_context(self, inputs: dict, outputs: dict) -> None:
                raise RuntimeError("stale None instructions boom")

            def clear(self) -> None:
                pass

        class _SimpleChain(Chain):
            memory: Any = None

            @property
            def input_keys(self) -> list[str]:
                return ["question"]

            @property
            def output_keys(self) -> list[str]:
                return ["answer"]

            def _call(self, inputs: dict, run_manager: Any = None) -> dict:
                return {"answer": "42"}

        chain = _SimpleChain(memory=_BrokenMemory())

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("chain-memory-error") as trace:
            cb = LangChainTracer(tracer=t, trace=trace)
            with pytest.raises(RuntimeError, match="stale None instructions boom"):
                chain.invoke({"question": "hi"}, config={"callbacks": [cb]})

        error_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        assert error_spans, "the memory-persistence exception was not captured"
        assert any(
            "stale None instructions boom" in e.attributes.get("exception.message", "")
            for s in error_spans
            for e in s.events
            if e.name == "exception"
        )

    def test_classic_agent_executor_produces_nested_spans(self, tmp_path: Path) -> None:
        """#22358: a classic (non-LangGraph) `langchain.agents.AgentExecutor`
        ReAct agent — no framework integration existed for this population
        before LangChainTracer. AgentExecutor is itself a Chain subclass,
        so it goes through the exact same callback machinery verified
        above; this test exercises the real class directly."""
        pytest.importorskip(
            "langchain_classic", reason="langchain-classic not installed"
        )
        from langchain_classic.agents import AgentType, initialize_agent
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool

        from agent_trace import Tracer
        from agent_trace.integrations.langchain_core import LangChainTracer

        @tool
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"sunny in {city}"

        class _FakeReActModel(BaseChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content="Final Answer: sunny"))
                    ]
                )

            @property
            def _llm_type(self) -> str:
                return "fake-react-model"

        agent = initialize_agent(
            tools=[get_weather],
            llm=_FakeReActModel(),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("agent-executor-test") as trace:
            cb = LangChainTracer(tracer=t, trace=trace)
            result = agent.invoke(
                {"input": "what's the weather in Boston?"},
                config={"callbacks": [cb]},
            )

        assert result["output"] == "sunny"
        names = [s.name for s in trace.spans]
        assert "chain:AgentExecutor" in names

    def test_tool_invocation_captures_input_output(self, tmp_path: Path) -> None:
        from langchain_core.tools import tool

        from agent_trace import Tracer
        from agent_trace.integrations.langchain_core import LangChainTracer

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-call") as trace:
            cb = LangChainTracer(tracer=t, trace=trace)
            add.invoke({"a": 2, "b": 3}, config={"callbacks": [cb]})

        tool_span = next(s for s in trace.spans if s.name.startswith("tool:"))
        assert "5" in tool_span.attributes.get("tool.output", "")
