"""
Integration tests for the LangGraph callback handler.

These tests require a real LangGraph installation but do NOT require live LLM API
calls — the graphs use pure-Python nodes only.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")


@pytest.mark.integration
class TestLangGraphIntegration:
    def test_callback_handler_captures_node_spans(self, tmp_path: Path) -> None:
        """LangGraphTracer must produce one span per node in a 2-node graph."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class AgentState(TypedDict):
            messages: list[str]

        def node_a(state: AgentState) -> AgentState:
            return {"messages": state["messages"] + ["node_a executed"]}

        def node_b(state: AgentState) -> AgentState:
            return {"messages": state["messages"] + ["node_b executed"]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("node_a", node_a)
        graph_builder.add_node("node_b", node_b)
        graph_builder.set_entry_point("node_a")
        graph_builder.add_edge("node_a", "node_b")
        graph_builder.add_edge("node_b", END)
        graph = graph_builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("langgraph-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            result = graph.invoke(
                {"messages": []},
                config={"callbacks": [cb]},
            )

        assert "node_a executed" in result["messages"]
        assert "node_b executed" in result["messages"]

        node_span_names = [s.name for s in trace.spans]
        assert any("node_a" in name for name in node_span_names), (
            f"Expected a span containing 'node_a', got: {node_span_names}"
        )
        assert any("node_b" in name for name in node_span_names), (
            f"Expected a span containing 'node_b', got: {node_span_names}"
        )

    def test_callback_handler_sets_langgraph_node_attribute(
        self, tmp_path: Path
    ) -> None:
        """Each node span must carry a 'langgraph.node' attribute."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        graph_builder = StateGraph(S)
        graph_builder.add_node("my_node", lambda s: {"x": s["x"] + 1})
        graph_builder.set_entry_point("my_node")
        graph_builder.add_edge("my_node", END)
        graph = graph_builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("attr-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"x": 0}, config={"callbacks": [cb]})

        node_spans = [s for s in trace.spans if "my_node" in s.name]
        assert node_spans, (
            f"No 'my_node' span found. Got: {[s.name for s in trace.spans]}"
        )
        assert node_spans[0].attributes.get("langgraph.node") == "my_node"

    def test_callback_handler_error_span(self, tmp_path: Path) -> None:
        """A node that raises must produce an ERROR span."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def failing_node(state: S) -> S:
            raise RuntimeError("intentional failure")

        graph_builder = StateGraph(S)
        graph_builder.add_node("fail_node", failing_node)
        graph_builder.set_entry_point("fail_node")
        graph_builder.add_edge("fail_node", END)
        graph = graph_builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(RuntimeError, match="intentional failure"):
            with t.start_trace("error-test") as trace:
                cb = LangGraphTracer(tracer=t, trace=trace)
                graph.invoke({"x": 0}, config={"callbacks": [cb]})

        error_spans = [
            s
            for s in trace.spans
            if s.status == SpanStatus.ERROR and "fail_node" in s.name
        ]
        assert error_spans, (
            f"Expected an ERROR span for fail_node. "
            f"Spans: {[(s.name, s.status) for s in trace.spans]}"
        )

    def test_replay_context_allows_pure_python_graph(self, tmp_path: Path) -> None:
        """Record then replay a pure-Python LangGraph graph.

        Pure-Python nodes make no HTTP calls, so AGENT_TRACE_NETWORK_GUARD=1 is
        satisfied automatically.  The replay context installs FixtureClock and
        serves any recorded HTTP exchanges from the fixture (none here).
        """
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer, replay
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            messages: list[str]

        call_count = {"n": 0}

        def counting_node(state: S) -> S:
            call_count["n"] += 1
            return {"messages": state["messages"] + [f"call_{call_count['n']}"]}

        graph_builder = StateGraph(S)
        graph_builder.add_node("counter", counting_node)
        graph_builder.set_entry_point("counter")
        graph_builder.add_edge("counter", END)
        graph = graph_builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace(
            "replay-record", record=True, run_id="lg-replay-run"
        ) as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"messages": []}, config={"callbacks": [cb]})

        assert len(trace.spans) >= 1

        with replay("lg-replay-run", trace_dir=tmp_path):
            result = graph.invoke({"messages": []})

        assert result["messages"]

    def test_llm_span_has_token_attributes(self) -> None:
        """Requires real LLM API keys — skip unless configured manually."""
        pytest.skip(
            "Requires real LLM API keys and langchain LLM integration — "
            "run manually with valid API keys set in environment."
        )
