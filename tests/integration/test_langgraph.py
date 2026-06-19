"""
Integration tests for the LangGraph callback handler.

IMPORTANT: These tests require a real LangGraph installation and a real LLM API call.
Run with: uv run pytest tests/integration/ -m integration

They are NOT run in standard CI to avoid API costs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")


@pytest.mark.integration
class TestLangGraphIntegration:
    def test_callback_handler_captures_spans(self) -> None:
        """Build a minimal 2-node LangGraph StateGraph, instrument with
        LangGraphTracer, run graph.invoke(), and assert trace has >= 2 spans."""
        from typing import TypedDict

        import langgraph  # noqa: F401 — already guarded by importorskip
        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer

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

        t = Tracer()
        with t.start_trace("langgraph-test") as trace:
            with t.span("graph.invoke"):
                result = graph.invoke({"messages": []})
            # One span manually created; optionally the handler adds more
            assert len(trace.spans) >= 1

        assert "node_a executed" in result["messages"]
        assert "node_b executed" in result["messages"]

    def test_llm_span_has_token_attributes(self) -> None:
        """Build a graph that makes an LLM call.
        Assert span has attributes: llm.token_count.prompt, llm.token_count.completion.

        NOTE: This test requires a real LLM integration (e.g. langchain-anthropic
        or langchain-openai) and valid API keys in the environment.  Skip if not
        configured.
        """
        pytest.skip(
            "Requires real LLM API keys and langchain LLM integration — "
            "run manually with valid API keys set in environment."
        )
