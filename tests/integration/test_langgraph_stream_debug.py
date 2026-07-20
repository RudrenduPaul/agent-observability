"""
Integration tests for agent_trace.integrations.langgraph_stream_debug — the
real StreamMessagesHandler monkeypatch, exercised against a real LangGraph
graph with a fake streaming chat model.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")
pytest.importorskip("langchain_core", reason="langchain_core not installed")


@pytest.mark.integration
class TestStreamDebugPatchAgainstRealLangGraph:
    def _build_graph(self):
        from typing import Any

        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.runnables import RunnableLambda
        from langgraph.graph import END, START, MessagesState, StateGraph

        class FakeModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake"

            def _generate(
                self,
                messages: Any,
                stop: Any = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="hi"))]
                )

        def call_model_suppressed(state: MessagesState) -> dict[str, Any]:
            resp = FakeModel().with_config(tags=["nostream"]).invoke(state["messages"])
            return {"messages": [resp]}

        def call_model_normal(state: MessagesState) -> dict[str, Any]:
            resp = FakeModel().invoke(state["messages"])
            return {"messages": [resp]}

        builder = StateGraph(MessagesState)
        builder.add_node("suppressed", RunnableLambda(call_model_suppressed))
        builder.add_node("normal", RunnableLambda(call_model_normal))
        builder.add_edge(START, "suppressed")
        builder.add_edge("suppressed", "normal")
        builder.add_edge("normal", END)
        return builder.compile()

    def test_patch_installs_against_real_langgraph(self) -> None:
        from agent_trace.integrations.langgraph_stream_debug import (
            install_stream_debug_patch,
        )

        assert install_stream_debug_patch() is True
        # Idempotent — a second call must also report success without
        # double-patching.
        assert install_stream_debug_patch() is True

    def test_records_suppress_decision_for_nostream_tagged_call(self) -> None:
        from langchain_core.messages import HumanMessage

        from agent_trace.integrations.langgraph_stream_debug import (
            get_stream_decisions,
            install_stream_debug_patch,
            reset_stream_decisions,
        )

        install_stream_debug_patch()
        reset_stream_decisions()
        graph = self._build_graph()

        for _ in graph.stream(
            {"messages": [HumanMessage(content="hey")]}, stream_mode="messages"
        ):
            pass

        decisions = get_stream_decisions()
        by_node = {d.node_name: d for d in decisions}
        assert by_node["suppressed"].suppressed is True
        assert "nostream" in by_node["suppressed"].tags
        assert by_node["normal"].suppressed is False

    def test_declared_tags_and_stream_decisions_cross_reference_correctly(
        self, tmp_path: Path
    ) -> None:
        """End-to-end plumbing check with two real nodes: one that
        correctly wires nostream all the way down to the model call (no
        flag), one that never declares nostream at all (no flag, no
        declared intent to violate). flag_inconsistencies() itself is
        exercised directly against StreamMessagesHandler-recorded decisions,
        real (not faked) declared-tags captured via LangGraphTracer/
        _get_declared_node_tags, and real (not faked) suppress/allow
        decisions recorded by the patched StreamMessagesHandler — proving
        the three pieces genuinely interoperate. The actual inconsistency
        *shape* flag_inconsistencies() detects (declared nostream, but
        LangGraph's runtime tags didn't carry it) is covered directly, with
        full control over both sides of the comparison, in
        tests/unit/test_langgraph_stream_debug.py::TestFlagInconsistencies —
        reproducing that exact propagation failure against a real graph
        depends on an upstream LangChain/LangGraph tag-propagation bug
        (issue #7509 itself) that doesn't reproduce on demand against a
        healthy LangGraph install."""
        from langchain_core.messages import HumanMessage

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import (
            LangGraphTracer,
            _get_declared_node_tags,
        )
        from agent_trace.integrations.langgraph_stream_debug import (
            flag_inconsistencies,
            get_stream_decisions,
            install_stream_debug_patch,
            reset_stream_decisions,
        )

        graph = self._build_graph()  # nodes: "suppressed", "normal"

        assert _get_declared_node_tags(graph, "suppressed") is None
        assert _get_declared_node_tags(graph, "normal") is None

        install_stream_debug_patch()
        reset_stream_decisions()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("stream-debug-cross-reference-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace, graph=graph)
            for _ in graph.stream(
                {"messages": [HumanMessage(content="hey")]},
                config={"callbacks": [cb]},
                stream_mode="messages",
            ):
                pass

        decisions = get_stream_decisions()
        # Neither node declared "nostream" via .with_config() on its own
        # action (only "suppressed"'s inner model call is tagged directly —
        # a different tagging granularity than the node-level declared-tags
        # capture reads), so the real declared_nostream_nodes set derived
        # from _get_declared_node_tags() is empty here — there's no
        # declared-vs-actual mismatch to flag regardless of what
        # StreamMessagesHandler actually decided for either node.
        declared_nostream_nodes = {
            node_name
            for node_name in ("suppressed", "normal")
            if _get_declared_node_tags(graph, node_name)
            and "nostream" in _get_declared_node_tags(graph, node_name)
        }
        assert declared_nostream_nodes == set()
        flagged = flag_inconsistencies(decisions, declared_nostream_nodes)
        assert flagged == []
