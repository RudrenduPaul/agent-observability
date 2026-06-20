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

    def test_all_spans_closed_after_clean_run(self, tmp_path: Path) -> None:
        """No span has end_time=None after graph.invoke() returns."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        builder = StateGraph(S)
        builder.add_node("step", lambda s: {"x": s["x"] + 1})
        builder.set_entry_point("step")
        builder.add_edge("step", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("lg-close") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"x": 0}, config={"callbacks": [cb]})

        unclosed = [s for s in trace.spans if s.end_time is None]
        assert unclosed == [], f"Spans left open: {[s.name for s in unclosed]}"

    def test_all_spans_ok_on_clean_run(self, tmp_path: Path) -> None:
        """All spans carry OK status when no node raises."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        builder = StateGraph(S)
        builder.add_node("step", lambda s: {"x": s["x"] + 1})
        builder.set_entry_point("step")
        builder.add_edge("step", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("lg-ok-status") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"x": 0}, config={"callbacks": [cb]})

        non_ok = [s for s in trace.spans if s.status != SpanStatus.OK]
        assert non_ok == [], (
            f"Non-OK spans on clean run: {[(s.name, s.status) for s in non_ok]}"
        )

    def test_span_registry_empty_after_graph_completes(self, tmp_path: Path) -> None:
        """handler._spans must be empty after a clean run — no leaked open spans."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        builder = StateGraph(S)
        builder.add_node("step", lambda s: {"x": s["x"] + 1})
        builder.set_entry_point("step")
        builder.add_edge("step", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("lg-registry") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"x": 0}, config={"callbacks": [cb]})

        assert cb._spans == {}, (
            f"Leaked entries in handler._spans: {list(cb._spans.keys())}"
        )

    def test_parent_child_span_hierarchy(self, tmp_path: Path) -> None:
        """At least one span must have a parent_id — LangGraph fires nested callbacks."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class AgentState(TypedDict):
            messages: list[str]

        def node_a(state: AgentState) -> AgentState:
            return {"messages": state["messages"] + ["a"]}

        def node_b(state: AgentState) -> AgentState:
            return {"messages": state["messages"] + ["b"]}

        builder = StateGraph(AgentState)
        builder.add_node("node_a", node_a)
        builder.add_node("node_b", node_b)
        builder.set_entry_point("node_a")
        builder.add_edge("node_a", "node_b")
        builder.add_edge("node_b", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("lg-hierarchy") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"messages": []}, config={"callbacks": [cb]})

        child_spans = [s for s in trace.spans if s.parent_id is not None]
        assert child_spans, (
            f"No span has a parent_id — LangGraph nested callbacks may not be wiring "
            f"parent_run_id correctly. All spans: "
            f"{[(s.name, s.parent_id) for s in trace.spans]}"
        )

    # ------------------------------------------------------------------
    # Gap 1: parent-child wiring under real LangGraph (tree, not flat list)
    # ------------------------------------------------------------------

    def test_node_spans_parent_ids_point_to_langgraph_root(
        self, tmp_path: Path
    ) -> None:
        """Node spans must be children of the LangGraph root chain span.

        LangGraph 1.x fires on_chain_start for the graph ('LangGraph') with
        no parent_run_id, then for each node with parent_run_id set to the
        graph's run_id.  This test verifies the resulting span tree is a proper
        tree (root → children), not a flat list where every span has parent_id=None.
        """
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            messages: list[str]

        builder = StateGraph(S)
        builder.add_node("node_a", lambda s: {"messages": s["messages"] + ["a"]})
        builder.add_node("node_b", lambda s: {"messages": s["messages"] + ["b"]})
        builder.set_entry_point("node_a")
        builder.add_edge("node_a", "node_b")
        builder.add_edge("node_b", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("parent-child-tree") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"messages": []}, config={"callbacks": [cb]})

        root_spans = [s for s in trace.spans if s.parent_id is None]
        child_spans = [s for s in trace.spans if s.parent_id is not None]

        assert root_spans, (
            "Expected a root span (the LangGraph-level chain). "
            f"All spans: {[(s.name, s.parent_id) for s in trace.spans]}"
        )
        assert child_spans, (
            "Expected node_a / node_b to be children of the root span — "
            "LangGraph may have stopped passing parent_run_id."
        )

        root_span_id = root_spans[0].span_id
        valid_parent_ids = {s.span_id for s in trace.spans}

        for child in child_spans:
            assert child.parent_id in valid_parent_ids, (
                f"Span '{child.name}' has parent_id={child.parent_id!r} "
                f"which does not match any span in the trace."
            )

        # node_a and node_b both have the LangGraph root as their direct parent
        node_spans = [
            s for s in child_spans if "node_a" in s.name or "node_b" in s.name
        ]
        assert node_spans, "Expected node_a and node_b spans among the children."
        for ns in node_spans:
            assert ns.parent_id == root_span_id, (
                f"Node span '{ns.name}' parent_id={ns.parent_id!r}, "
                f"expected root span_id={root_span_id!r}"
            )

    # ------------------------------------------------------------------
    # Gap 2 + 3: on_chat_model_start + token usage via FakeChatModel
    # ------------------------------------------------------------------

    def test_chat_model_callbacks_fire_through_langgraph(self, tmp_path: Path) -> None:
        """on_chat_model_start must fire when a real BaseChatModel is invoked
        inside a LangGraph node that passes RunnableConfig through.

        Uses a FakeChatModel stub — no HTTP calls, no API key required.
        """
        pytest.importorskip("langchain_core", reason="langchain_core not installed")

        from typing import Any, TypedDict

        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.runnables import RunnableConfig
        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class _FakeChatModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-chat"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="fake"))],
                    llm_output={
                        "token_usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 10,
                            "total_tokens": 15,
                        }
                    },
                )

        class S(TypedDict):
            messages: list[str]

        model = _FakeChatModel()

        def llm_node(state: S, config: RunnableConfig) -> S:
            result = model.invoke([HumanMessage(content="hello")], config=config)
            return {"messages": state["messages"] + [str(result.content)]}

        builder = StateGraph(S)
        builder.add_node("llm_node", llm_node)
        builder.set_entry_point("llm_node")
        builder.add_edge("llm_node", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("chat-model-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"messages": []}, config={"callbacks": [cb]})

        llm_spans = [s for s in trace.spans if s.name.startswith("llm")]
        assert llm_spans, (
            f"on_chat_model_start did not produce an 'llm' span. "
            f"Got: {[s.name for s in trace.spans]}"
        )

    def test_llm_span_has_token_attributes(self, tmp_path: Path) -> None:
        """LLM span must carry prompt/completion/total token counts from llm_output.

        Uses FakeChatModel so no API key is required.  The token counts come
        from the ChatResult.llm_output dict that on_llm_end reads.
        """
        pytest.importorskip("langchain_core", reason="langchain_core not installed")

        from typing import Any, TypedDict

        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.runnables import RunnableConfig
        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class _FakeChatModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-chat"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="fake"))],
                    llm_output={
                        "token_usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 10,
                            "total_tokens": 15,
                        }
                    },
                )

        class S(TypedDict):
            messages: list[str]

        model = _FakeChatModel()

        def llm_node(state: S, config: RunnableConfig) -> S:
            result = model.invoke([HumanMessage(content="hello")], config=config)
            return {"messages": state["messages"] + [str(result.content)]}

        builder = StateGraph(S)
        builder.add_node("llm_node", llm_node)
        builder.set_entry_point("llm_node")
        builder.add_edge("llm_node", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("token-attrs-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"messages": []}, config={"callbacks": [cb]})

        llm_span = next((s for s in trace.spans if s.name.startswith("llm")), None)
        assert llm_span is not None, (
            f"No llm span found. Spans: {[s.name for s in trace.spans]}"
        )
        assert llm_span.attributes.get("llm.usage.prompt_tokens") == 5
        assert llm_span.attributes.get("llm.usage.completion_tokens") == 10
        assert llm_span.attributes.get("llm.usage.total_tokens") == 15

    # ------------------------------------------------------------------
    # Gap 4: concurrent graph invocations on one LangGraphTracer
    # ------------------------------------------------------------------

    def test_concurrent_invocations_no_cross_contamination(
        self, tmp_path: Path
    ) -> None:
        """Two simultaneous graph.invoke() calls on the same LangGraphTracer
        must not cross-contaminate spans or leak open spans.

        The _lock in LangGraphTracer guards the _spans dict; this test verifies
        the locking is sufficient under real concurrent load.

        Python 3.14 does not inherit ContextVars in threads by default
        (sys.flags.thread_inherit_context == 0), so we pass an explicit
        contextvars.copy_context() to each thread to propagate the active trace.
        """
        import contextvars
        import threading
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            value: int

        builder = StateGraph(S)
        builder.add_node("step", lambda s: {"value": s["value"] + 1})
        builder.set_entry_point("step")
        builder.add_edge("step", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        results: list[dict] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        with t.start_trace("concurrent-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)

            def invoke_graph(init_value: int) -> None:
                try:
                    result = graph.invoke(
                        {"value": init_value}, config={"callbacks": [cb]}
                    )
                    with lock:
                        results.append(result)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

            # Each thread gets its own copy of the context so _active_trace_var
            # is visible inside the thread (Python 3.14 doesn't inherit by default).
            threads = [
                threading.Thread(
                    target=contextvars.copy_context().run,
                    args=(invoke_graph, i),
                )
                for i in range(2)
            ]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

        assert not errors, f"Concurrent invocations raised exceptions: {errors}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert {r["value"] for r in results} == {1, 2}, (
            f"Expected values {{1, 2}}, got {{{', '.join(str(r['value']) for r in results)}}}"
        )
        assert cb._spans == {}, (
            f"Span registry leaked after concurrent run: {list(cb._spans.keys())}"
        )
        # Each single-node graph fires at least 2 callbacks (LangGraph root + node)
        # so 2 concurrent runs must produce at least 4 spans total.
        assert len(trace.spans) >= 4, (
            f"Expected >= 4 spans from 2 concurrent runs, got {len(trace.spans)}"
        )

    # ------------------------------------------------------------------
    # Gap 5: replay determinism — span tree comparison
    # ------------------------------------------------------------------

    def test_replay_span_tree_matches_record_span_tree(self, tmp_path: Path) -> None:
        """Replayed span tree must match the recorded span tree name-for-name
        and attribute-for-attribute in order.

        This is stronger than the existing replay test, which only checks that
        len(trace.spans) >= 1 and that result["messages"] is truthy.
        """
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer, replay
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        builder = StateGraph(S)
        builder.add_node("step_a", lambda s: {"x": s["x"] + 1})
        builder.add_node("step_b", lambda s: {"x": s["x"] * 2})
        builder.set_entry_point("step_a")
        builder.add_edge("step_a", "step_b")
        builder.add_edge("step_b", END)
        graph = builder.compile()

        run_id = "replay-determinism-test"
        t_rec = Tracer(trace_dir=tmp_path)

        # Record pass
        with t_rec.start_trace("record", record=True, run_id=run_id) as record_trace:
            cb_rec = LangGraphTracer(tracer=t_rec, trace=record_trace)
            record_result = graph.invoke({"x": 1}, config={"callbacks": [cb_rec]})

        record_span_names = [s.name for s in record_trace.spans]
        record_lg_attrs = [
            {k: v for k, v in s.attributes.items() if k.startswith("langgraph.")}
            for s in record_trace.spans
        ]
        assert record_result == {"x": 4}, f"(1+1)*2 should be 4, got {record_result}"
        assert len(record_span_names) >= 1

        # Replay pass — wire up a fresh tracer so we can capture spans
        t_rep = Tracer(trace_dir=tmp_path)
        with t_rep.start_trace("replay") as replay_trace:
            cb_rep = LangGraphTracer(tracer=t_rep, trace=replay_trace)
            with replay(run_id, trace_dir=tmp_path):
                replay_result = graph.invoke({"x": 1}, config={"callbacks": [cb_rep]})

        replay_span_names = [s.name for s in replay_trace.spans]
        replay_lg_attrs = [
            {k: v for k, v in s.attributes.items() if k.startswith("langgraph.")}
            for s in replay_trace.spans
        ]

        assert replay_result == record_result, (
            f"Replay produced different output: record={record_result} replay={replay_result}"
        )
        assert record_span_names == replay_span_names, (
            f"Span tree mismatch between record and replay.\n"
            f"  Record: {record_span_names}\n"
            f"  Replay: {replay_span_names}"
        )
        assert record_lg_attrs == replay_lg_attrs, (
            f"LangGraph attribute mismatch between record and replay.\n"
            f"  Record: {record_lg_attrs}\n"
            f"  Replay: {replay_lg_attrs}"
        )
