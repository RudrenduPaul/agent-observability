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

    # ------------------------------------------------------------------
    # Previously-discarded data — now captured onto spans (real LangGraph)
    # ------------------------------------------------------------------

    def test_node_span_captures_inputs_and_outputs(self, tmp_path: Path) -> None:
        """chain.inputs/chain.outputs must reflect the real node state dict."""
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
        with t.start_trace("inputs-outputs-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"x": 5}, config={"callbacks": [cb]})

        node_span = next((s for s in trace.spans if "step" in s.name), None)
        assert node_span is not None
        assert node_span.attributes.get("chain.inputs") == '{"x": 5}'
        assert node_span.attributes.get("chain.outputs") == '{"x": 6}'

    def test_node_span_captures_runtime_context(self, tmp_path: Path) -> None:
        """chain.runtime must be populated via the RunnableCallable monkeypatch
        (_install_runtime_capture_patch) for a real graph.invoke() call."""
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
        with t.start_trace("runtime-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"x": 0}, config={"callbacks": [cb]})

        node_span = next((s for s in trace.spans if "step" in s.name), None)
        assert node_span is not None
        assert "chain.runtime" in node_span.attributes, (
            f"Expected chain.runtime on the node span; attributes were: "
            f"{node_span.attributes}"
        )
        assert "Runtime" in node_span.attributes["chain.runtime"]

    def test_tool_span_captures_input_and_output(self, tmp_path: Path) -> None:
        """tool.input/tool.output must reflect the real ToolNode call."""
        pytest.importorskip("langchain_core", reason="langchain_core not installed")

        from typing import Any, TypedDict

        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        @tool
        def echo(text: str) -> str:
            """Echo the given text back."""
            return f"echo:{text}"

        class _ToolCallingModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-tool-caller"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "echo",
                                        "args": {"text": "hi"},
                                        "id": "call_1",
                                    }
                                ],
                            )
                        )
                    ]
                )

        class S(TypedDict):
            messages: list

        model = _ToolCallingModel()
        tool_node = ToolNode([echo])

        def agent_node(state: S, config) -> S:
            result = model.invoke(state["messages"], config=config)
            return {"messages": state["messages"] + [result]}

        builder = StateGraph(S)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)
        builder.set_entry_point("agent")
        builder.add_edge("agent", "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("tool-io-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke(
                {"messages": [HumanMessage(content="hi")]},
                config={"callbacks": [cb]},
            )

        tool_span = next((s for s in trace.spans if s.name.startswith("tool:")), None)
        assert tool_span is not None, (
            f"No tool span found. Spans: {[s.name for s in trace.spans]}"
        )
        assert "hi" in tool_span.attributes.get("tool.input", "")
        assert "echo:hi" in tool_span.attributes.get("tool.output", "")
        assert tool_span.attributes.get("tool.has_event_loop") is False

    def test_llm_span_captures_content_via_fake_chat_model(
        self, tmp_path: Path
    ) -> None:
        """llm.content must carry the actual generated text, not just usage."""
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
                    generations=[
                        ChatGeneration(message=AIMessage(content="distinctive-text"))
                    ],
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
        with t.start_trace("content-capture-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"messages": []}, config={"callbacks": [cb]})

        llm_span = next((s for s in trace.spans if s.name.startswith("llm")), None)
        assert llm_span is not None
        assert llm_span.attributes.get("llm.content") == "distinctive-text"

    def test_chat_model_start_captures_messages_via_langgraph(
        self, tmp_path: Path
    ) -> None:
        """llm.messages must carry the real HumanMessage content."""
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
                    generations=[ChatGeneration(message=AIMessage(content="ok"))],
                )

        class S(TypedDict):
            messages: list[str]

        model = _FakeChatModel()

        def llm_node(state: S, config: RunnableConfig) -> S:
            result = model.invoke(
                [HumanMessage(content="a very distinctive prompt")], config=config
            )
            return {"messages": state["messages"] + [str(result.content)]}

        builder = StateGraph(S)
        builder.add_node("llm_node", llm_node)
        builder.set_entry_point("llm_node")
        builder.add_edge("llm_node", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("messages-capture-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            graph.invoke({"messages": []}, config={"callbacks": [cb]})

        llm_span = next((s for s in trace.spans if s.name.startswith("llm")), None)
        assert llm_span is not None
        assert "a very distinctive prompt" in llm_span.attributes.get(
            "llm.messages", ""
        )


# ---------------------------------------------------------------------------
# LangGraph internal control-flow signals — must not be marked ERROR
# ---------------------------------------------------------------------------
#
# Verified against the real langgraph package's actual exception types
# (langgraph.errors.ParentCommand / GraphInterrupt / GraphBubbleUp) — not
# stand-ins — since these tests require a real langgraph installation.


@pytest.mark.integration
class TestControlFlowSignalNotMarkedError:
    def test_command_parent_handoff_closes_span_ok_not_error(
        self, tmp_path: Path
    ) -> None:
        """A node returning Command(graph=Command.PARENT, ...) raises
        langgraph.errors.ParentCommand internally to implement the handoff
        jump — this must close OK with langgraph.handoff=true, not ERROR."""
        from typing import TypedDict

        from langgraph.errors import ParentCommand
        from langgraph.graph import END, StateGraph
        from langgraph.types import Command

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def handoff_node(state: S) -> Command:
            return Command(graph=Command.PARENT, goto=END, update={"x": state["x"] + 1})

        builder = StateGraph(S)
        builder.add_node("handoff", handoff_node)
        builder.set_entry_point("handoff")
        builder.add_edge("handoff", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(ParentCommand):
            with t.start_trace("handoff-test") as trace:
                cb = LangGraphTracer(tracer=t, trace=trace)
                graph.invoke({"x": 0}, config={"callbacks": [cb]})

        handoff_spans = [s for s in trace.spans if "handoff" in s.name]
        assert handoff_spans, f"Expected a handoff span. Got: {trace.spans}"
        for span in handoff_spans:
            assert span.status == SpanStatus.OK, (
                f"{span.name} should close OK, not {span.status}"
            )
            assert span.attributes.get("langgraph.handoff") is True
            assert (
                span.attributes.get("langgraph.control_flow_signal")
                == "ParentCommand"
            )

    def test_no_error_status_span_anywhere_on_handoff_run(self, tmp_path: Path) -> None:
        """A clean handoff run must produce zero ERROR-status spans."""
        from typing import TypedDict

        from langgraph.errors import ParentCommand
        from langgraph.graph import END, StateGraph
        from langgraph.types import Command

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def handoff_node(state: S) -> Command:
            return Command(graph=Command.PARENT, goto=END, update={"x": state["x"]})

        builder = StateGraph(S)
        builder.add_node("handoff", handoff_node)
        builder.set_entry_point("handoff")
        builder.add_edge("handoff", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(ParentCommand):
            with t.start_trace("handoff-no-error-test") as trace:
                cb = LangGraphTracer(tracer=t, trace=trace)
                graph.invoke({"x": 0}, config={"callbacks": [cb]})

        error_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        assert error_spans == [], (
            f"A handoff jump must not produce any ERROR span. "
            f"Got: {[(s.name, s.status) for s in error_spans]}"
        )

    def test_graph_interrupt_closes_span_ok_not_error(self, tmp_path: Path) -> None:
        """A node calling interrupt() raises langgraph.errors.GraphInterrupt
        internally — the node's span must close OK with
        langgraph.interrupted=true, not ERROR."""
        from typing import TypedDict

        from langgraph.graph import END, StateGraph
        from langgraph.types import interrupt

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def pause_node(state: S) -> S:
            interrupt("need human input")
            return {"x": state["x"] + 1}  # pragma: no cover — unreachable pre-resume

        builder = StateGraph(S)
        builder.add_node("pause", pause_node)
        builder.set_entry_point("pause")
        builder.add_edge("pause", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("interrupt-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            result = graph.invoke({"x": 0}, config={"callbacks": [cb]})

        assert "__interrupt__" in result

        pause_spans = [s for s in trace.spans if "pause" in s.name]
        assert pause_spans, f"Expected a pause span. Got: {trace.spans}"
        for span in pause_spans:
            assert span.status == SpanStatus.OK, (
                f"{span.name} should close OK, not {span.status}"
            )
            assert span.attributes.get("langgraph.interrupted") is True
            assert "langgraph.handoff" not in span.attributes


# ---------------------------------------------------------------------------
# CANCELLED span status — asyncio.CancelledError distinct from ERROR
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCancelledSpanStatus:
    async def test_cancelled_async_node_closes_spans_cancelled_not_error(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        async def slow_node(state: S) -> S:
            await asyncio.sleep(5)
            return {"x": state["x"] + 1}  # pragma: no cover — never reached

        builder = StateGraph(S)
        builder.add_node("slow", slow_node)
        builder.set_entry_point("slow")
        builder.add_edge("slow", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("cancel-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    graph.ainvoke({"x": 0}, config={"callbacks": [cb]}),
                    timeout=0.2,
                )

        slow_spans = [s for s in trace.spans if "slow" in s.name]
        assert slow_spans, f"Expected a slow-node span. Got: {trace.spans}"
        for span in slow_spans:
            assert span.status == SpanStatus.CANCELLED
        error_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        assert error_spans == [], (
            f"Cancellation must not produce ERROR spans. "
            f"Got: {[(s.name, s.status) for s in error_spans]}"
        )


# ---------------------------------------------------------------------------
# Branch (conditional-edge) dispatch exception capture despite trace=False
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBranchDispatchExceptionCapture:
    def test_unregistered_destination_produces_branch_dispatch_span(
        self, tmp_path: Path
    ) -> None:
        """A router returning a destination absent from the registered
        path_map raises KeyError inside BranchSpec._finish() — a component
        LangGraph itself builds with trace=False. Without the patch this
        produces zero additional evidence; with it, a 'branch:dispatch'
        ERROR span is captured.

        Graph is compiled *before* any LangGraphTracer is constructed —
        the realistic ordering (module-level build_graph(), tracer
        constructed per-invocation) — to guard against a patch that only
        works when installed before compile-time.
        """
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def route(state) -> str:
            return "not_a_registered_destination"

        def a_node(state: S) -> S:
            return {"x": state["x"] + 100}  # pragma: no cover — unreachable

        builder = StateGraph(S)
        builder.add_node("a", a_node)
        builder.add_conditional_edges(START, route, {"a": "a"})
        builder.add_edge("a", END)
        graph = builder.compile()  # compiled before any LangGraphTracer exists

        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(KeyError):
            with t.start_trace("branch-dispatch-test") as trace:
                cb = LangGraphTracer(tracer=t, trace=trace)
                graph.invoke({"x": 0}, config={"callbacks": [cb]})

        branch_spans = [s for s in trace.spans if s.name == "branch:dispatch"]
        assert branch_spans, (
            f"Expected a 'branch:dispatch' span. Got: {[s.name for s in trace.spans]}"
        )
        span = branch_spans[0]
        assert span.status == SpanStatus.ERROR
        assert span.attributes.get("langgraph.branch_dispatch") is True
        assert "a" in span.attributes.get("branch.registered_destinations", "")
        exception_events = [e for e in span.events if e.name == "exception"]
        assert exception_events
        assert exception_events[0].attributes["exception.type"] == "KeyError"

    def test_normal_conditional_edge_dispatch_produces_no_branch_error_span(
        self, tmp_path: Path
    ) -> None:
        """A router returning a *registered* destination must not produce
        any branch:dispatch span — only the failure path should."""
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def route(state) -> str:
            return "a"

        def a_node(state: S) -> S:
            return {"x": state["x"] + 1}

        builder = StateGraph(S)
        builder.add_node("a", a_node)
        builder.add_conditional_edges(START, route, {"a": "a"})
        builder.add_edge("a", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("branch-dispatch-ok-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            result = graph.invoke({"x": 0}, config={"callbacks": [cb]})

        assert result["x"] == 1
        branch_spans = [s for s in trace.spans if s.name == "branch:dispatch"]
        assert branch_spans == []


@pytest.mark.integration
class TestStreamingCallbackHooksAgainstRealLangGraph:
    """on_llm_new_token, exercised via a real streaming BaseChatModel run
    through a real LangGraph node."""

    def test_streaming_chat_model_call_records_deltas(self, tmp_path: Path) -> None:
        from typing import Any

        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.messages.ai import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk, ChatResult
        from langgraph.graph import END, START, MessagesState, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class FakeStreamingModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake-streaming"

            def _generate(
                self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
            ) -> ChatResult:
                raise AssertionError("expected _stream, not _generate, to be used")

            def _stream(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
                for tok in ["hel", "lo"]:
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=tok))
                    if run_manager:
                        run_manager.on_llm_new_token(tok, chunk=chunk)
                    yield chunk

        def call_model(state: MessagesState) -> dict[str, Any]:
            chunks = list(FakeStreamingModel().stream(state["messages"]))
            full = chunks[0]
            for c in chunks[1:]:
                full = full + c
            return {"messages": [AIMessage(content=full.content)]}

        builder = StateGraph(MessagesState)
        builder.add_node("n1", call_model)
        builder.add_edge(START, "n1")
        builder.add_edge("n1", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("streaming-callback-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace)
            result = graph.invoke(
                {"messages": [HumanMessage(content="hi")]}, config={"callbacks": [cb]}
            )

        assert result["messages"][-1].content == "hello"

        llm_span = next(s for s in trace.spans if s.name.startswith("llm:"))
        assert llm_span.attributes.get("llm.streamed") is True
        assert llm_span.attributes.get("llm.stream_token_count", 0) >= 2
        delta_events = [e for e in llm_span.events if e.name == "llm_stream_delta"]
        tokens = [e.attributes.get("token") for e in delta_events if "token" in e.attributes]
        assert "hel" in tokens
        assert "lo" in tokens


@pytest.mark.integration
class TestDeclaredNodeTagsAgainstRealLangGraph:
    """on_chain_start: a compiled graph's node-level declared tags (set via
    .with_config(tags=[...]) on the node's own action, the only mechanism
    the installed LangGraph version actually supports — confirmed via direct
    inspection, since StateGraph.add_node() has no tags= kwarg) land on the
    node span when graph= is supplied."""

    def test_declared_tags_captured_from_with_config(self, tmp_path: Path) -> None:
        from typing import TypedDict

        from langchain_core.runnables import RunnableLambda
        from langgraph.graph import END, START, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def n1(state: S) -> S:
            return {"x": state["x"] + 1}

        builder = StateGraph(S)
        builder.add_node("n1", RunnableLambda(n1).with_config(tags=["nostream"]))
        builder.add_edge(START, "n1")
        builder.add_edge("n1", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("declared-tags-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace, graph=graph)
            graph.invoke({"x": 0}, config={"callbacks": [cb]})

        node_span = next(s for s in trace.spans if s.name == "node:n1")
        assert node_span.attributes.get("langgraph.declared_tags") == "nostream"
        # The runtime `tags` kwarg on_chain_start receives never carries it —
        # confirming this is genuinely new information, not a duplicate of
        # what langgraph.tags already captured.
        assert "nostream" not in node_span.attributes.get("langgraph.tags", "")

    def test_no_declared_tags_when_node_not_configured(self, tmp_path: Path) -> None:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import LangGraphTracer

        class S(TypedDict):
            x: int

        def n1(state: S) -> S:
            return {"x": state["x"] + 1}

        builder = StateGraph(S)
        builder.add_node("n1", n1)
        builder.add_edge(START, "n1")
        builder.add_edge("n1", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("no-declared-tags-test") as trace:
            cb = LangGraphTracer(tracer=t, trace=trace, graph=graph)
            graph.invoke({"x": 0}, config={"callbacks": [cb]})

        node_span = next(s for s in trace.spans if s.name == "node:n1")
        assert "langgraph.declared_tags" not in node_span.attributes


@pytest.mark.integration
class TestTracedStreamAgainstRealLangGraph:
    """traced_stream(): wraps graph.stream() against a real LangGraph graph,
    recording one stream_yield SpanEvent per item actually delivered to the
    caller."""

    def test_wraps_real_graph_stream(self, tmp_path: Path) -> None:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import traced_stream

        class S(TypedDict):
            x: int

        def n1(state: S) -> S:
            return {"x": state["x"] + 1}

        def n2(state: S) -> S:
            return {"x": state["x"] + 10}

        builder = StateGraph(S)
        builder.add_node("n1", n1)
        builder.add_node("n2", n2)
        builder.add_edge(START, "n1")
        builder.add_edge("n1", "n2")
        builder.add_edge("n2", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("traced-stream-test") as trace:
            results = list(
                traced_stream(t, graph.stream({"x": 0}, stream_mode="updates"))
            )

        assert results == [{"n1": {"x": 1}}, {"n2": {"x": 11}}]
        stream_span = next(s for s in trace.spans if s.name == "graph:stream")
        assert stream_span.attributes.get("stream.chunk_count") == 2
        assert stream_span.status.value == "OK"
        yield_events = [e for e in stream_span.events if e.name == "stream_yield"]
        assert len(yield_events) == 2
        assert "n1" in yield_events[0].attributes["stream.chunk"]
        assert "n2" in yield_events[1].attributes["stream.chunk"]

    def test_invoke_vs_stream_have_different_span_shapes(self, tmp_path: Path) -> None:
        """graph.invoke() produces no graph:stream span at all (it drains
        internally, never yielding progressively to caller code); wrapping
        graph.stream() in traced_stream() does. This is the exact delivery-
        timing distinction #4653 is about — the two must be observably
        different at the span-tree level, not just at the Python-API level."""
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph import traced_stream

        class S(TypedDict):
            x: int

        def n1(state: S) -> S:
            return {"x": state["x"] + 1}

        builder = StateGraph(S)
        builder.add_node("n1", n1)
        builder.add_edge(START, "n1")
        builder.add_edge("n1", END)
        graph = builder.compile()

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("invoke-test") as trace:
            graph.invoke({"x": 0})
        assert not any(s.name == "graph:stream" for s in trace.spans)

        with t.start_trace("stream-test") as trace:
            list(traced_stream(t, graph.stream({"x": 0}, stream_mode="updates")))
        assert any(s.name == "graph:stream" for s in trace.spans)
