"""
Integration tests for agent_trace.integrations.langgraph_checkpoint —
checkpointer write/serde instrumentation, CachePolicy hit/miss instrumentation,
and as_node/task-scheduling instrumentation for update_state().

These tests require a real LangGraph installation (InMemorySaver,
InMemoryCache, CachePolicy, StateGraph) but do NOT require live LLM API
calls — the graphs use pure-Python nodes only.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

from agent_trace import Tracer
from agent_trace.integrations.langgraph_checkpoint import (
    TracingCache,
    TracingCheckpointSaver,
    traced_aupdate_state,
    traced_update_state,
    wrap_cache_policy,
)


class _State(TypedDict):
    x: int


def _node_a(state: _State) -> _State:
    return {"x": state["x"] + 1}


def _node_b(state: _State) -> _State:
    return {"x": state["x"] + 10}


def _build_graph(checkpointer=None, cache=None, cache_policy=None):
    from langgraph.graph import END, StateGraph

    builder = StateGraph(_State)
    builder.add_node("node_a", _node_a, cache_policy=cache_policy)
    builder.add_node("node_b", _node_b)
    builder.set_entry_point("node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder.compile(checkpointer=checkpointer, cache=cache)


@pytest.mark.integration
class TestTracingCheckpointSaver:
    def test_is_a_genuine_basecheckpointsaver_instance(self, tmp_path: Path) -> None:
        """LangGraph's Pregel._defaults() gates checkpoint behavior on
        isinstance(checkpointer, BaseCheckpointSaver) — a duck-typed wrapper
        would be silently treated as no checkpointer at all."""
        from langgraph.checkpoint.base import BaseCheckpointSaver
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        saver = TracingCheckpointSaver(InMemorySaver(), t)
        assert isinstance(saver, BaseCheckpointSaver)

    def test_put_and_put_writes_record_spans(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("checkpoint-write-test") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": "th1"}}
            result = graph.invoke({"x": 0}, config=config)

        assert result["x"] == 11
        put_spans = [s for s in trace.spans if s.name == "checkpoint:put"]
        put_writes_spans = [s for s in trace.spans if s.name == "checkpoint:put_writes"]
        assert put_spans, "expected at least one checkpoint:put span"
        assert put_writes_spans, "expected at least one checkpoint:put_writes span"
        for span in put_spans:
            assert span.attributes.get("checkpoint.thread_id") == "th1"
            assert span.attributes.get("checkpoint.completed") is True
            assert isinstance(span.attributes.get("checkpoint.payload_size_bytes"), int)
            assert span.status.value == "OK"

    def test_put_writes_records_task_id_and_channels(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("checkpoint-write-test-2") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            graph.invoke({"x": 0}, config={"configurable": {"thread_id": "th2"}})

        put_writes = [s for s in trace.spans if s.name == "checkpoint:put_writes"]
        assert put_writes
        assert all(s.attributes.get("checkpoint.task_id") for s in put_writes)
        assert any(
            "x" in s.attributes.get("checkpoint.channels", "") for s in put_writes
        )

    def test_serde_boundary_spans_recorded(self, tmp_path: Path) -> None:
        """dumps_typed/loads_typed calls made internally by InMemorySaver's
        own put()/get_tuple() are captured — not just calls this wrapper
        happens to make directly (confirms the inner checkpointer's own
        .serde attribute was actually replaced in place)."""
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("serde-boundary-test") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            graph.invoke({"x": 0}, config={"configurable": {"thread_id": "th3"}})

        serde_spans = [s for s in trace.spans if s.name.startswith("checkpoint:serde:")]
        assert serde_spans, "expected serde-boundary spans"
        dumps_spans = [
            s for s in serde_spans if s.name == "checkpoint:serde:dumps_typed"
        ]
        assert dumps_spans
        for span in dumps_spans:
            assert isinstance(span.attributes.get("serde.byte_size"), int)
            assert span.attributes.get("serde.byte_size") >= 0
            assert "serde.type" in span.attributes
            assert isinstance(span.attributes.get("serde.duration_ms"), float)

    def test_reads_and_admin_delegate_correctly(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("delegation-test"):
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": "th4"}}
            graph.invoke({"x": 0}, config=config)

            # get/get_tuple/list all delegate to the wrapped InMemorySaver.
            tuple_ = saver.get_tuple(config)
            assert tuple_ is not None
            history = list(graph.get_state_history(config))
            assert len(history) > 0

            # Admin operation delegates without raising.
            saver.delete_thread("th4")
            assert saver.get_tuple(config) is None

    def test_async_aput_and_aput_writes_record_spans(self, tmp_path: Path) -> None:
        import asyncio

        from langgraph.checkpoint.memory import InMemorySaver

        async def _run():
            t = Tracer(trace_dir=tmp_path)
            with t.start_trace("async-checkpoint-test") as trace:
                saver = TracingCheckpointSaver(InMemorySaver(), t)
                graph = _build_graph(checkpointer=saver)
                config = {"configurable": {"thread_id": "ath1"}}
                result = await graph.ainvoke({"x": 0}, config=config)
            return trace, result

        trace, result = asyncio.run(_run())
        assert result["x"] == 11
        aput_spans = [s for s in trace.spans if s.name == "checkpoint:aput"]
        aput_writes_spans = [
            s for s in trace.spans if s.name == "checkpoint:aput_writes"
        ]
        assert aput_spans
        assert aput_writes_spans
        assert all(s.attributes.get("checkpoint.completed") is True for s in aput_spans)

    def test_write_failure_closes_span_error_and_reraises(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.base import BaseCheckpointSaver

        class _BoomSaver(BaseCheckpointSaver):
            def put(self, config, checkpoint, metadata, new_versions):
                raise RuntimeError("disk full")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("checkpoint-failure-test") as trace:
            saver = TracingCheckpointSaver(_BoomSaver(), t)
            with pytest.raises(RuntimeError, match="disk full"):
                saver.put(
                    {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                    {"id": "c1"},
                    {},
                    {},
                )
        put_spans = [s for s in trace.spans if s.name == "checkpoint:put"]
        assert len(put_spans) == 1
        assert put_spans[0].status.value == "ERROR"
        assert put_spans[0].attributes.get("checkpoint.completed") is False


@pytest.mark.integration
class TestTracingCacheAndCachePolicy:
    def test_cache_miss_then_hit_recorded(self, tmp_path: Path) -> None:
        from langgraph.cache.memory import InMemoryCache
        from langgraph.types import CachePolicy

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("cache-test") as trace:
            cache = TracingCache(InMemoryCache(), t)
            policy = wrap_cache_policy(CachePolicy(), t)
            graph = _build_graph(cache=cache, cache_policy=policy)

            # Same input both times -> second call should hit the cache for node_a.
            graph.invoke({"x": 0}, config={"configurable": {"thread_id": "c1"}})
            graph.invoke({"x": 0}, config={"configurable": {"thread_id": "c2"}})

        get_spans = [s for s in trace.spans if s.name == "cache:get"]
        assert len(get_spans) == 2
        misses = [s for s in get_spans if s.attributes.get("cache.miss_count", 0) > 0]
        hits = [s for s in get_spans if s.attributes.get("cache.hit_count", 0) > 0]
        assert misses, "first call should have missed the cache"
        assert hits, "second call with identical input should have hit the cache"

    def test_cache_set_recorded_on_miss(self, tmp_path: Path) -> None:
        from langgraph.cache.memory import InMemoryCache
        from langgraph.types import CachePolicy

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("cache-set-test") as trace:
            cache = TracingCache(InMemoryCache(), t)
            policy = wrap_cache_policy(CachePolicy(), t)
            graph = _build_graph(cache=cache, cache_policy=policy)
            graph.invoke({"x": 0}, config={"configurable": {"thread_id": "c3"}})

        set_spans = [s for s in trace.spans if s.name == "cache:set"]
        assert set_spans
        assert set_spans[0].attributes.get("cache.set_count") == 1

    def test_key_func_records_hashed_state_input(self, tmp_path: Path) -> None:
        from langgraph.cache.memory import InMemoryCache
        from langgraph.types import CachePolicy

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("cache-keyfunc-test") as trace:
            cache = TracingCache(InMemoryCache(), t)
            policy = wrap_cache_policy(CachePolicy(), t)
            graph = _build_graph(cache=cache, cache_policy=policy)
            graph.invoke({"x": 5}, config={"configurable": {"thread_id": "c4"}})

        key_func_spans = [s for s in trace.spans if s.name == "cache:key_func"]
        assert key_func_spans
        # The actual state object hashed to compute the cache key must be
        # observable — not just the resulting opaque key.
        assert '"x": 5' in key_func_spans[0].attributes.get("cache.key_input", "")

    def test_wrap_cache_policy_none_returns_none(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        assert wrap_cache_policy(None, t) is None


@pytest.mark.integration
class TestTracedUpdateState:
    def test_records_as_node_and_next_task_count(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("update-state-test") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": "u1"}}
            graph.invoke({"x": 0}, config=config)

            # node_b -> END: updating state "as" node_b (the last node before
            # the graph's terminal edge) schedules zero further tasks.
            traced_update_state(t, graph, config, {"x": 999}, as_node="node_b")

        update_spans = [s for s in trace.spans if s.name == "checkpoint:update_state"]
        assert len(update_spans) == 1
        span = update_spans[0]
        assert span.attributes.get("checkpoint.as_node") == "node_b"
        assert span.attributes.get("checkpoint.as_node_provided") is True
        assert span.attributes.get("checkpoint.zero_tasks_scheduled") is True
        assert span.attributes.get("checkpoint.next_task_count") == 0
        assert span.status.value == "OK"

    def test_as_node_not_provided_is_honestly_flagged(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("update-state-no-as-node-test") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": "u2"}}
            graph.invoke({"x": 0}, config=config)
            traced_update_state(t, graph, config, {"x": 5})

        span = next(s for s in trace.spans if s.name == "checkpoint:update_state")
        assert span.attributes.get("checkpoint.as_node_provided") is False
        assert "checkpoint.as_node" not in span.attributes

    def test_async_variant_records_same_shape(self, tmp_path: Path) -> None:
        import asyncio

        from langgraph.checkpoint.memory import InMemorySaver

        async def _run():
            t = Tracer(trace_dir=tmp_path)
            with t.start_trace("async-update-state-test") as trace:
                saver = TracingCheckpointSaver(InMemorySaver(), t)
                graph = _build_graph(checkpointer=saver)
                config = {"configurable": {"thread_id": "u3"}}
                await graph.ainvoke({"x": 0}, config=config)
                await traced_aupdate_state(t, graph, config, {"x": 5}, as_node="node_b")
            return trace

        trace = asyncio.run(_run())
        span = next(s for s in trace.spans if s.name == "checkpoint:update_state")
        assert span.attributes.get("checkpoint.as_node") == "node_b"
        assert span.attributes.get("checkpoint.zero_tasks_scheduled") is True

    def test_update_state_error_closes_span_error(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("update-state-error-test") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": "u4"}}
            graph.invoke({"x": 0}, config=config)
            with pytest.raises(Exception):
                # Nonexistent node -> LangGraph raises InvalidUpdateError.
                traced_update_state(t, graph, config, {"x": 1}, as_node="not_a_node")

        span = next(s for s in trace.spans if s.name == "checkpoint:update_state")
        assert span.status.value == "ERROR"


@pytest.mark.integration
class TestCheckpointDurabilityCliDiagnosticEndToEnd:
    """End-to-end: real checkpoint:put spans -> _cli.py's durability
    diagnostic actually reports 'durable' for a normal successful run."""

    def test_successful_run_reports_durable(self, tmp_path: Path) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace._cli import _checkpoint_durability_summary

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("durability-e2e-test") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            graph.invoke({"x": 0}, config={"configurable": {"thread_id": "d1"}})

        spans_as_dicts = [s.to_dict() for s in trace.spans]
        summary = _checkpoint_durability_summary(spans_as_dicts)
        assert summary is not None
        assert summary["checkpoint_status"] == "durable"
        assert summary["writes_flushed_count"] == summary["writes_enqueued_count"]

    def test_zero_tasks_scheduled_cli_diagnostic_end_to_end(
        self, tmp_path: Path
    ) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace._cli import _zero_task_update_rows

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("zero-tasks-e2e-test") as trace:
            saver = TracingCheckpointSaver(InMemorySaver(), t)
            graph = _build_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": "z1"}}
            graph.invoke({"x": 0}, config=config)
            traced_update_state(t, graph, config, {"x": 1}, as_node="node_b")

        spans_as_dicts = [s.to_dict() for s in trace.spans]
        rows = _zero_task_update_rows(spans_as_dicts)
        assert len(rows) == 1
        assert rows[0]["as_node"] == "node_b"
