"""
Integration tests for agent_trace.integrations.langgraph_state_diff.

Requires a real LangGraph installation (BaseCheckpointSaver, InMemorySaver).
Exercises the wrapped checkpointer against the real BaseCheckpointSaver
contract (confirmed signatures: put(config, checkpoint, metadata,
new_versions), put_writes(config, writes, task_id, task_path)) rather than
a full multi-agent Command(graph=Command.PARENT) graph run, since the
wrapper's diagnostic hooks the checkpointer boundary directly — the same
boundary every superstep's writes pass through regardless of which internal
LangGraph mechanism (parallel node writes, parallel tool-call Command.PARENT
routing, ...) produced them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")


@pytest.mark.integration
class TestWrapCheckpointer:
    def test_wrapped_checkpointer_is_a_base_checkpoint_saver(
        self, tmp_path: Path
    ) -> None:
        from langgraph.checkpoint.base import BaseCheckpointSaver
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        t = Tracer(trace_dir=tmp_path)
        wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
        assert isinstance(wrapped, BaseCheckpointSaver)

    def test_wrapped_checkpointer_passes_ensure_valid_checkpointer(
        self, tmp_path: Path
    ) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        try:
            from langgraph.types import ensure_valid_checkpointer
        except ImportError:
            from langgraph.checkpoint.base import (  # type: ignore[no-redef]
                ensure_valid_checkpointer,
            )

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        t = Tracer(trace_dir=tmp_path)
        wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
        # Must not raise.
        ensure_valid_checkpointer(wrapped)

    def test_put_and_get_tuple_still_work_through_the_wrapper(
        self, tmp_path: Path
    ) -> None:
        """The wrapper must not break the checkpointer's real read/write
        contract — put()'d data must be retrievable via get_tuple()."""
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("state-diff-passthrough"):
            wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
            config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
            checkpoint = {
                "v": 1,
                "id": "cp-1",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": {"x": 1},
                "channel_versions": {"x": "1"},
                "versions_seen": {},
            }
            wrapped.put(
                config,
                checkpoint,
                metadata={"source": "input", "step": 0},
                new_versions={"x": "1"},
            )

            tuple_ = wrapped.get_tuple(
                {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
            )
        assert tuple_ is not None
        assert tuple_.checkpoint["channel_values"]["x"] == 1

    def test_no_diagnostic_span_when_single_task_writes_channel(
        self, tmp_path: Path
    ) -> None:
        """Only one task proposing a write to a channel is normal — must not
        be flagged."""
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("state-diff-single-task") as trace:
            wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
            config = {
                "configurable": {
                    "thread_id": "t1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "parent-0",
                }
            }
            wrapped.put_writes(config, [("result", "only-value")], task_id="task-a")
            checkpoint = {"id": "cp-1", "channel_values": {"result": "only-value"}}
            wrapped.put(config, checkpoint, metadata={}, new_versions={})

        merge_spans = [s for s in trace.spans if s.name == "checkpoint:superstep_merge"]
        assert merge_spans == []

    def test_dropped_parallel_update_produces_superstep_merge_span(
        self, tmp_path: Path
    ) -> None:
        """The exact #7129 shape: 3 parallel tasks propose a write to the
        same channel, only 1 survives in the persisted checkpoint — must be
        flagged as an explicit N-of-M-dropped fact."""
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace import Tracer
        from agent_trace.core.span import SpanStatus
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("state-diff-drop") as trace:
            wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
            config = {
                "configurable": {
                    "thread_id": "t1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "parent-0",
                }
            }
            wrapped.put_writes(config, [("result", "from-a")], task_id="task-a")
            wrapped.put_writes(config, [("result", "from-b")], task_id="task-b")
            wrapped.put_writes(config, [("result", "from-c")], task_id="task-c")

            checkpoint = {"id": "cp-1", "channel_values": {"result": "from-b"}}
            wrapped.put(config, checkpoint, metadata={}, new_versions={})

        merge_spans = [s for s in trace.spans if s.name == "checkpoint:superstep_merge"]
        assert len(merge_spans) == 1
        span = merge_spans[0]
        assert span.status == SpanStatus.OK
        merge_events = [e for e in span.events if e.name == "superstep_state_merge"]
        assert len(merge_events) == 1
        attrs = merge_events[0].attributes
        assert attrs["channel"] == "result"
        assert attrs["proposed_count"] == 3
        assert attrs["survived_count"] == 1
        assert attrs["dropped_count"] == 2
        dropped_ids = set(attrs["dropped_task_ids"].split(","))
        assert dropped_ids == {"task-a", "task-c"}

    def test_reducer_channel_merging_all_values_produces_no_span(
        self, tmp_path: Path
    ) -> None:
        """When a reducer (e.g. add_messages) combines all N proposed values
        into the final list, nothing was dropped — no span expected."""
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("state-diff-reducer") as trace:
            wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
            config = {
                "configurable": {
                    "thread_id": "t1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "parent-0",
                }
            }
            wrapped.put_writes(config, [("messages", "a")], task_id="task-a")
            wrapped.put_writes(config, [("messages", "b")], task_id="task-b")

            checkpoint = {"id": "cp-1", "channel_values": {"messages": ["a", "b"]}}
            wrapped.put(config, checkpoint, metadata={}, new_versions={})

        merge_spans = [s for s in trace.spans if s.name == "checkpoint:superstep_merge"]
        assert merge_spans == []

    def test_unrelated_channels_not_conflated(self, tmp_path: Path) -> None:
        """Two tasks writing to *different* channels in the same superstep
        must not be treated as a merge conflict."""
        from langgraph.checkpoint.memory import InMemorySaver

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("state-diff-unrelated") as trace:
            wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
            config = {
                "configurable": {
                    "thread_id": "t1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "parent-0",
                }
            }
            wrapped.put_writes(config, [("x", 1)], task_id="task-a")
            wrapped.put_writes(config, [("y", 2)], task_id="task-b")

            checkpoint = {"id": "cp-1", "channel_values": {"x": 1, "y": 2}}
            wrapped.put(config, checkpoint, metadata={}, new_versions={})

        merge_spans = [s for s in trace.spans if s.name == "checkpoint:superstep_merge"]
        assert merge_spans == []

    def test_wrapped_checkpointer_works_in_a_real_graph_run(
        self, tmp_path: Path
    ) -> None:
        """End-to-end sanity check: a real graph.invoke() with the wrapped
        checkpointer must run and persist a checkpoint exactly as it would
        with the unwrapped one."""
        from typing import TypedDict

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, StateGraph

        from agent_trace import Tracer
        from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer

        class S(TypedDict):
            x: int

        builder = StateGraph(S)
        builder.add_node("step", lambda s: {"x": s["x"] + 1})
        builder.set_entry_point("step")
        builder.add_edge("step", END)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("state-diff-real-graph"):
            wrapped = wrap_checkpointer(InMemorySaver(), tracer=t)
            graph = builder.compile(checkpointer=wrapped)
            config = {"configurable": {"thread_id": "real-1"}}
            result = graph.invoke({"x": 0}, config=config)

        assert result["x"] == 1
