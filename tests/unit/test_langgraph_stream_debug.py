"""
Unit tests for agent_trace.integrations.langgraph_stream_debug — the pure
helper functions behind the optional StreamMessagesHandler deep
instrumentation.

These test only the module-level pure functions (StreamDecision,
flag_inconsistencies, get/reset_stream_decisions), which don't require
langgraph to be installed. End-to-end coverage of the actual
StreamMessagesHandler monkeypatch against a real LangGraph graph lives in
tests/integration/test_langgraph_stream_debug.py.
"""

from __future__ import annotations

from agent_trace.integrations.langgraph_stream_debug import (
    StreamDecision,
    flag_inconsistencies,
    get_stream_decisions,
    reset_stream_decisions,
)

# ---------------------------------------------------------------------------
# StreamDecision
# ---------------------------------------------------------------------------


class TestStreamDecision:
    def test_defaults(self) -> None:
        d = StreamDecision(node_name="n1", run_id="r1")
        assert d.tags == ()
        assert d.suppressed is False

    def test_is_frozen(self) -> None:
        d = StreamDecision(node_name="n1", run_id="r1")
        try:
            d.node_name = "other"  # type: ignore[misc]
            raised = False
        except Exception:
            raised = True
        assert raised


# ---------------------------------------------------------------------------
# flag_inconsistencies
# ---------------------------------------------------------------------------


class TestFlagInconsistencies:
    def test_flags_declared_nostream_node_that_still_streamed(self) -> None:
        decisions = [
            StreamDecision(node_name="n1", run_id="r1", tags=(), suppressed=False),
        ]
        flagged = flag_inconsistencies(decisions, declared_nostream_nodes={"n1"})
        assert flagged == decisions

    def test_does_not_flag_correctly_suppressed_node(self) -> None:
        decisions = [
            StreamDecision(
                node_name="n1", run_id="r1", tags=("nostream",), suppressed=True
            ),
        ]
        flagged = flag_inconsistencies(decisions, declared_nostream_nodes={"n1"})
        assert flagged == []

    def test_does_not_flag_node_with_no_declared_nostream_intent(self) -> None:
        decisions = [
            StreamDecision(node_name="n2", run_id="r1", tags=(), suppressed=False),
        ]
        flagged = flag_inconsistencies(decisions, declared_nostream_nodes={"n1"})
        assert flagged == []

    def test_mixed_decisions_only_flags_the_inconsistent_one(self) -> None:
        decisions = [
            StreamDecision(
                node_name="n1", run_id="r1", tags=("nostream",), suppressed=True
            ),  # correctly suppressed
            StreamDecision(
                node_name="n2", run_id="r2", tags=(), suppressed=False
            ),  # never declared nostream
            StreamDecision(
                node_name="n3", run_id="r3", tags=(), suppressed=False
            ),  # declared nostream but wasn't honored
        ]
        flagged = flag_inconsistencies(
            decisions, declared_nostream_nodes={"n1", "n3"}
        )
        assert [d.node_name for d in flagged] == ["n3"]

    def test_empty_inputs(self) -> None:
        assert flag_inconsistencies([], declared_nostream_nodes=set()) == []


# ---------------------------------------------------------------------------
# get/reset_stream_decisions — module-level state
# ---------------------------------------------------------------------------


class TestStreamDecisionsRegistry:
    def test_reset_clears_recorded_decisions(self) -> None:
        import agent_trace.integrations.langgraph_stream_debug as mod

        with mod._lock:
            mod._decisions.append(StreamDecision(node_name="n1", run_id="r1"))
        assert get_stream_decisions() != []
        reset_stream_decisions()
        assert get_stream_decisions() == []

    def test_get_returns_a_copy_not_the_live_list(self) -> None:
        import agent_trace.integrations.langgraph_stream_debug as mod

        reset_stream_decisions()
        with mod._lock:
            mod._decisions.append(StreamDecision(node_name="n1", run_id="r1"))
        snapshot = get_stream_decisions()
        snapshot.append(StreamDecision(node_name="n2", run_id="r2"))
        assert len(get_stream_decisions()) == 1
        reset_stream_decisions()
