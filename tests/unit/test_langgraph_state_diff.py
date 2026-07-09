"""
Unit tests for agent_trace.integrations.langgraph_state_diff — the pure
helper functions behind the per-superstep state-merge diagnostic.

These test only the module-level pure functions (_value_survived,
_superstep_key, _safe_eq), which don't require langgraph to be installed —
importing the module itself is safe without langgraph since the real
BaseCheckpointSaver import only happens lazily inside wrap_checkpointer().
End-to-end coverage against a real checkpointer lives in
tests/integration/test_langgraph_state_diff.py.
"""

from __future__ import annotations

from agent_trace.integrations.langgraph_state_diff import (
    _safe_eq,
    _superstep_key,
    _value_survived,
)

# ---------------------------------------------------------------------------
# _superstep_key()
# ---------------------------------------------------------------------------


class TestSuperstepKey:
    def test_returns_none_without_thread_id(self) -> None:
        assert _superstep_key({"configurable": {}}) is None

    def test_returns_none_for_none_config(self) -> None:
        assert _superstep_key(None) is None

    def test_extracts_thread_id_ns_and_checkpoint_id(self) -> None:
        config = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "ns1",
                "checkpoint_id": "cp1",
            }
        }
        assert _superstep_key(config) == ("t1", "ns1", "cp1")

    def test_defaults_checkpoint_ns_to_empty_string(self) -> None:
        config = {"configurable": {"thread_id": "t1", "checkpoint_id": "cp1"}}
        assert _superstep_key(config) == ("t1", "", "cp1")

    def test_missing_checkpoint_id_is_none(self) -> None:
        config = {"configurable": {"thread_id": "t1"}}
        assert _superstep_key(config) == ("t1", "", None)


# ---------------------------------------------------------------------------
# _safe_eq()
# ---------------------------------------------------------------------------


class TestSafeEq:
    def test_equal_values(self) -> None:
        assert _safe_eq("a", "a") is True

    def test_unequal_values(self) -> None:
        assert _safe_eq("a", "b") is False

    def test_swallows_comparison_exception(self) -> None:
        class Explodes:
            def __eq__(self, other: object) -> bool:
                raise RuntimeError("no comparison for you")

            __hash__ = None  # type: ignore[assignment]

        assert _safe_eq(Explodes(), "x") is False


# ---------------------------------------------------------------------------
# _value_survived()
# ---------------------------------------------------------------------------


class TestValueSurvived:
    def test_last_value_wins_channel_direct_match(self) -> None:
        assert _value_survived("from-task-b", "from-task-b") is True

    def test_last_value_wins_channel_dropped(self) -> None:
        assert _value_survived("from-task-a", "from-task-b") is False

    def test_reducer_channel_member_survives(self) -> None:
        assert _value_survived("a", ["a", "b"]) is True
        assert _value_survived("b", ["a", "b"]) is True

    def test_reducer_channel_member_dropped(self) -> None:
        assert _value_survived("c", ["a", "b"]) is False

    def test_tuple_final_value_also_checked(self) -> None:
        assert _value_survived("a", ("a", "b")) is True

    def test_none_final_value_is_dropped(self) -> None:
        assert _value_survived("a", None) is False
