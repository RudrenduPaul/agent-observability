"""
Unit tests for agent_trace._cli — the pure-function helpers behind
`agent-trace show`/`agent-trace replay`'s error-classification summary and
duplicate-node-span detection.

Only the data-shaping helpers are tested directly (no subprocess/argparse
wiring) — they operate on plain dicts shaped like trace.json's "spans" list,
so no LangGraph/langchain_core dependency is needed here.
"""

from __future__ import annotations

from agent_trace._cli import (
    _checkpoint_durability_summary,
    _duplicate_node_span_counts,
    _error_classification_rows,
    _print_checkpoint_durability,
    _print_duplicate_node_spans,
    _print_error_classification,
    _print_streaming_timing,
    _print_zero_task_updates,
    _streaming_timing_rows,
    _zero_task_update_rows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span(
    name: str,
    status: str = "OK",
    attributes: dict[str, object] | None = None,
) -> dict[str, object]:
    return {"name": name, "status": status, "attributes": attributes or {}}


# ---------------------------------------------------------------------------
# _error_classification_rows()
# ---------------------------------------------------------------------------


class TestErrorClassificationRows:
    def test_empty_spans_returns_empty(self) -> None:
        assert _error_classification_rows([]) == []

    def test_ok_span_excluded(self) -> None:
        spans = [_span("node:a", status="OK")]
        assert _error_classification_rows(spans) == []

    def test_error_span_included(self) -> None:
        spans = [
            _span(
                "node:a",
                status="ERROR",
                attributes={"error.origin": "application"},
            )
        ]
        rows = _error_classification_rows(spans)
        assert len(rows) == 1
        assert rows[0]["name"] == "node:a"
        assert rows[0]["origin"] == "application"

    def test_error_span_without_origin_attribute_still_included(self) -> None:
        spans = [_span("node:a", status="ERROR")]
        rows = _error_classification_rows(spans)
        assert rows[0]["origin"] == ""

    def test_known_pattern_captured_when_present(self) -> None:
        spans = [
            _span(
                "node:a",
                status="ERROR",
                attributes={
                    "error.origin": "chain",
                    "error.known_pattern": "langgraph_invalid_chat_history",
                },
            )
        ]
        rows = _error_classification_rows(spans)
        assert rows[0]["known_pattern"] == "langgraph_invalid_chat_history"

    def test_multiple_error_spans_all_included(self) -> None:
        spans = [
            _span("node:a", status="ERROR", attributes={"error.origin": "provider"}),
            _span("node:b", status="OK"),
            _span(
                "node:c", status="ERROR", attributes={"error.origin": "application"}
            ),
        ]
        rows = _error_classification_rows(spans)
        assert [r["name"] for r in rows] == ["node:a", "node:c"]

    def test_malformed_attributes_field_skipped_not_raised(self) -> None:
        spans = [{"name": "node:a", "status": "ERROR", "attributes": "not-a-dict"}]
        # Must not raise — malformed data degrades gracefully.
        assert _error_classification_rows(spans) == []


class TestPrintErrorClassification:
    def test_no_output_when_no_errors(self, capsys) -> None:
        _print_error_classification([_span("node:a", status="OK")])
        assert capsys.readouterr().out == ""

    def test_prints_header_and_row_for_error_span(self, capsys) -> None:
        spans = [
            _span(
                "node:a",
                status="ERROR",
                attributes={"error.origin": "provider"},
            )
        ]
        _print_error_classification(spans)
        out = capsys.readouterr().out
        assert "Error classification:" in out
        assert "node:a" in out
        assert "origin=provider" in out

    def test_prints_known_pattern_when_present(self, capsys) -> None:
        spans = [
            _span(
                "node:a",
                status="ERROR",
                attributes={
                    "error.origin": "chain",
                    "error.known_pattern": "middleware_invalid_tool_selection",
                },
            )
        ]
        _print_error_classification(spans)
        out = capsys.readouterr().out
        assert "known_pattern=middleware_invalid_tool_selection" in out

    def test_unclassified_origin_shown_as_unclassified(self, capsys) -> None:
        _print_error_classification([_span("node:a", status="ERROR")])
        out = capsys.readouterr().out
        assert "origin=unclassified" in out


# ---------------------------------------------------------------------------
# _duplicate_node_span_counts()
# ---------------------------------------------------------------------------


class TestDuplicateNodeSpanCounts:
    def test_empty_spans_returns_empty(self) -> None:
        assert _duplicate_node_span_counts([]) == {}

    def test_single_occurrence_not_flagged(self) -> None:
        spans = [_span("node:a"), _span("node:b")]
        assert _duplicate_node_span_counts(spans) == {}

    def test_repeated_node_span_flagged_with_count(self) -> None:
        spans = [_span("node:get_time"), _span("node:other"), _span("node:get_time")]
        assert _duplicate_node_span_counts(spans) == {"node:get_time": 2}

    def test_non_node_spans_ignored(self) -> None:
        spans = [_span("llm:gpt-4"), _span("llm:gpt-4"), _span("tool:search")]
        assert _duplicate_node_span_counts(spans) == {}

    def test_three_occurrences_counted_correctly(self) -> None:
        spans = [_span("node:loop") for _ in range(3)]
        assert _duplicate_node_span_counts(spans) == {"node:loop": 3}

    def test_multiple_duplicated_names(self) -> None:
        spans = [
            _span("node:a"),
            _span("node:a"),
            _span("node:b"),
            _span("node:b"),
            _span("node:c"),
        ]
        result = _duplicate_node_span_counts(spans)
        assert result == {"node:a": 2, "node:b": 2}


class TestPrintDuplicateNodeSpans:
    def test_no_output_when_no_duplicates(self, capsys) -> None:
        _print_duplicate_node_spans([_span("node:a"), _span("node:b")])
        assert capsys.readouterr().out == ""

    def test_prints_duplicate_with_count(self, capsys) -> None:
        spans = [_span("node:get_time"), _span("node:get_time")]
        _print_duplicate_node_spans(spans)
        out = capsys.readouterr().out
        assert "node:get_time" in out
        assert "executed 2 times" in out
        assert "Duplicate node spans" in out


# ---------------------------------------------------------------------------
# _streaming_timing_rows() / _print_streaming_timing()
# ---------------------------------------------------------------------------


def _exchange(
    url: str = "https://api.example.com/v1/chat",
    method: str = "POST",
    chunk_timestamps: list[float] | None = None,
) -> dict[str, object]:
    return {"url": url, "method": method, "chunk_timestamps": chunk_timestamps}


class TestStreamingTimingRows:
    def test_empty_exchanges_returns_empty(self) -> None:
        assert _streaming_timing_rows([]) == []

    def test_exchange_without_chunk_timestamps_skipped(self) -> None:
        assert _streaming_timing_rows([_exchange(chunk_timestamps=None)]) == []

    def test_exchange_with_chunk_timestamps_included(self) -> None:
        rows = _streaming_timing_rows(
            [_exchange(chunk_timestamps=[0.0, 0.1, 0.2])]
        )
        assert len(rows) == 1
        assert rows[0]["chunk_count"] == 3
        assert rows[0]["time_to_first_chunk_ms"] == 0.0
        assert rows[0]["max_inter_chunk_gap_ms"] == 100.0

    def test_mixed_exchanges_only_streaming_ones_included(self) -> None:
        rows = _streaming_timing_rows(
            [
                _exchange(url="https://a", chunk_timestamps=[0.0, 0.05]),
                _exchange(url="https://b", chunk_timestamps=None),
            ]
        )
        assert len(rows) == 1
        assert rows[0]["url"] == "https://a"


class TestPrintStreamingTiming:
    def test_no_output_when_no_streaming_exchanges(self, capsys) -> None:
        _print_streaming_timing([_exchange(chunk_timestamps=None)])
        assert capsys.readouterr().out == ""

    def test_prints_timing_line_for_streaming_exchange(self, capsys) -> None:
        _print_streaming_timing([_exchange(chunk_timestamps=[0.0, 0.02, 0.05])])
        out = capsys.readouterr().out
        assert "Streaming timing" in out
        assert "chunks=" in out
        assert "first_chunk=" in out
        assert "max_gap=" in out


# ---------------------------------------------------------------------------
# _checkpoint_durability_summary() / _print_checkpoint_durability()
# ---------------------------------------------------------------------------


def _write_span(
    name: str = "checkpoint:put",
    completed: bool = True,
    status: str = "OK",
) -> dict[str, object]:
    return _span(name, status=status, attributes={"checkpoint.completed": completed})


class TestCheckpointDurabilitySummary:
    def test_no_checkpoint_spans_returns_none(self) -> None:
        spans = [_span("node:a"), _span("llm:gpt-4")]
        assert _checkpoint_durability_summary(spans) is None

    def test_all_writes_flushed_is_durable(self) -> None:
        spans = [_write_span(), _write_span(name="checkpoint:put_writes")]
        summary = _checkpoint_durability_summary(spans)
        assert summary["checkpoint_status"] == "durable"
        assert summary["writes_enqueued_count"] == 2
        assert summary["writes_flushed_count"] == 2
        assert summary["cancellation_requested"] is False

    def test_zero_writes_flushed_is_abandoned(self) -> None:
        spans = [_write_span(completed=False)]
        summary = _checkpoint_durability_summary(spans)
        assert summary["checkpoint_status"] == "abandoned"
        assert summary["writes_flushed_count"] == 0

    def test_some_but_not_all_flushed_is_partial(self) -> None:
        spans = [_write_span(completed=True), _write_span(completed=False)]
        summary = _checkpoint_durability_summary(spans)
        assert summary["checkpoint_status"] == "partial"

    def test_cancellation_downgrades_fully_flushed_from_durable(self) -> None:
        spans = [
            _write_span(completed=True),
            _span("tool:x", status="CANCELLED"),
        ]
        summary = _checkpoint_durability_summary(spans)
        assert summary["cancellation_requested"] is True
        assert summary["checkpoint_status"] != "durable"

    def test_async_write_span_names_also_counted(self) -> None:
        spans = [
            _write_span(name="checkpoint:aput"),
            _write_span(name="checkpoint:aput_writes"),
        ]
        summary = _checkpoint_durability_summary(spans)
        assert summary["writes_enqueued_count"] == 2
        assert summary["writes_flushed_count"] == 2


class TestPrintCheckpointDurability:
    def test_no_output_when_no_checkpoint_spans(self, capsys) -> None:
        _print_checkpoint_durability([_span("node:a")])
        assert capsys.readouterr().out == ""

    def test_prints_durable_status(self, capsys) -> None:
        _print_checkpoint_durability([_write_span()])
        out = capsys.readouterr().out
        assert "Checkpoint durability:" in out
        assert "checkpoint_status:        durable" in out

    def test_prints_warning_when_not_durable(self, capsys) -> None:
        _print_checkpoint_durability([_write_span(completed=False)])
        out = capsys.readouterr().out
        assert "issue #5672" in out


# ---------------------------------------------------------------------------
# _zero_task_update_rows() / _print_zero_task_updates()
# ---------------------------------------------------------------------------


def _update_span(
    zero_tasks: bool, as_node: str | None = None, as_node_provided: bool = False
) -> dict[str, object]:
    attrs: dict[str, object] = {"checkpoint.zero_tasks_scheduled": zero_tasks}
    if as_node is not None:
        attrs["checkpoint.as_node"] = as_node
    attrs["checkpoint.as_node_provided"] = as_node_provided
    return _span("checkpoint:update_state", attributes=attrs)


class TestZeroTaskUpdateRows:
    def test_no_update_spans_returns_empty(self) -> None:
        assert _zero_task_update_rows([_span("node:a")]) == []

    def test_non_zero_task_update_excluded(self) -> None:
        spans = [_update_span(zero_tasks=False)]
        assert _zero_task_update_rows(spans) == []

    def test_zero_task_update_included(self) -> None:
        spans = [_update_span(zero_tasks=True, as_node="my_node", as_node_provided=True)]
        rows = _zero_task_update_rows(spans)
        assert len(rows) == 1
        assert rows[0]["as_node"] == "my_node"
        assert rows[0]["as_node_provided"] is True

    def test_zero_task_update_without_as_node(self) -> None:
        spans = [_update_span(zero_tasks=True)]
        rows = _zero_task_update_rows(spans)
        assert rows[0]["as_node"] == "<not provided>"


class TestPrintZeroTaskUpdates:
    def test_no_output_when_none_flagged(self, capsys) -> None:
        _print_zero_task_updates([_update_span(zero_tasks=False)])
        assert capsys.readouterr().out == ""

    def test_prints_warning_with_as_node(self, capsys) -> None:
        _print_zero_task_updates(
            [_update_span(zero_tasks=True, as_node="my_node", as_node_provided=True)]
        )
        out = capsys.readouterr().out
        assert "Zero tasks scheduled" in out
        assert "as_node=my_node" in out
        assert "issue #4217" in out
