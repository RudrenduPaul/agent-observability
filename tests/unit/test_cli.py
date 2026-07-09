"""
Unit tests for agent_trace._cli — the pure-function helpers behind
`agent-trace show`/`agent-trace replay`'s error-classification summary and
duplicate-node-span detection, plus `agent-trace run`'s command-line parsing
and subprocess wiring.

Most of this file tests pure data-shaping helpers directly (no
subprocess/argparse wiring) — they operate on plain dicts shaped like
trace.json's "spans" list, so no LangGraph/langchain_core dependency is
needed here. `TestCmdRun`/`TestStripLeadingSeparator`/`TestRunSubcommand
ArgParsing` are the exception: `agent-trace run` is inherently a
subprocess-wrapping command, so it's tested by actually invoking
`python -m agent_trace._cli run ...` in a real subprocess.
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
    _strip_leading_separator,
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


# ---------------------------------------------------------------------------
# _strip_leading_separator() — pure function behind `agent-trace run`
# ---------------------------------------------------------------------------


class TestStripLeadingSeparator:
    def test_strips_leading_double_dash(self) -> None:
        assert _strip_leading_separator(["--", "langgraph", "dev"]) == [
            "langgraph",
            "dev",
        ]

    def test_leaves_command_without_separator_unchanged(self) -> None:
        assert _strip_leading_separator(["langgraph", "dev"]) == ["langgraph", "dev"]

    def test_empty_list_stays_empty(self) -> None:
        assert _strip_leading_separator([]) == []

    def test_only_strips_the_leading_separator_not_later_ones(self) -> None:
        assert _strip_leading_separator(["--", "echo", "--", "x"]) == [
            "echo",
            "--",
            "x",
        ]


# ---------------------------------------------------------------------------
# `agent-trace run` — subprocess-wrapping CLI command. Genuinely exercised
# via a real subprocess (python -m agent_trace._cli run ...) since its whole
# job is to launch a child process with recording pre-enabled.
# ---------------------------------------------------------------------------


def _run_cli(args: list[str], env: dict[str, str]):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "agent_trace._cli", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class TestRunSubcommand:
    def test_no_command_given_exits_nonzero_with_usage(self, tmp_path) -> None:
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        result = _run_cli(["run"], env=env)
        assert result.returncode != 0
        assert "no command given" in (result.stdout + result.stderr)

    def test_execs_child_and_relays_exit_code(self, tmp_path) -> None:
        import os
        import sys

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        result = _run_cli(
            ["run", "--run-id", "cli-exit-test", "--", sys.executable, "-c", "exit(7)"],
            env=env,
        )
        assert result.returncode == 7

    def test_sets_auto_record_env_vars_for_the_child(self, tmp_path) -> None:
        """The child process, importing agent_trace itself, must observe
        AGENT_TRACE_AUTO_RECORD=1 and record an HTTP exchange into the
        run_id/trace_dir agent-trace run selected — proving the env vars
        set by cmd_run() actually reach and activate the child."""
        import os
        import sys

        from agent_trace._replay.fixture import Fixture

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        child_script = (
            "import agent_trace, httpx\n"
            "client = httpx.Client(transport=httpx.MockTransport("
            "lambda r: httpx.Response(200, json={'ok': True})))\n"
            "client.get('https://api.example.com/from-child')\n"
        )

        result = _run_cli(
            ["run", "--run-id", "cli-env-test", "--", sys.executable, "-c", child_script],
            env=env,
        )
        assert result.returncode == 0, result.stderr

        with Fixture(tmp_path / "cli-env-test" / "fixture.db") as fixture:
            exchanges = fixture.all_exchanges()
        assert [e["url"] for e in exchanges] == ["https://api.example.com/from-child"]

    def test_uses_custom_run_id_when_given(self, tmp_path) -> None:
        # Run dir creation is driven by the child process's own
        # start_auto_record() call (triggered by `import agent_trace`
        # observing AGENT_TRACE_AUTO_RECORD=1) — a child that never imports
        # agent_trace creates no run dir at all, so use one that does.
        import os
        import sys

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        result = _run_cli(
            [
                "run",
                "--run-id",
                "my-custom-run",
                "--",
                sys.executable,
                "-c",
                "import agent_trace",
            ],
            env=env,
        )
        assert result.returncode == 0
        assert (tmp_path / "my-custom-run").is_dir()

    def test_generates_a_run_id_when_not_given(self, tmp_path) -> None:
        import os
        import sys

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        result = _run_cli(
            ["run", "--", sys.executable, "-c", "import agent_trace"], env=env
        )
        assert result.returncode == 0
        run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        assert run_dirs[0].name.startswith("run_")
