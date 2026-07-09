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
    _diff_text,
    _duplicate_node_span_counts,
    _error_classification_rows,
    _error_spans,
    _exchanges_by_url,
    _http_sequence_confirms,
    _misattributed_span_rows,
    _print_checkpoint_durability,
    _print_duplicate_node_spans,
    _print_error_classification,
    _print_errors_only,
    _print_http_error_exchanges,
    _print_misattributed_spans,
    _print_retry_storms,
    _print_streaming_timing,
    _print_zero_task_updates,
    _retry_storm_rows,
    _span_exception_http_detail,
    _span_exception_message,
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


# ---------------------------------------------------------------------------
# _span_exception_message() / _error_spans() / _print_errors_only()
# ---------------------------------------------------------------------------


def _error_span_with_exception(name: str, message: str) -> dict[str, object]:
    return {
        "name": name,
        "status": "ERROR",
        "attributes": {},
        "events": [
            {
                "name": "exception",
                "attributes": {"exception.type": "ValueError", "exception.message": message},
            }
        ],
    }


class TestSpanExceptionMessage:
    def test_no_exception_event_returns_none(self) -> None:
        assert _span_exception_message(_span("node:a")) is None

    def test_exception_event_returns_message(self) -> None:
        span = _error_span_with_exception("node:a", "boom")
        assert _span_exception_message(span) == "boom"

    def test_ignores_non_exception_events(self) -> None:
        span = {
            "name": "node:a",
            "status": "OK",
            "attributes": {},
            "events": [{"name": "other_event", "attributes": {}}],
        }
        assert _span_exception_message(span) is None


def _error_span_with_http_body(
    name: str, message: str, status_code: int, body: str
) -> dict[str, object]:
    return {
        "name": name,
        "status": "ERROR",
        "attributes": {},
        "events": [
            {
                "name": "exception",
                "attributes": {
                    "exception.type": "HTTPError",
                    "exception.message": message,
                    "exception.http_status_code": status_code,
                    "exception.http_response_body": body,
                },
            }
        ],
    }


class TestSpanExceptionHttpDetail:
    def test_no_exception_event_returns_none(self) -> None:
        assert _span_exception_http_detail(_span("node:a")) is None

    def test_exception_without_http_body_returns_none(self) -> None:
        span = _error_span_with_exception("node:a", "boom")
        assert _span_exception_http_detail(span) is None

    def test_exception_with_http_body_returns_formatted_detail(self) -> None:
        span = _error_span_with_http_body(
            "llm:bedrock", "400 Client Error", 400, "Malformed input request"
        )
        detail = _span_exception_http_detail(span)
        assert detail == "HTTP 400: Malformed input request"


class TestErrorSpans:
    def test_filters_to_error_status_only(self) -> None:
        spans = [_span("node:a", status="OK"), _span("node:b", status="ERROR")]
        result = _error_spans(spans)
        assert [s["name"] for s in result] == ["node:b"]

    def test_empty_input_returns_empty(self) -> None:
        assert _error_spans([]) == []


class TestPrintErrorsOnly:
    def test_no_errors_prints_zero_count(self, capsys) -> None:
        _print_errors_only([_span("node:a", status="OK")])
        out = capsys.readouterr().out
        assert "Error spans: 0 of 1 total" in out

    def test_error_span_prints_name_and_message(self, capsys) -> None:
        spans = [_error_span_with_exception("llm:gpt-4", "rate limit exceeded")]
        _print_errors_only(spans)
        out = capsys.readouterr().out
        assert "Error spans: 1 of 1 total" in out
        assert "[ERR] llm:gpt-4" in out
        assert "rate limit exceeded" in out

    def test_non_error_span_excluded_from_detail(self, capsys) -> None:
        spans = [
            _span("node:a", status="OK"),
            _error_span_with_exception("llm:gpt-4", "boom"),
        ]
        _print_errors_only(spans)
        out = capsys.readouterr().out
        assert "[ERR] node:a" not in out
        assert "[ERR] llm:gpt-4" in out

    def test_error_span_prints_http_response_body_when_present(self, capsys) -> None:
        spans = [
            _error_span_with_http_body(
                "llm:bedrock", "400 Client Error", 400, "Malformed input request"
            )
        ]
        _print_errors_only(spans)
        out = capsys.readouterr().out
        assert "HTTP 400: Malformed input request" in out

    def test_error_span_no_http_line_when_absent(self, capsys) -> None:
        spans = [_error_span_with_exception("llm:gpt-4", "boom")]
        _print_errors_only(spans)
        out = capsys.readouterr().out
        assert "HTTP" not in out


# ---------------------------------------------------------------------------
# _print_http_error_exchanges()
# ---------------------------------------------------------------------------


class TestPrintHttpErrorExchanges:
    def test_no_output_when_all_2xx(self, capsys) -> None:
        exchanges = [{"url": "https://a", "method": "GET", "response_status": 200}]
        _print_http_error_exchanges(exchanges)
        assert capsys.readouterr().out == ""

    def test_prints_4xx_5xx_exchanges(self, capsys) -> None:
        exchanges = [
            {"url": "https://a", "method": "GET", "response_status": 200},
            {"url": "https://b", "method": "POST", "response_status": 500},
        ]
        _print_http_error_exchanges(exchanges)
        out = capsys.readouterr().out
        assert "HTTP error exchanges (1):" in out
        assert "https://b" in out
        assert "HTTP 500" in out
        assert "https://a" not in out


# ---------------------------------------------------------------------------
# _retry_storm_rows() / _print_retry_storms()
# ---------------------------------------------------------------------------


def _node_span(span_id: str, name: str = "node:a") -> dict[str, object]:
    return {"span_id": span_id, "parent_id": None, "name": name, "status": "OK"}


def _llm_child_span(span_id: str, parent_id: str) -> dict[str, object]:
    return {"span_id": span_id, "parent_id": parent_id, "name": "llm:gpt-4", "status": "OK"}


class TestRetryStormRows:
    def test_single_llm_child_not_flagged(self) -> None:
        spans = [_node_span("n1"), _llm_child_span("l1", "n1")]
        assert _retry_storm_rows(spans) == []

    def test_multiple_llm_children_flagged(self) -> None:
        spans = [
            _node_span("n1"),
            _llm_child_span("l1", "n1"),
            _llm_child_span("l2", "n1"),
            _llm_child_span("l3", "n1"),
        ]
        rows = _retry_storm_rows(spans)
        assert len(rows) == 1
        assert rows[0]["node"] == "node:a"
        assert rows[0]["llm_child_count"] == 3

    def test_non_node_span_ignored(self) -> None:
        spans = [
            {"span_id": "t1", "parent_id": None, "name": "tool:x", "status": "OK"},
            _llm_child_span("l1", "t1"),
            _llm_child_span("l2", "t1"),
        ]
        assert _retry_storm_rows(spans) == []

    def test_tool_child_spans_not_counted(self) -> None:
        spans = [
            _node_span("n1"),
            {"span_id": "t1", "parent_id": "n1", "name": "tool:x", "status": "OK"},
            {"span_id": "t2", "parent_id": "n1", "name": "tool:y", "status": "OK"},
        ]
        assert _retry_storm_rows(spans) == []


class TestPrintRetryStorms:
    def test_no_output_when_no_storms(self, capsys) -> None:
        _print_retry_storms([_node_span("n1"), _llm_child_span("l1", "n1")])
        assert capsys.readouterr().out == ""

    def test_prints_storm_summary(self, capsys) -> None:
        spans = [
            _node_span("n1"),
            _llm_child_span("l1", "n1"),
            _llm_child_span("l2", "n1"),
        ]
        _print_retry_storms(spans)
        out = capsys.readouterr().out
        assert "Repeated LLM calls" in out
        assert "node:a" in out
        assert "issue #2920" in out


# ---------------------------------------------------------------------------
# _misattributed_span_rows() / _print_misattributed_spans()
# ---------------------------------------------------------------------------


class TestMisattributedSpanRows:
    def test_single_root_not_flagged(self) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 1.0}
        ]
        assert _misattributed_span_rows(spans) == []

    def test_sequential_non_overlapping_roots_not_flagged(self) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 1.0},
            {"span_id": "s2", "parent_id": None, "name": "node:b", "start_time": 2.0, "end_time": 3.0},
        ]
        assert _misattributed_span_rows(spans) == []

    def test_overlapping_root_flagged_with_likely_parent(self) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 2.0},
            {"span_id": "s2", "parent_id": None, "name": "llm:gpt-4", "start_time": 0.5, "end_time": 1.0},
        ]
        rows = _misattributed_span_rows(spans)
        assert len(rows) == 1
        assert rows[0]["span"] == "llm:gpt-4"
        assert rows[0]["likely_parent"] == "node:a"

    def test_empty_spans_returns_empty(self) -> None:
        assert _misattributed_span_rows([]) == []


class TestPrintMisattributedSpans:
    def test_no_output_when_none_flagged(self, capsys) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 1.0}
        ]
        _print_misattributed_spans(spans)
        assert capsys.readouterr().out == ""

    def test_prints_flagged_span(self, capsys) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 2.0},
            {"span_id": "s2", "parent_id": None, "name": "llm:gpt-4", "start_time": 0.5, "end_time": 1.0},
        ]
        _print_misattributed_spans(spans)
        out = capsys.readouterr().out
        assert "Possibly misattributed spans" in out
        assert "llm:gpt-4" in out
        assert "langgraph#3975" in out


# ---------------------------------------------------------------------------
# _http_sequence_confirms() — reconciling callback-derived span attribution
# against fixture.db's sequence_num-ordered HTTP capture.
# ---------------------------------------------------------------------------


_SEQ_PARENT = {"name": "node:a", "start_time": 0.0, "end_time": 2.0}
_SEQ_SUSPECT = {"name": "llm:gpt-4", "start_time": 0.5, "end_time": 1.0}


class TestHttpSequenceConfirms:
    def test_confirms_when_http_sequence_agrees(self) -> None:
        exchanges = [
            {"recorded_at": 0.1, "sequence_num": 1},  # inside parent's window
            {"recorded_at": 0.7, "sequence_num": 2},  # inside suspect's window
        ]
        assert (
            _http_sequence_confirms(_SEQ_SUSPECT, _SEQ_PARENT, exchanges) is True
        )

    def test_contradicts_when_http_sequence_disagrees(self) -> None:
        exchanges = [
            # Suspect's own HTTP call was recorded (lower sequence_num)
            # *before* the exchange inside the guessed parent's window —
            # the HTTP layer disagrees with the wall-clock-timestamp guess.
            {"recorded_at": 0.7, "sequence_num": 1},
            {"recorded_at": 0.1, "sequence_num": 2},
        ]
        assert (
            _http_sequence_confirms(_SEQ_SUSPECT, _SEQ_PARENT, exchanges) is False
        )

    def test_none_when_no_exchanges_in_either_window(self) -> None:
        exchanges = [{"recorded_at": 10.0, "sequence_num": 1}]
        assert _http_sequence_confirms(_SEQ_SUSPECT, _SEQ_PARENT, exchanges) is None

    def test_none_when_no_exchanges_at_all(self) -> None:
        assert _http_sequence_confirms(_SEQ_SUSPECT, _SEQ_PARENT, []) is None

    def test_ignores_malformed_exchange_rows(self) -> None:
        exchanges = [
            {"recorded_at": None, "sequence_num": 1},
            {"recorded_at": 0.1, "sequence_num": "not-an-int"},
        ]
        assert _http_sequence_confirms(_SEQ_SUSPECT, _SEQ_PARENT, exchanges) is None


class TestMisattributedSpanRowsWithExchanges:
    def test_row_carries_confirmed_flag_when_exchanges_agree(self) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 2.0},
            {"span_id": "s2", "parent_id": None, "name": "llm:gpt-4", "start_time": 0.5, "end_time": 1.0},
        ]
        exchanges = [
            {"recorded_at": 0.1, "sequence_num": 1},
            {"recorded_at": 0.7, "sequence_num": 2},
        ]
        rows = _misattributed_span_rows(spans, exchanges)
        assert len(rows) == 1
        assert rows[0]["http_sequence_confirmed"] is True

    def test_row_omits_confirmed_flag_when_no_exchanges_passed(self) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 2.0},
            {"span_id": "s2", "parent_id": None, "name": "llm:gpt-4", "start_time": 0.5, "end_time": 1.0},
        ]
        rows = _misattributed_span_rows(spans)
        assert "http_sequence_confirmed" not in rows[0]


class TestPrintMisattributedSpansWithExchanges:
    def test_prints_confirmation_suffix(self, capsys) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 2.0},
            {"span_id": "s2", "parent_id": None, "name": "llm:gpt-4", "start_time": 0.5, "end_time": 1.0},
        ]
        exchanges = [
            {"recorded_at": 0.1, "sequence_num": 1},
            {"recorded_at": 0.7, "sequence_num": 2},
        ]
        _print_misattributed_spans(spans, exchanges)
        out = capsys.readouterr().out
        assert "confirmed via HTTP sequence_num ordering" in out
        assert "sequence_num-ordered HTTP capture" in out

    def test_prints_contradiction_suffix(self, capsys) -> None:
        spans = [
            {"span_id": "s1", "parent_id": None, "name": "node:a", "start_time": 0.0, "end_time": 2.0},
            {"span_id": "s2", "parent_id": None, "name": "llm:gpt-4", "start_time": 0.5, "end_time": 1.0},
        ]
        exchanges = [
            {"recorded_at": 0.7, "sequence_num": 1},
            {"recorded_at": 0.1, "sequence_num": 2},
        ]
        _print_misattributed_spans(spans, exchanges)
        out = capsys.readouterr().out
        assert "does NOT confirm" in out


# ---------------------------------------------------------------------------
# _exchanges_by_url() / _diff_text()
# ---------------------------------------------------------------------------


class TestExchangesByUrl:
    def test_groups_by_url(self) -> None:
        exchanges = [
            {"url": "https://a", "request_body": "1"},
            {"url": "https://b", "request_body": "2"},
            {"url": "https://a", "request_body": "3"},
        ]
        result = _exchanges_by_url(exchanges)
        assert len(result["https://a"]) == 2
        assert len(result["https://b"]) == 1

    def test_empty_list_returns_empty_dict(self) -> None:
        assert _exchanges_by_url([]) == {}


class TestDiffText:
    def test_identical_text_produces_no_diff_lines(self) -> None:
        lines = _diff_text("a", "same\n", "b", "same\n")
        assert lines == []

    def test_different_text_produces_diff_lines(self) -> None:
        lines = _diff_text("a", "line1\n", "b", "line2\n")
        assert any("line1" in line for line in lines)
        assert any("line2" in line for line in lines)


# ---------------------------------------------------------------------------
# `agent-trace inspect` / `agent-trace diff` / `agent-trace show --errors-only`
# — exercised as real subprocesses against a real Fixture/trace.json, the
# same style TestRunSubcommand already uses for `agent-trace run`.
# ---------------------------------------------------------------------------


def _write_run(
    trace_dir,
    run_id: str,
    *,
    exchanges: list[dict[str, object]] | None = None,
    spans: list[dict[str, object]] | None = None,
) -> None:
    import json as _json

    from agent_trace._replay.fixture import Fixture

    run_dir = trace_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if exchanges:
        with Fixture(run_dir / "fixture.db") as fixture:
            for exchange in exchanges:
                fixture.record_exchange(**exchange)

    trace_data = {
        "trace_id": "t1",
        "run_id": run_id,
        "metadata": {"name": "test"},
        "spans": spans or [],
    }
    (run_dir / "trace.json").write_text(_json.dumps(trace_data), encoding="utf-8")


class TestInspectSubcommand:
    def test_flags_orphaned_tool_call_id_and_http_error(self, tmp_path) -> None:
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        request_body = _json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "call_1", "function": {"name": "x"}}],
                    }
                ]
            }
        )
        _write_run(
            tmp_path,
            "run1",
            exchanges=[
                {
                    "url": "https://api.openai.com/v1/chat/completions",
                    "method": "POST",
                    "request_headers": {},
                    "request_body": request_body,
                    "response_status": 500,
                    "response_headers": {},
                    "response_body": "server error",
                }
            ],
        )

        result = _run_cli(["inspect", "run1"], env=env)
        assert result.returncode == 0, result.stderr
        assert "orphaned_tool_call_ids" in result.stdout
        assert "http_error_status" in result.stdout

    def test_no_run_directory_exits_nonzero(self, tmp_path) -> None:
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        result = _run_cli(["inspect", "does-not-exist"], env=env)
        assert result.returncode != 0

    def test_no_anomalies_reports_clean(self, tmp_path) -> None:
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)
        _write_run(tmp_path, "run-clean")

        result = _run_cli(["inspect", "run-clean"], env=env)
        assert result.returncode == 0, result.stderr
        assert "No anomalies flagged" in result.stdout

    def test_flags_orphaned_responses_api_call_id(self, tmp_path) -> None:
        """#33895: a Responses API `function_call` with no matching
        `function_call_output` — the "No call message found for call_*"
        shape — must surface through `agent-trace inspect` the same way
        the Chat Completions orphaned-tool_call_id shape does."""
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        request_body = _json.dumps(
            {
                "input": [
                    {"type": "function_call", "call_id": "call_1", "name": "get_weather"},
                ]
            }
        )
        _write_run(
            tmp_path,
            "run-responses-api",
            exchanges=[
                {
                    "url": "https://api.openai.com/v1/responses",
                    "method": "POST",
                    "request_headers": {},
                    "request_body": request_body,
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": "{}",
                }
            ],
        )

        result = _run_cli(["inspect", "run-responses-api"], env=env)
        assert result.returncode == 0, result.stderr
        assert "orphaned_responses_api_call_ids" in result.stdout

    def test_tool_call_absent_from_request_tools_flagged_unconditionally(
        self, tmp_path
    ) -> None:
        """#6037: `transfer_back_to_supervisor is not a valid tool` inside a
        LangGraph supervisor topology — surfaced by `agent-trace inspect`
        with no CLI flag needed, since the check only needs this one
        exchange's own request/response pair."""
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        request_body = _json.dumps(
            {"tools": [{"type": "function", "function": {"name": "search"}}]}
        )
        response_body = _json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "transfer_back_to_supervisor"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )
        _write_run(
            tmp_path,
            "run-supervisor",
            exchanges=[
                {
                    "url": "https://api.openai.com/v1/chat/completions",
                    "method": "POST",
                    "request_headers": {},
                    "request_body": request_body,
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": response_body,
                },
            ],
        )

        result = _run_cli(["inspect", "run-supervisor"], env=env)
        assert result.returncode == 0, result.stderr
        assert "tool_call_name_absent_from_request_tools" in result.stdout
        assert "transfer_back_to_supervisor" in result.stdout

    def test_registered_tools_flags_unregistered_tool_call_name(self, tmp_path) -> None:
        """#325: --registered-tools enables check_tool_call_name_not_registered
        (unconditional, no edit-distance threshold) alongside the existing
        fuzzy_match/dotted_compound/action_name_not_registered checks."""
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        response_body = _json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "completely_made_up_tool"}}
                            ]
                        }
                    }
                ]
            }
        )
        _write_run(
            tmp_path,
            "run-unregistered-tool",
            exchanges=[
                {
                    "url": "https://api.openai.com/v1/chat/completions",
                    "method": "POST",
                    "request_headers": {},
                    "request_body": "{}",
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": response_body,
                },
            ],
        )

        result = _run_cli(
            ["inspect", "run-unregistered-tool", "--registered-tools", "search"],
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "tool_call_name_not_registered" in result.stdout
        assert "completely_made_up_tool" in result.stdout

    def test_diff_get_post_field_flags_stale_instructions(self, tmp_path) -> None:
        """#2620 (GPTAssistantAgent): a POST /runs referencing the same
        assistant_id as an earlier GET /assistants/{id} sends a stale
        ``instructions`` value that no longer matches what the GET returned —
        should be auto-flagged via `agent-trace inspect --diff-get-post-field`
        without a developer having to hand-diff the two bodies."""
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        get_response = _json.dumps(
            {"id": "asst_123", "instructions": "You are a fresh, updated agent."}
        )
        post_request = _json.dumps(
            {"assistant_id": "asst_123", "instructions": "You are a STALE agent."}
        )
        _write_run(
            tmp_path,
            "run-stale",
            exchanges=[
                {
                    "url": "https://api.openai.com/v1/assistants/asst_123",
                    "method": "GET",
                    "request_headers": {},
                    "request_body": "",
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": get_response,
                },
                {
                    "url": "https://api.openai.com/v1/threads/t1/runs",
                    "method": "POST",
                    "request_headers": {},
                    "request_body": post_request,
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": "{}",
                },
            ],
        )

        result = _run_cli(
            [
                "inspect",
                "run-stale",
                "--diff-get-post-field",
                "instructions",
                "--diff-get-post-post-id-field",
                "assistant_id",
            ],
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "get_post_field_mismatch" in result.stdout
        assert "STALE agent" in result.stdout

    def test_diff_get_post_field_absent_when_flag_not_passed(self, tmp_path) -> None:
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        get_response = _json.dumps({"id": "asst_123", "instructions": "fresh"})
        post_request = _json.dumps({"assistant_id": "asst_123", "instructions": "stale"})
        _write_run(
            tmp_path,
            "run-stale-2",
            exchanges=[
                {
                    "url": "https://api.openai.com/v1/assistants/asst_123",
                    "method": "GET",
                    "request_headers": {},
                    "request_body": "",
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": get_response,
                },
                {
                    "url": "https://api.openai.com/v1/threads/t1/runs",
                    "method": "POST",
                    "request_headers": {},
                    "request_body": post_request,
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": "{}",
                },
            ],
        )

        result = _run_cli(["inspect", "run-stale-2"], env=env)
        assert result.returncode == 0, result.stderr
        assert "get_post_field_mismatch" not in result.stdout

    def test_diff_field_nested_path_flags_deepseek_reasoning_content(self, tmp_path) -> None:
        """#5526: DeepSeek's `reasoning_content` lives nested at
        `choices[0].message.reasoning_content`, not at the response body's
        top level — `--diff-field` must accept a dotted/nested path
        (reusing `_get_path`) to see it at all, and flag that no span
        attribute reflects it (create_tool_calling_agent drops it)."""
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        response_body = _json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "final answer",
                            "reasoning_content": "step-by-step reasoning...",
                        }
                    }
                ]
            }
        )
        _write_run(
            tmp_path,
            "run-deepseek",
            exchanges=[
                {
                    "url": "https://api.deepseek.com/chat/completions",
                    "method": "POST",
                    "request_headers": {},
                    "request_body": "{}",
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": response_body,
                },
            ],
            spans=[
                {
                    "name": "llm:x",
                    "status": "OK",
                    "attributes": {"llm.content": "final answer"},
                    "events": [],
                },
            ],
        )

        result = _run_cli(
            ["inspect", "run-deepseek", "--diff-field", "choices.0.message.reasoning_content"],
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "field_present_on_wire_absent_downstream" in result.stdout


class TestDiffSubcommand:
    def test_diffs_matching_url_request_bodies(self, tmp_path) -> None:
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        common_kwargs = {
            "url": "https://api.openai.com/v1/chat/completions",
            "method": "POST",
            "request_headers": {},
            "response_status": 200,
            "response_headers": {},
            "response_body": "{}",
        }
        _write_run(
            tmp_path,
            "run-a",
            exchanges=[{**common_kwargs, "request_body": '{"prompt": "hello"}'}],
        )
        _write_run(
            tmp_path,
            "run-b",
            exchanges=[{**common_kwargs, "request_body": '{"prompt": "goodbye"}'}],
        )

        result = _run_cli(["diff", "run-a", "run-b"], env=env)
        assert result.returncode == 0, result.stderr
        assert "hello" in result.stdout
        assert "goodbye" in result.stdout

    def test_identical_runs_report_no_differences(self, tmp_path) -> None:
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        common_kwargs = {
            "url": "https://api.openai.com/v1/chat/completions",
            "method": "POST",
            "request_headers": {},
            "request_body": "{}",
            "response_status": 200,
            "response_headers": {},
            "response_body": "{}",
        }
        _write_run(tmp_path, "run-a", exchanges=[dict(common_kwargs)])
        _write_run(tmp_path, "run-b", exchanges=[dict(common_kwargs)])

        result = _run_cli(["diff", "run-a", "run-b"], env=env)
        assert result.returncode == 0, result.stderr
        assert "No differences found" in result.stdout

    def test_url_only_in_one_run_reported(self, tmp_path) -> None:
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        _write_run(
            tmp_path,
            "run-a",
            exchanges=[
                {
                    "url": "https://only-in-a.example.com",
                    "method": "GET",
                    "request_headers": {},
                    "request_body": "",
                    "response_status": 200,
                    "response_headers": {},
                    "response_body": "{}",
                }
            ],
        )
        _write_run(tmp_path, "run-b")

        result = _run_cli(["diff", "run-a", "run-b"], env=env)
        assert result.returncode == 0, result.stderr
        assert "only present in run-a" in result.stdout

    def test_restarted_run_flags_restart_vs_resume(self, tmp_path) -> None:
        """#161: run-b's root chain span shares thread_id="thread-1" with
        run-a but restarts at langgraph_step=0 instead of continuing past
        run-a's last recorded step — `agent-trace diff` must surface this
        without requiring a developer to hand-read both trace.json files."""
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        def _root_span(thread_id: str, step: int) -> dict[str, object]:
            return {
                "span_id": "root",
                "trace_id": "t1",
                "parent_id": None,
                "name": "node:graph",
                "start_time": 0.0,
                "end_time": 1.0,
                "status": "OK",
                "attributes": {
                    "chain.metadata": _json.dumps(
                        {"thread_id": thread_id, "langgraph_step": step}
                    )
                },
                "events": [],
            }

        _write_run(tmp_path, "run-a", spans=[_root_span("thread-1", 2)])
        _write_run(tmp_path, "run-b", spans=[_root_span("thread-1", 0)])

        result = _run_cli(["diff", "run-a", "run-b"], env=env)
        assert result.returncode == 0, result.stderr
        assert "restart_vs_resume" in result.stdout
        assert "thread-1" in result.stdout

    def test_resumed_run_does_not_flag_restart_vs_resume(self, tmp_path) -> None:
        import json as _json
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        def _root_span(thread_id: str, step: int) -> dict[str, object]:
            return {
                "span_id": "root",
                "trace_id": "t1",
                "parent_id": None,
                "name": "node:graph",
                "start_time": 0.0,
                "end_time": 1.0,
                "status": "OK",
                "attributes": {
                    "chain.metadata": _json.dumps(
                        {"thread_id": thread_id, "langgraph_step": step}
                    )
                },
                "events": [],
            }

        _write_run(tmp_path, "run-a", spans=[_root_span("thread-1", 2)])
        _write_run(tmp_path, "run-b", spans=[_root_span("thread-1", 3)])

        result = _run_cli(["diff", "run-a", "run-b"], env=env)
        assert result.returncode == 0, result.stderr
        assert "restart_vs_resume" not in result.stdout


class TestShowErrorsOnlyFlag:
    def test_errors_only_filters_output(self, tmp_path) -> None:
        import os

        env = dict(os.environ)
        env["AGENT_TRACE_TRACE_DIR"] = str(tmp_path)

        spans = [
            {
                "span_id": "s1",
                "trace_id": "t1",
                "parent_id": None,
                "name": "node:a",
                "start_time": 0.0,
                "end_time": 1.0,
                "status": "ERROR",
                "attributes": {},
                "events": [
                    {
                        "name": "exception",
                        "timestamp": 0.5,
                        "attributes": {
                            "exception.type": "ValueError",
                            "exception.message": "boom-details",
                        },
                    }
                ],
            },
            {
                "span_id": "s2",
                "trace_id": "t1",
                "parent_id": None,
                "name": "node:b",
                "start_time": 2.0,
                "end_time": 3.0,
                "status": "OK",
                "attributes": {},
                "events": [],
            },
        ]
        _write_run(tmp_path, "run1", spans=spans)

        result = _run_cli(["show", "run1", "--errors-only"], env=env)
        assert result.returncode == 0, result.stderr
        assert "boom-details" in result.stdout
        assert "Error spans: 1 of 2 total" in result.stdout
