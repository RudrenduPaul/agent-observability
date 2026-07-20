"""
Unit tests for agent_trace exporters:
  - StdoutExporter
  - FileExporter
  - OTLPExporter (import-error path)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_trace.core.span import Span, SpanStatus
from agent_trace.core.trace import Trace
from agent_trace.exporters.file import FileExporter
from agent_trace.exporters.stdout import StdoutExporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(n_spans: int = 2) -> Trace:
    """Create a core.Trace instance compatible with StdoutExporter.

    StdoutExporter references trace.name, trace.end_time, and trace.start_time
    which are not dataclass fields on core.Trace. We patch them in dynamically
    so the exporter can run without raising AttributeError.
    """
    trace = Trace(trace_id="t-exporter", run_id="run-exporter")
    trace.metadata["name"] = "test-trace"
    # StdoutExporter reads trace.name, trace.end_time, trace.start_time
    # directly — add them as instance attributes to satisfy the exporter
    trace.name = "test-trace"  # type: ignore[attr-defined]
    trace.end_time = None  # type: ignore[attr-defined]
    trace.start_time = None  # type: ignore[attr-defined]
    for i in range(n_spans):
        s = Span(name=f"span-{i}", span_id=f"s-{i:03d}", trace_id="t-exporter")
        s.set_attribute("step", i)
        s.end()
        trace.add_span(s)
    return trace


# ---------------------------------------------------------------------------
# StdoutExporter
# ---------------------------------------------------------------------------


class TestStdoutExporter:
    def test_export_does_not_raise(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = _make_trace(2)
        exporter.export(trace)  # must not raise

    def test_export_outputs_to_stdout(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = _make_trace(2)
        exporter.export(trace)
        captured = capsys.readouterr()
        # Something must have been printed (either rich or plain)
        assert len(captured.out) > 0

    def test_export_contains_span_names(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = _make_trace(2)
        exporter.export(trace)
        captured = capsys.readouterr()
        # At least one span name should appear in output
        assert "span-0" in captured.out or "span-1" in captured.out

    def test_export_empty_trace_does_not_raise(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = _make_trace(0)
        exporter.export(trace)  # must not raise

    def test_export_span_plain_text(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="my-span")
        span.end()
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "my-span" in captured.out

    def test_export_span_with_depth_indented(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="deep-span")
        span.end()
        exporter.export_span(span, depth=2)
        captured = capsys.readouterr()
        # Depth=2 means 4 spaces of indent
        assert "    " in captured.out

    def test_export_span_shows_status_symbol(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="err-span")
        span.end(SpanStatus.ERROR)
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "[ERR]" in captured.out or "ERROR" in captured.out

    def test_export_span_unset_shows_symbol(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="unset-span")
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "[---]" in captured.out or "UNSET" in captured.out

    def test_export_span_shows_duration_when_ended(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="timed-span")
        span.end()
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "ms" in captured.out

    def test_plain_text_fallback_when_rich_unavailable(self, capsys) -> None:
        """With rich import blocked, export() must fall back to plain text."""
        import sys
        import unittest.mock

        exporter = StdoutExporter()
        trace = _make_trace(2)

        with unittest.mock.patch.dict(
            sys.modules, {"rich": None, "rich.console": None, "rich.tree": None}
        ):
            exporter.export(trace)

        captured = capsys.readouterr()
        assert "Trace:" in captured.out

    def test_plain_text_output_contains_span_names(self, capsys) -> None:
        import sys
        import unittest.mock

        exporter = StdoutExporter()
        trace = _make_trace(2)

        with unittest.mock.patch.dict(
            sys.modules, {"rich": None, "rich.console": None, "rich.tree": None}
        ):
            exporter.export(trace)

        captured = capsys.readouterr()
        assert "span-0" in captured.out

    def test_rich_export_parent_child_nesting(self, capsys) -> None:
        """Parent span and child span in the same trace — child must be nested."""
        exporter = StdoutExporter()

        trace = Trace(trace_id="t-nest", run_id="run-nest")
        trace.metadata["name"] = "nested"

        parent = Span(name="parent", span_id="p001", trace_id="t-nest")
        parent.end()
        child = Span(name="child", span_id="c001", trace_id="t-nest", parent_id="p001")
        child.end()

        trace.add_span(parent)
        trace.add_span(child)

        exporter.export(trace)
        captured = capsys.readouterr()
        assert "parent" in captured.out
        assert "child" in captured.out

    def test_rich_export_includes_duration_when_spans_have_times(self, capsys) -> None:
        """When spans have start/end times, the trace header shows total duration."""
        exporter = StdoutExporter()
        trace = _make_trace(2)
        exporter.export(trace)
        captured = capsys.readouterr()
        assert "ms" in captured.out or "total" in captured.out

    def test_export_error_span_does_not_raise(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = Trace(trace_id="t-err", run_id="run-err")
        trace.metadata["name"] = "error-trace"
        span = Span(name="broken", span_id="e001", trace_id="t-err")
        span.end(SpanStatus.ERROR)
        trace.add_span(span)
        exporter.export(trace)

    def test_export_span_prints_exception_message_for_error_status(
        self, capsys
    ) -> None:
        exporter = StdoutExporter()
        span = Span(name="broken")
        span.record_exception(ValueError("boom-details"))
        span.end(SpanStatus.ERROR)
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "boom-details" in captured.out

    def test_export_span_does_not_print_exception_for_ok_status(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="fine")
        span.add_event(
            "exception", {"exception.type": "X", "exception.message": "should-not-show"}
        )
        span.end(SpanStatus.OK)
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "should-not-show" not in captured.out

    def test_export_span_prints_inline_attributes(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="llm:gpt-4")
        span.set_attribute("llm.model", "gpt-4")
        span.set_attribute("llm.usage.prompt_tokens", 100)
        span.set_attribute("llm.usage.completion_tokens", 20)
        span.end()
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "llm.model=gpt-4" in captured.out
        assert "llm.usage.prompt_tokens=100" in captured.out
        assert "llm.usage.completion_tokens=20" in captured.out

    def test_export_span_no_attribute_suffix_when_no_tracked_attrs(
        self, capsys
    ) -> None:
        exporter = StdoutExporter()
        span = Span(name="tool:x")
        span.set_attribute("tool.output", "some output")
        span.end()
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "tool.output" not in captured.out

    def test_rich_export_includes_exception_message_for_error_span(
        self, capsys
    ) -> None:
        exporter = StdoutExporter()
        trace = Trace(trace_id="t-err2", run_id="run-err2")
        trace.metadata["name"] = "error-trace"
        span = Span(name="broken", span_id="e002", trace_id="t-err2")
        span.record_exception(ValueError("rich-boom-details"))
        span.end(SpanStatus.ERROR)
        trace.add_span(span)
        exporter.export(trace)
        captured = capsys.readouterr()
        assert "rich-boom-details" in captured.out

    def test_rich_export_includes_inline_attributes(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = Trace(trace_id="t-attrs", run_id="run-attrs")
        trace.metadata["name"] = "attrs-trace"
        span = Span(name="llm:gpt-4", span_id="a001", trace_id="t-attrs")
        span.set_attribute("llm.model", "gpt-4")
        span.end()
        trace.add_span(span)
        exporter.export(trace)
        captured = capsys.readouterr()
        assert "llm.model=gpt-4" in captured.out

    def test_plain_export_includes_exception_message(self, capsys) -> None:
        import sys
        import unittest.mock

        exporter = StdoutExporter()
        trace = Trace(trace_id="t-plain-err", run_id="run-plain-err")
        trace.metadata["name"] = "plain-error-trace"
        span = Span(name="broken", span_id="pe001", trace_id="t-plain-err")
        span.record_exception(ValueError("plain-boom-details"))
        span.end(SpanStatus.ERROR)
        trace.add_span(span)

        with unittest.mock.patch.dict(
            sys.modules, {"rich": None, "rich.console": None, "rich.tree": None}
        ):
            exporter.export(trace)

        captured = capsys.readouterr()
        assert "plain-boom-details" in captured.out

    def _http_error_exc(self) -> ValueError:
        class _FakeResponse:
            status_code = 400
            text = "Malformed input request (Bedrock validation)"

        class _FakeHTTPError(Exception):
            def __init__(self) -> None:
                super().__init__("400 Client Error: None for url: ...")
                self.response = _FakeResponse()

        return _FakeHTTPError()

    def test_export_span_prints_http_response_body_for_error_status(
        self, capsys
    ) -> None:
        exporter = StdoutExporter()
        span = Span(name="llm:bedrock")
        span.record_exception(self._http_error_exc())
        span.end(SpanStatus.ERROR)
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "HTTP 400" in captured.out
        assert "Malformed input request" in captured.out

    def test_export_span_no_http_detail_line_when_absent(self, capsys) -> None:
        exporter = StdoutExporter()
        span = Span(name="broken")
        span.record_exception(ValueError("plain error, no .response"))
        span.end(SpanStatus.ERROR)
        exporter.export_span(span, depth=0)
        captured = capsys.readouterr()
        assert "HTTP" not in captured.out

    def test_rich_export_includes_http_response_body(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = Trace(trace_id="t-http-err", run_id="run-http-err")
        trace.metadata["name"] = "http-error-trace"
        span = Span(name="llm:bedrock", span_id="he001", trace_id="t-http-err")
        span.record_exception(self._http_error_exc())
        span.end(SpanStatus.ERROR)
        trace.add_span(span)
        exporter.export(trace)
        captured = capsys.readouterr()
        assert "HTTP 400" in captured.out
        assert "Malformed input request" in captured.out

    def test_plain_export_includes_http_response_body(self, capsys) -> None:
        import sys
        import unittest.mock

        exporter = StdoutExporter()
        trace = Trace(trace_id="t-plain-http-err", run_id="run-plain-http-err")
        trace.metadata["name"] = "plain-http-error-trace"
        span = Span(name="llm:bedrock", span_id="phe001", trace_id="t-plain-http-err")
        span.record_exception(self._http_error_exc())
        span.end(SpanStatus.ERROR)
        trace.add_span(span)

        with unittest.mock.patch.dict(
            sys.modules, {"rich": None, "rich.console": None, "rich.tree": None}
        ):
            exporter.export(trace)

        captured = capsys.readouterr()
        assert "HTTP 400" in captured.out
        assert "Malformed input request" in captured.out


# ---------------------------------------------------------------------------
# Computed "unaccounted time" per span (#3515, #2920)
# ---------------------------------------------------------------------------


def _span_with_duration(
    name: str, start: float, end: float | None, span_id: str = "s1"
) -> Span:
    span = Span(name=name, span_id=span_id, start_time=start, end_time=end)
    return span


class TestUnaccountedMs:
    def test_no_children_returns_none(self) -> None:
        from agent_trace.exporters.stdout import _unaccounted_ms

        span = _span_with_duration("node:a", 0.0, 1.0)
        assert _unaccounted_ms(span, None) is None
        assert _unaccounted_ms(span, []) is None

    def test_open_span_returns_none(self) -> None:
        from agent_trace.exporters.stdout import _unaccounted_ms

        span = _span_with_duration("node:a", 0.0, None)
        child = _span_with_duration("llm:x", 0.0, 0.5, span_id="c1")
        assert _unaccounted_ms(span, [child]) is None

    def test_open_child_returns_none(self) -> None:
        from agent_trace.exporters.stdout import _unaccounted_ms

        span = _span_with_duration("node:a", 0.0, 1.0)
        child = _span_with_duration("llm:x", 0.0, None, span_id="c1")
        assert _unaccounted_ms(span, [child]) is None

    def test_computes_gap_between_parent_and_single_child(self) -> None:
        from agent_trace.exporters.stdout import _unaccounted_ms

        # 1050ms total, 40ms in the one child -> 1010ms unaccounted —
        # the exact #2920 "slow node, fast LLM call" pattern.
        span = _span_with_duration("node:respond", 0.0, 1.050)
        child = _span_with_duration("llm:gpt-4", 0.0, 0.040, span_id="c1")
        assert _unaccounted_ms(span, [child]) == pytest.approx(1010.0)

    def test_sums_multiple_children(self) -> None:
        from agent_trace.exporters.stdout import _unaccounted_ms

        span = _span_with_duration("node:a", 0.0, 1.0)
        child1 = _span_with_duration("llm:x", 0.0, 0.2, span_id="c1")
        child2 = _span_with_duration("tool:y", 0.2, 0.5, span_id="c2")
        # child1: 200ms, child2: 300ms -> 500ms in children, 1000-500=500ms
        # unaccounted.
        assert _unaccounted_ms(span, [child1, child2]) == pytest.approx(500.0)


class TestUnaccountedSuffix:
    def test_no_children_empty_suffix(self) -> None:
        from agent_trace.exporters.stdout import _unaccounted_suffix

        span = _span_with_duration("node:a", 0.0, 1.0)
        assert _unaccounted_suffix(span, None) == ""

    def test_with_children_formats_both_numbers(self) -> None:
        from agent_trace.exporters.stdout import _unaccounted_suffix

        span = _span_with_duration("node:respond", 0.0, 1.050)
        child = _span_with_duration("llm:gpt-4", 0.0, 0.040, span_id="c1")
        suffix = _unaccounted_suffix(span, [child])
        assert "1010.0ms unaccounted" in suffix
        assert "1050.0ms" in suffix


class TestExportSpanUnaccountedTime:
    def test_export_span_without_children_arg_shows_no_unaccounted_line(
        self, capsys
    ) -> None:
        exporter = StdoutExporter()
        span = _span_with_duration("node:a", 0.0, 1.0)
        exporter.export_span(span, depth=0)
        assert "unaccounted" not in capsys.readouterr().out

    def test_export_span_with_children_shows_unaccounted_line(self, capsys) -> None:
        exporter = StdoutExporter()
        span = _span_with_duration("node:respond", 0.0, 1.050)
        child = _span_with_duration("llm:gpt-4", 0.0, 0.040, span_id="c1")
        exporter.export_span(span, depth=0, children=[child])
        out = capsys.readouterr().out
        assert "1010.0ms unaccounted of 1050.0ms" in out

    def test_plain_export_shows_unaccounted_time_for_parent_span(self, capsys) -> None:
        import sys
        import unittest.mock

        exporter = StdoutExporter()
        trace = Trace(trace_id="t-unacc", run_id="run-unacc")
        trace.metadata["name"] = "unaccounted-trace"
        parent = _span_with_duration("node:respond", 0.0, 1.050, span_id="p1")
        child = Span(
            name="llm:gpt-4",
            span_id="c1",
            trace_id="t-unacc",
            parent_id="p1",
            start_time=0.0,
            end_time=0.040,
        )
        trace.add_span(parent)
        trace.add_span(child)

        with unittest.mock.patch.dict(
            sys.modules, {"rich": None, "rich.console": None, "rich.tree": None}
        ):
            exporter.export(trace)

        out = capsys.readouterr().out
        assert "1010.0ms unaccounted of 1050.0ms" in out
        # The child (leaf, no children of its own) must NOT show a line.
        child_line = next(line for line in out.splitlines() if "llm:gpt-4" in line)
        assert "unaccounted" not in child_line

    def test_rich_export_shows_unaccounted_time_for_parent_span(self, capsys) -> None:
        exporter = StdoutExporter()
        trace = Trace(trace_id="t-unacc-rich", run_id="run-unacc-rich")
        trace.metadata["name"] = "unaccounted-trace-rich"
        parent = _span_with_duration("node:respond", 0.0, 1.050, span_id="p1")
        child = Span(
            name="llm:gpt-4",
            span_id="c1",
            trace_id="t-unacc-rich",
            parent_id="p1",
            start_time=0.0,
            end_time=0.040,
        )
        trace.add_span(parent)
        trace.add_span(child)

        exporter.export(trace)

        out = capsys.readouterr().out
        assert "1010.0ms unaccounted of 1050.0ms" in out


# ---------------------------------------------------------------------------
# FileExporter
# ---------------------------------------------------------------------------


class TestFileExporter:
    def test_export_creates_json_file(self, tmp_path: Path) -> None:
        trace = _make_trace(2)
        exporter = FileExporter(tmp_path)
        out_path = exporter.export(trace)
        assert out_path.exists()
        assert out_path.suffix == ".json"

    def test_export_json_is_valid(self, tmp_path: Path) -> None:
        trace = _make_trace(2)
        exporter = FileExporter(tmp_path)
        out_path = exporter.export(trace)
        data = json.loads(out_path.read_text())
        assert isinstance(data, dict)

    def test_export_json_contains_all_spans(self, tmp_path: Path) -> None:
        trace = _make_trace(3)
        exporter = FileExporter(tmp_path)
        out_path = exporter.export(trace)
        data = json.loads(out_path.read_text())
        assert len(data["spans"]) == 3
        span_names = [s["name"] for s in data["spans"]]
        for i in range(3):
            assert f"span-{i}" in span_names

    def test_export_json_filename_uses_run_id(self, tmp_path: Path) -> None:
        trace = _make_trace(1)
        exporter = FileExporter(tmp_path)
        out_path = exporter.export(trace)
        assert trace.run_id in out_path.name

    def test_export_returns_path(self, tmp_path: Path) -> None:
        trace = _make_trace(1)
        exporter = FileExporter(tmp_path)
        result = exporter.export(trace)
        assert isinstance(result, Path)
        assert result.exists()

    def test_export_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "subdir" / "traces"
        assert not new_dir.exists()
        trace = _make_trace(1)
        exporter = FileExporter(new_dir)
        exporter.export(trace)
        assert new_dir.exists()

    def test_export_jsonl_one_line_per_span(self, tmp_path: Path) -> None:
        trace = _make_trace(3)
        exporter = FileExporter(tmp_path, format="jsonl")
        out_path = exporter.export(trace)
        assert out_path.suffix == ".jsonl"
        lines = [l for l in out_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_export_jsonl_each_line_is_valid_json(self, tmp_path: Path) -> None:
        trace = _make_trace(2)
        exporter = FileExporter(tmp_path, format="jsonl")
        out_path = exporter.export(trace)
        for line in out_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                assert "span_id" in obj

    def test_export_jsonl_contains_all_spans(self, tmp_path: Path) -> None:
        trace = _make_trace(4)
        exporter = FileExporter(tmp_path, format="jsonl")
        out_path = exporter.export(trace)
        lines = [l for l in out_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 4


# ---------------------------------------------------------------------------
# OTLPExporter — ImportError path
# ---------------------------------------------------------------------------


class TestOTLPExporter:
    def test_raises_import_error_when_opentelemetry_not_installed(self) -> None:
        """OTLPExporter.export() must raise ImportError with install hint
        when opentelemetry is not available."""
        from agent_trace.exporters.otlp import OTLPExporter

        exporter = OTLPExporter()
        trace = _make_trace(1)

        # Mock the import to simulate opentelemetry not being installed
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": None,
                "opentelemetry.exporter": None,
                "opentelemetry.exporter.otlp": None,
                "opentelemetry.exporter.otlp.proto": None,
                "opentelemetry.exporter.otlp.proto.grpc": None,
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None,
                "opentelemetry.sdk": None,
                "opentelemetry.sdk.resources": None,
                "opentelemetry.sdk.trace": None,
                "opentelemetry.sdk.trace.export": None,
            },
        ):
            with pytest.raises(ImportError) as exc_info:
                exporter.export(trace)

        # The error message must contain an install hint
        error_msg = str(exc_info.value)
        assert (
            "opentelemetry" in error_msg.lower() or "pip install" in error_msg.lower()
        )

    def test_otlp_exporter_init_does_not_require_opentelemetry(self) -> None:
        """Creating an OTLPExporter instance must not import opentelemetry."""
        from agent_trace.exporters.otlp import OTLPExporter

        # Should not raise even if opentelemetry is absent
        exporter = OTLPExporter(endpoint="http://localhost:4317")
        assert exporter.endpoint == "http://localhost:4317"

    def test_otlp_exporter_default_endpoint(self) -> None:
        from agent_trace.exporters.otlp import OTLPExporter

        exporter = OTLPExporter()
        assert exporter.endpoint == "http://localhost:4317"


class TestFileExporterPathTraversal:
    def test_file_exporter_rejects_traversal_in_run_id(self, tmp_path: Path) -> None:
        from agent_trace.core.trace import Trace
        from agent_trace.exporters.file import FileExporter

        trace = Trace(trace_id="t", run_id="../../etc/passwd")
        exporter = FileExporter(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            exporter.export(trace)

    def test_file_exporter_accepts_safe_run_id(self, tmp_path: Path) -> None:
        from agent_trace.core.trace import Trace
        from agent_trace.exporters.file import FileExporter

        trace = Trace(trace_id="safe", run_id="safe-run-id")
        exporter = FileExporter(tmp_path)
        out = exporter.export(trace)
        assert out.parent == tmp_path
        assert out.name == "safe-run-id.json"
