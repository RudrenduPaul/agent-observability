"""
Command-line interface for agent-trace.

Commands:
    agent-trace replay <run_id>     — replay a recorded run and print span tree
    agent-trace list                — list all recorded runs in trace_dir
    agent-trace show <run_id>       — show the stored trace.json for a run
    agent-trace version             — print version
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

__all__ = ["main"]

_VERSION = "0.1.0"
_DEFAULT_TRACE_DIR = Path.home() / ".agent-trace" / "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_dir() -> Path:
    """Return the active trace directory, honouring the env override."""
    env_dir = os.environ.get("AGENT_TRACE_TRACE_DIR")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_TRACE_DIR


def _run_dir(run_id: str) -> Path:
    base = _trace_dir().resolve()
    candidate = (base / run_id).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Invalid run_id {run_id!r}: path traversal detected"
        ) from None
    return candidate


def _fixture_path(run_id: str) -> Path:
    return _run_dir(run_id) / "fixture.db"


def _trace_json_path(run_id: str) -> Path:
    return _run_dir(run_id) / "trace.json"


def _require_run_dir(run_id: str) -> Path:
    """Return the run directory or exit with a clear message."""
    d = _run_dir(run_id)
    if not d.exists():
        sys.exit(
            f"error: no run directory found for {run_id!r}\n"
            f"Expected: {d}\n"
            "Run 'agent-trace list' to see all recorded runs."
        )
    return d


# ---------------------------------------------------------------------------
# Subcommand: version
# ---------------------------------------------------------------------------


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"agent-trace {_VERSION}")


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def cmd_list(_args: argparse.Namespace) -> None:
    """List all recorded runs in the trace directory."""
    trace_dir = _trace_dir()
    if not trace_dir.exists():
        print(f"No runs yet. Trace directory does not exist: {trace_dir}")
        return

    run_dirs = sorted(
        (d for d in trace_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not run_dirs:
        print(f"No recorded runs in {trace_dir}")
        return

    # Header
    print(f"{'RUN ID':<30}  {'EXCHANGES':>9}  {'SPANS':>6}  RECORDED AT")
    print("-" * 70)

    for run_dir in run_dirs:
        run_id = run_dir.name
        exchanges = _count_exchanges(run_dir)
        spans = _count_spans(run_dir)
        mtime = _format_mtime(run_dir.stat().st_mtime)
        print(f"{run_id:<30}  {exchanges:>9}  {spans:>6}  {mtime}")

    print()
    print(f"Total: {len(run_dirs)} run(s) in {trace_dir}")


def _count_exchanges(run_dir: Path) -> str:
    fixture_db = run_dir / "fixture.db"
    if not fixture_db.exists():
        return "-"
    try:
        from agent_trace._replay.fixture import Fixture

        with Fixture(fixture_db) as f:
            return str(f.exchange_count())
    except Exception:
        return "?"


def _count_spans(run_dir: Path) -> str:
    trace_json = run_dir / "trace.json"
    if not trace_json.exists():
        return "-"
    try:
        data = json.loads(trace_json.read_text(encoding="utf-8"))
        return str(len(data.get("spans", [])))
    except Exception:
        return "?"


def _format_mtime(ts: float) -> str:
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Error classification summary — surfaces error.origin/error.known_pattern
# (set by the LangGraph integration's exception classifier) as a one-line
# pointer per ERROR span, instead of requiring a developer to read the raw
# trace.json blob and find those attributes by eye.
# ---------------------------------------------------------------------------


def _error_classification_rows(spans: list[dict[str, object]]) -> list[dict[str, str]]:
    """Return one row per ERROR-status span with its origin/known-pattern
    classification (empty string for a field that wasn't set)."""
    rows: list[dict[str, str]] = []
    for span in spans:
        if span.get("status") != "ERROR":
            continue
        attrs = span.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        rows.append(
            {
                "name": str(span.get("name", "?")),
                "origin": str(attrs.get("error.origin", "")),
                "known_pattern": str(attrs.get("error.known_pattern", "")),
            }
        )
    return rows


def _print_error_classification(spans: list[dict[str, object]]) -> None:
    rows = _error_classification_rows(spans)
    if not rows:
        return
    print()
    print("Error classification:")
    for row in rows:
        origin = row["origin"] or "unclassified"
        line = f"  {row['name']:<30}  origin={origin}"
        if row["known_pattern"]:
            line += f"  known_pattern={row['known_pattern']}"
        print(line)


# ---------------------------------------------------------------------------
# Duplicate node span detection — flags a node:<name> span that appears more
# than once within a single trace, the shape produced when a resumed
# graph.stream()/.invoke() call re-executes a task that should have reused a
# cached checkpointed write (see issue #6050). Heuristic, not a definitive
# bug detector: an intentionally looping node (e.g. a ReAct agent) also
# produces repeated node:<name> spans by design — this surfaces the count as
# a fact for a developer to interpret, rather than claiming a verdict.
# ---------------------------------------------------------------------------


def _duplicate_node_span_counts(spans: list[dict[str, object]]) -> dict[str, int]:
    """Return {node span name: occurrence count} for every node:<name> span
    name that appears more than once in *spans*."""
    counts: dict[str, int] = {}
    for span in spans:
        name = span.get("name")
        if isinstance(name, str) and name.startswith("node:"):
            counts[name] = counts.get(name, 0) + 1
    return {name: n for name, n in counts.items() if n > 1}


def _print_duplicate_node_spans(spans: list[dict[str, object]]) -> None:
    duplicates = _duplicate_node_span_counts(spans)
    if not duplicates:
        return
    print()
    print("Duplicate node spans (same node:<name> executed more than once):")
    for name, count in sorted(duplicates.items()):
        print(f"  {name:<30}  executed {count} times")
    print(
        "  (Expected for an intentionally looping node — e.g. a ReAct agent. "
        "Unexpected for a task that should have reused a cached checkpointed "
        "write across a resume — see issue #6050.)"
    )


# ---------------------------------------------------------------------------
# Streaming timing diagnostic — surfaces time-to-first-chunk and max
# inter-chunk gap for any exchange recorded via a pass-through streaming
# transport (RecordingTransport(..., stream=True) /
# AsyncRecordingTransport(..., stream=True)). Exchanges recorded the default
# (eager-buffering) way carry no chunk_timestamps and are silently skipped —
# absence means "not captured this way", not "instant delivery".
# ---------------------------------------------------------------------------


def _streaming_timing_rows(
    exchanges: list[dict[str, object]],
) -> list[dict[str, object]]:
    """One row per exchange with recorded per-chunk timestamps."""
    from agent_trace._replay.fixture import (
        max_inter_chunk_gap_ms,
        time_to_first_chunk_ms,
    )

    rows: list[dict[str, object]] = []
    for exchange in exchanges:
        if not exchange.get("chunk_timestamps"):
            continue
        rows.append(
            {
                "url": exchange.get("url", "?"),
                "method": exchange.get("method", "?"),
                "chunk_count": len(exchange["chunk_timestamps"]),  # type: ignore[arg-type]
                "time_to_first_chunk_ms": time_to_first_chunk_ms(exchange),
                "max_inter_chunk_gap_ms": max_inter_chunk_gap_ms(exchange),
            }
        )
    return rows


def _print_streaming_timing(exchanges: list[dict[str, object]]) -> None:
    rows = _streaming_timing_rows(exchanges)
    if not rows:
        return
    print()
    print("Streaming timing (time-to-first-chunk / max inter-chunk gap):")
    for row in rows:
        ttfc = row["time_to_first_chunk_ms"]
        gap = row["max_inter_chunk_gap_ms"]
        ttfc_str = f"{ttfc:.1f}ms" if isinstance(ttfc, (int, float)) else "?"
        gap_str = f"{gap:.1f}ms" if isinstance(gap, (int, float)) else "?"
        print(
            f"  {row['method']:<6} {row['url']:<50}  "
            f"chunks={row['chunk_count']:>4}  "
            f"first_chunk={ttfc_str:>10}  max_gap={gap_str:>10}"
        )


# ---------------------------------------------------------------------------
# Checkpoint durability diagnostic — correlates checkpoint:put/put_writes
# spans (recorded by TracingCheckpointSaver,
# agent_trace.integrations.langgraph_checkpoint) against the rest of the
# trace, surfacing a terminal durable|partial|abandoned status per the shape
# independently proposed in the #5672 thread (cancellation_requested,
# writes_enqueued_count, writes_flushed_count, checkpoint_status) instead of
# requiring a developer to manually diff two span sets by hand to answer
# "did what the user saw actually get durably persisted before the run
# ended".
# ---------------------------------------------------------------------------

_CHECKPOINT_WRITE_SPAN_NAMES = frozenset(
    {
        "checkpoint:put",
        "checkpoint:aput",
        "checkpoint:put_writes",
        "checkpoint:aput_writes",
    }
)


def _checkpoint_durability_summary(
    spans: list[dict[str, object]],
) -> dict[str, object] | None:
    """Return the durability summary for *spans*, or None if no
    checkpoint-write spans were recorded in this trace at all (the
    TracingCheckpointSaver wrapper wasn't wired in — nothing to report,
    distinct from "wired in but zero writes flushed")."""
    write_spans = [s for s in spans if s.get("name") in _CHECKPOINT_WRITE_SPAN_NAMES]
    if not write_spans:
        return None

    enqueued = len(write_spans)
    flushed = 0
    for span in write_spans:
        attrs = span.get("attributes") or {}
        if isinstance(attrs, dict) and attrs.get("checkpoint.completed") is True:
            flushed += 1

    cancellation_requested = any(s.get("status") == "CANCELLED" for s in spans)

    if flushed == enqueued and not cancellation_requested:
        status = "durable"
    elif flushed == 0:
        status = "abandoned"
    else:
        status = "partial"

    return {
        "cancellation_requested": cancellation_requested,
        "writes_enqueued_count": enqueued,
        "writes_flushed_count": flushed,
        "checkpoint_status": status,
    }


def _print_checkpoint_durability(spans: list[dict[str, object]]) -> None:
    summary = _checkpoint_durability_summary(spans)
    if summary is None:
        return
    print()
    print("Checkpoint durability:")
    print(f"  checkpoint_status:        {summary['checkpoint_status']}")
    print(f"  writes_enqueued_count:    {summary['writes_enqueued_count']}")
    print(f"  writes_flushed_count:     {summary['writes_flushed_count']}")
    print(f"  cancellation_requested:   {summary['cancellation_requested']}")
    if summary["checkpoint_status"] != "durable":
        print(
            "  (Not every checkpoint write span closed checkpoint.completed=True "
            "before the run ended — state observed during the run may not "
            "match what was actually persisted. See issue #5672.)"
        )


# ---------------------------------------------------------------------------
# Zero-tasks-scheduled diagnostic — flags a checkpoint:update_state span
# (recorded by traced_update_state()/traced_aupdate_state(),
# agent_trace.integrations.langgraph_checkpoint) whose post-write task
# schedule came back empty, the exact silent-no-op-resume shape behind issue
# #4217 (a missing/incorrect as_node on an external state write).
# ---------------------------------------------------------------------------


def _zero_task_update_rows(spans: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for span in spans:
        if span.get("name") != "checkpoint:update_state":
            continue
        attrs = span.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        if attrs.get("checkpoint.zero_tasks_scheduled") is True:
            rows.append(
                {
                    "as_node": attrs.get("checkpoint.as_node", "<not provided>"),
                    "as_node_provided": attrs.get("checkpoint.as_node_provided", False),
                }
            )
    return rows


def _print_zero_task_updates(spans: list[dict[str, object]]) -> None:
    rows = _zero_task_update_rows(spans)
    if not rows:
        return
    print()
    print("Zero tasks scheduled after a state update (likely misattributed write):")
    for row in rows:
        provided = "yes" if row["as_node_provided"] else "no"
        print(f"  as_node={row['as_node']}  (explicitly provided: {provided})")
    print(
        "  (An update_state() call was followed by an empty scheduled-task "
        "list — the graph will not advance from this state. Usually a "
        "missing/incorrect as_node. See issue #4217.)"
    )


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------


def cmd_show(args: argparse.Namespace) -> None:
    """Print the trace.json for a run, pretty-printed."""
    run_id: str = args.run_id
    _require_run_dir(run_id)

    trace_path = _trace_json_path(run_id)
    if not trace_path.exists():
        sys.exit(
            f"error: no trace.json found for run {run_id!r}\n"
            f"Expected: {trace_path}\n"
            "The run directory exists but no trace was written. "
            "Did the trace context exit normally?"
        )

    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"error: trace.json is not valid JSON: {exc}")

    # Try rich for colored output; fall back to plain json.dumps
    try:
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()
        syntax = Syntax(
            json.dumps(data, indent=2),
            "json",
            theme="monokai",
            line_numbers=False,
        )
        console.print(syntax)
    except ImportError:
        print(json.dumps(data, indent=2))

    print()
    print(f"File: {trace_path}")
    spans = data.get("spans", [])
    print(f"Spans: {len(spans)}")
    _print_error_classification(spans)
    _print_duplicate_node_spans(spans)
    _print_checkpoint_durability(spans)
    _print_zero_task_updates(spans)


# ---------------------------------------------------------------------------
# Subcommand: replay
# ---------------------------------------------------------------------------


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay a recorded run and print the resulting span tree."""
    run_id: str = args.run_id
    _require_run_dir(run_id)

    fixture = _fixture_path(run_id)
    if not fixture.exists():
        sys.exit(
            f"error: no fixture.db found for run {run_id!r}\n"
            f"Expected: {fixture}\n"
            "Did you record this run with record=True?\n"
            "Without record=True only trace.json is written, not fixture.db."
        )

    trace_json_path = _trace_json_path(run_id)

    print(f"Replaying run: {run_id}")
    print(f"Fixture:       {fixture}")

    # Load the original trace to find span names
    original_spans: list[dict[str, object]] = []
    if trace_json_path.exists():
        try:
            data = json.loads(trace_json_path.read_text(encoding="utf-8"))
            original_spans = data.get("spans", [])
        except Exception:  # noqa: S110
            pass

    print(f"Original span count: {len(original_spans)}")

    # Count exchanges via Fixture (honours WAL mode + schema)
    from agent_trace._replay.fixture import Fixture

    try:
        with Fixture(fixture) as f:
            exchange_count = f.exchange_count()
            all_exchanges = f.all_exchanges()
    except Exception:
        exchange_count = 0
        all_exchanges = []

    print(f"Recorded exchanges: {exchange_count}")
    _print_streaming_timing(all_exchanges)
    print()

    # Enter replay mode
    from agent_trace import replay as _replay

    with _replay(run_id, trace_dir=_trace_dir()) as ctx:
        print(f"Replay active — {ctx.fixture.exchange_count()} exchange(s) available")
        print("(No HTTP requests were made — fixture is ready for agent code)")
        print()

    # Load and print the original trace
    if trace_json_path.exists():
        try:
            from agent_trace.core.trace import Trace
            from agent_trace.exporters.stdout import StdoutExporter

            trace_data = json.loads(trace_json_path.read_text(encoding="utf-8"))
            trace_obj = Trace.from_dict(trace_data)
            print("--- Original span tree (from trace.json) ---")
            StdoutExporter().export(trace_obj)
            _print_error_classification(trace_data.get("spans", []))
            _print_duplicate_node_spans(trace_data.get("spans", []))
            _print_checkpoint_durability(trace_data.get("spans", []))
            _print_zero_task_updates(trace_data.get("spans", []))
        except Exception as exc:
            print(f"Could not render span tree: {exc}")
    else:
        print(f"(No trace.json — run 'agent-trace show {run_id}' after recording)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-trace",
        description="agent-trace — AI agent observability with record/replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  agent-trace list\n"
            "  agent-trace show run_abc123def456\n"
            "  agent-trace replay run_abc123def456\n"
            "  agent-trace version\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # version
    sub.add_parser("version", help="Print version and exit")

    # list
    sub.add_parser("list", help="List all recorded runs")

    # show
    show_p = sub.add_parser("show", help="Pretty-print trace.json for a run")
    show_p.add_argument("run_id", help="Run ID (e.g. run_abc123def456)")

    # replay
    replay_p = sub.add_parser(
        "replay",
        help="Enter replay mode for a run and print the span tree",
    )
    replay_p.add_argument("run_id", help="Run ID (e.g. run_abc123def456)")

    args = parser.parse_args()

    dispatch = {
        "version": cmd_version,
        "list": cmd_list,
        "show": cmd_show,
        "replay": cmd_replay,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
