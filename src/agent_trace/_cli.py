"""
Command-line interface for agent-trace.

Commands:
    agent-trace replay <run_id>          — replay a recorded run and print span tree
    agent-trace list                     — list all recorded runs in trace_dir
    agent-trace show <run_id>            — show the stored trace.json for a run
    agent-trace inspect <run_id>         — auto-flag anomalies in captured bodies
    agent-trace diff <run_id_a> <b>      — diff two recorded runs' exchanges
    agent-trace run -- <command>         — exec a child process, recording pre-enabled
    agent-trace version                  — print version
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

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
# HTTP error exchange summary — surfaces 4xx/5xx exchanges as flagged errors
# in `agent-trace replay` output instead of leaving them as undifferentiated
# raw rows indistinguishable from a normal 200 (a captured error response
# like Azure's content_filter 400 sits in the fixture with nothing in the
# CLI distinguishing it today).
# ---------------------------------------------------------------------------


def _print_http_error_exchanges(exchanges: list[dict[str, object]]) -> None:
    from agent_trace import _inspect as ins

    flags = ins.flag_4xx_5xx_exchanges(exchanges)
    if not flags:
        return
    print()
    print(f"HTTP error exchanges ({len(flags)}):")
    for flag in flags:
        print(f"  {flag['method']:<6} {flag['url']:<50}  HTTP {flag['status']}")


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
# Exception-text surfacing — pulls the exception.message off any ERROR-status
# span (already captured by Span.record_exception) into a print-ready view,
# for both `agent-trace show --errors-only` and the plain-text summary block
# both `show`/`replay` print after their main output.
# ---------------------------------------------------------------------------


def _span_exception_message(span: dict[str, object]) -> str | None:
    """Return the exception.message text for *span*, or None if it has no
    recorded exception event."""
    events = span.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict) or event.get("name") != "exception":
            continue
        attrs = event.get("attributes") or {}
        if isinstance(attrs, dict) and "exception.message" in attrs:
            return str(attrs["exception.message"])
    return None


def _span_exception_http_detail(span: dict[str, object]) -> str | None:
    """Return "HTTP <status>: <body preview>" for *span*'s recorded
    exception event when it carried an HTTP error response body (see
    Span.record_exception's exception.http_response_body/
    exception.http_status_code attributes, core/span.py) — the actual
    provider/proxy error text (#4940), not just the generic one-line
    str(exc) message which is all `exception.message` alone gives you for
    e.g. requests.exceptions.HTTPError."""
    events = span.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict) or event.get("name") != "exception":
            continue
        attrs = event.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        body = attrs.get("exception.http_response_body")
        if not body:
            continue
        status = attrs.get("exception.http_status_code")
        prefix = f"HTTP {status}: " if status is not None else "HTTP: "
        return f"{prefix}{body}"
    return None


def _error_spans(spans: list[dict[str, object]]) -> list[dict[str, object]]:
    return [s for s in spans if s.get("status") == "ERROR"]


def _print_errors_only(spans: list[dict[str, object]]) -> None:
    """`agent-trace show --errors-only` — filter to ERROR-status spans and
    print each with its captured exception text inline, instead of dumping
    the full trace.json and requiring a manual grep for
    "exception.stacktrace"."""
    errors = _error_spans(spans)
    print(f"Error spans: {len(errors)} of {len(spans)} total")
    if not errors:
        return
    print()
    for span in errors:
        print(f"[ERR] {span.get('name', '?')}")
        message = _span_exception_message(span)
        if message:
            print(f"      {message}")
        http_detail = _span_exception_http_detail(span)
        if http_detail:
            print(f"      {http_detail}")
    _print_error_classification(spans)


# ---------------------------------------------------------------------------
# Retry-storm detection — flags a single node:* span with more than one
# llm:* child span, the shape produced by application code that silently
# re-invokes the LLM in a loop when a response is empty/malformed (#2920):
# an invisible cause of "sometimes fast, sometimes very slow" with no error
# and no user-facing signal.
# ---------------------------------------------------------------------------


def _retry_storm_rows(spans: list[dict[str, object]]) -> list[dict[str, object]]:
    children_by_parent: dict[str | None, list[dict[str, object]]] = {}
    for span in spans:
        parent_id = span.get("parent_id")
        parent_key = parent_id if isinstance(parent_id, str) else None
        children_by_parent.setdefault(parent_key, []).append(span)

    rows: list[dict[str, object]] = []
    for span in spans:
        name = span.get("name")
        if not isinstance(name, str) or not name.startswith("node:"):
            continue
        span_id = span.get("span_id")
        children = (
            children_by_parent.get(span_id, []) if isinstance(span_id, str) else []
        )
        llm_children = [
            c for c in children if str(c.get("name", "")).startswith("llm:")
        ]
        if len(llm_children) > 1:
            rows.append({"node": name, "llm_child_count": len(llm_children)})
    return rows


def _print_retry_storms(spans: list[dict[str, object]]) -> None:
    rows = _retry_storm_rows(spans)
    if not rows:
        return
    print()
    print("Repeated LLM calls under one node span (possible retry storm):")
    for row in rows:
        print(f"  {row['node']:<30}  {row['llm_child_count']} llm:* child spans")
    print(
        "  (A node re-invoking the LLM multiple times per call produces no "
        "error and no user-facing signal, but explains 'sometimes fast, "
        "sometimes very slow' behavior. See issue #2920.)"
    )


# ---------------------------------------------------------------------------
# Orphaned/misattributed span detection — flags a span whose callback-derived
# parent_id looks suspicious: a root span (parent_id is None) that is not the
# trace's earliest root and whose start_time falls inside another span's
# active [start, end) window. This is the exact visible symptom of
# langgraph#3975 (ChatOpenAI calls flattened to the trace root instead of
# nested under their originating node when a compiled graph is piped into a
# raw lambda) — a reconciliation pass against the trace's own chronological
# span ordering (the callback-independent signal LangGraphTracer's
# parent_run_id-registry lookup has no visibility into) surfaces it
# automatically instead of requiring a developer to eyeball trace.json.
# ---------------------------------------------------------------------------


def _numeric_start_time(span: dict[str, object]) -> float | None:
    start = span.get("start_time")
    return float(start) if isinstance(start, (int, float)) else None


def _http_sequence_confirms(
    suspect: dict[str, object],
    likely_parent: dict[str, object],
    exchanges: list[dict[str, object]],
) -> bool | None:
    """Cross-check a *suspect*/*likely_parent* guess — built purely from
    wall-clock span `start_time`/`end_time` — against the independent,
    always-correct total ordering already captured in fixture.db's
    `sequence_num` column (see `_replay/fixture.py`'s module docstring: "a
    monotonically increasing sequence_num" assigned once per HTTP round-trip
    at record time, immune to the callback-registry misses that produce
    misattributed spans in the first place). This is the reconciliation
    pass the "unaccounted-for root span" heuristic above needs: two spans'
    wall-clock timestamps can only be trusted so far, but the HTTP capture
    layer's sequence_num ordering is a second, structurally independent
    signal for the same underlying chronology.

    Returns True when at least one HTTP exchange was recorded during
    *likely_parent*'s activity strictly before *suspect* started (i.e. the
    HTTP layer independently confirms *likely_parent* was already doing
    work before *suspect* began — corroborating the timestamp-based guess),
    False when every such exchange has a `sequence_num` *greater than* an
    exchange recorded during *suspect*'s own window (the HTTP layer
    contradicts the guess), or None when there isn't enough HTTP-layer data
    on either side to judge (note *likely_parent*'s window necessarily
    contains *suspect*'s, since that's how it was chosen — so only the
    portion of *likely_parent*'s window strictly before *suspect* started is
    used, otherwise every exchange inside *suspect*'s own window would also
    trivially count as "inside likely_parent's window" too).
    """

    def _sequence_nums_in(start: float, end_bound: float) -> list[int]:
        seqs: list[int] = []
        for exchange in exchanges:
            recorded_at = exchange.get("recorded_at")
            seq = exchange.get("sequence_num")
            if not isinstance(recorded_at, (int, float)) or not isinstance(
                seq, int
            ):
                continue
            if start <= recorded_at < end_bound:
                seqs.append(seq)
        return seqs

    parent_start = _numeric_start_time(likely_parent)
    suspect_start = _numeric_start_time(suspect)
    if parent_start is None or suspect_start is None:
        return None

    suspect_end = suspect.get("end_time")
    suspect_end_bound = (
        suspect_end if isinstance(suspect_end, (int, float)) else float("inf")
    )

    parent_exclusive_seqs = _sequence_nums_in(parent_start, suspect_start)
    suspect_seqs = _sequence_nums_in(suspect_start, suspect_end_bound)
    if not parent_exclusive_seqs or not suspect_seqs:
        return None
    return max(parent_exclusive_seqs) < min(suspect_seqs)


def _misattributed_span_rows(
    spans: list[dict[str, object]],
    exchanges: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rootlike = [s for s in spans if s.get("parent_id") is None]
    timed_roots = [
        (s, t) for s in rootlike if (t := _numeric_start_time(s)) is not None
    ]
    roots = [s for s, _t in sorted(timed_roots, key=lambda pair: pair[1])]
    if len(roots) <= 1:
        return []

    suspicious_root_ids = {id(s) for s in roots[1:]}
    suspicious_roots = roots[1:]

    # Spans that could plausibly be the "real" parent: anything open (no
    # end_time, or start <= suspect.start < end) at the moment the
    # suspicious root started. Includes the trace's earliest/genuine root —
    # the langgraph#3975 shape is exactly "this call should have nested
    # under the run's actual root/node span but didn't". Excludes other
    # suspicious roots (one flattened call is never the "real" parent of
    # another).
    candidates = [s for s in spans if id(s) not in suspicious_root_ids]

    rows: list[dict[str, object]] = []
    for suspect in suspicious_roots:
        start = _numeric_start_time(suspect)
        if start is None:
            continue
        best: dict[str, object] | None = None
        best_start: float | None = None
        for candidate in candidates:
            if candidate is suspect:
                continue
            c_start = _numeric_start_time(candidate)
            if c_start is None:
                continue
            c_end = candidate.get("end_time")
            if isinstance(c_end, (int, float)):
                still_open = c_start <= start < c_end
            else:
                # No end_time (still open) or a malformed non-numeric value —
                # treat both as "open" rather than excluding the candidate.
                still_open = c_start <= start
            if still_open and (best_start is None or c_start > best_start):
                best = candidate
                best_start = c_start
        if best is not None:
            row: dict[str, object] = {
                "span": suspect.get("name"),
                "likely_parent": best.get("name"),
            }
            if exchanges is not None:
                row["http_sequence_confirmed"] = _http_sequence_confirms(
                    suspect, best, exchanges
                )
            rows.append(row)
    return rows


def _print_misattributed_spans(
    spans: list[dict[str, object]],
    exchanges: list[dict[str, object]] | None = None,
) -> None:
    rows = _misattributed_span_rows(spans, exchanges)
    if not rows:
        return
    print()
    print(
        "Possibly misattributed spans (unexpected root, chronologically "
        "overlaps another span):"
    )
    for row in rows:
        suffix = ""
        confirmed = row.get("http_sequence_confirmed")
        if confirmed is True:
            suffix = "  [confirmed via HTTP sequence_num ordering]"
        elif confirmed is False:
            suffix = "  [HTTP sequence_num ordering does NOT confirm this guess]"
        print(
            f"  {row['span']:<30}  likely belongs under "
            f"{row['likely_parent']}{suffix}"
        )
    print(
        "  (A span with no parent that started while another span was still "
        "open may have been flattened to the trace root instead of nested "
        "under its originating node — the shape behind langgraph#3975. "
        "Reconciled against fixture.db's sequence_num-ordered HTTP capture "
        "where available, per the timestamp-independent cross-check above.)"
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

    spans = data.get("spans", [])

    if getattr(args, "errors_only", False):
        _print_errors_only(spans)
        return

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
    print(f"Spans: {len(spans)}")

    # Load fixture.db's exchanges (if this run was recorded with
    # record=True) purely so _print_misattributed_spans can reconcile its
    # wall-clock-timestamp guess against the independent, always-correct
    # sequence_num ordering the HTTP capture layer already has.
    exchanges: list[dict[str, object]] | None = None
    fixture_db = _fixture_path(run_id)
    if fixture_db.exists():
        try:
            from agent_trace._replay.fixture import Fixture

            with Fixture(fixture_db) as f:
                exchanges = f.all_exchanges()
        except Exception:
            exchanges = None

    _print_error_classification(spans)
    _print_duplicate_node_spans(spans)
    _print_retry_storms(spans)
    _print_misattributed_spans(spans, exchanges)
    _print_checkpoint_durability(spans)
    _print_zero_task_updates(spans)


# ---------------------------------------------------------------------------
# Subcommand: replay
# ---------------------------------------------------------------------------


def _print_replay_span_diagnostics(
    spans: list[dict[str, object]],
    exchanges: list[dict[str, object]] | None = None,
) -> None:
    """The full diagnostic block `agent-trace replay` prints after the span
    tree — split out from cmd_replay() purely to keep that function's
    statement count manageable, not for reuse elsewhere."""
    _print_error_classification(spans)
    _print_duplicate_node_spans(spans)
    _print_retry_storms(spans)
    _print_misattributed_spans(spans, exchanges)
    _print_checkpoint_durability(spans)
    _print_zero_task_updates(spans)


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
    _print_http_error_exchanges(all_exchanges)
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
            _print_replay_span_diagnostics(trace_data.get("spans", []), all_exchanges)
        except Exception as exc:
            print(f"Could not render span tree: {exc}")
    else:
        print(f"(No trace.json — run 'agent-trace show {run_id}' after recording)")


# ---------------------------------------------------------------------------
# Subcommand: inspect
# ---------------------------------------------------------------------------
#
# The one-stop diagnosis command: decodes fixture.db request/response bodies
# and runs every pattern check in agent_trace._inspect against them,
# printing a flagged summary instead of requiring a developer to hand-write
# a comparison script against Fixture.all_exchanges() (the exact manual step
# this command replaces — see #531's backlog entry for the full mapping of
# each check to the issue it closes).
# ---------------------------------------------------------------------------


def _load_run_exchanges_and_spans(
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (exchanges, spans) for *run_id* — empty lists for whichever
    file (fixture.db / trace.json) doesn't exist rather than raising, since
    a run may have been recorded with record=False (spans only) or may not
    have a trace.json yet."""
    from agent_trace._replay.fixture import Fixture

    exchanges: list[dict[str, Any]] = []
    fixture_path = _fixture_path(run_id)
    if fixture_path.exists():
        with Fixture(fixture_path) as f:
            exchanges = f.all_exchanges()

    spans: list[dict[str, Any]] = []
    trace_path = _trace_json_path(run_id)
    if trace_path.exists():
        try:
            data = json.loads(trace_path.read_text(encoding="utf-8"))
            spans = data.get("spans", [])
        except json.JSONDecodeError:
            pass

    return exchanges, spans


def _print_flags(title: str, flags: list[dict[str, Any]]) -> None:
    if not flags:
        return
    print()
    print(f"{title} ({len(flags)}):")
    for flag in flags:
        print(f"  - {flag.get('detail', flag)}")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Auto-flag/search raw request-response bodies for known malformed
    shapes, plus a set of cross-span diagnostics — the CLI command
    #531's backlog entry asked for so a developer doesn't have to
    hand-write a comparison script against Fixture.all_exchanges()."""
    from agent_trace import _inspect as ins

    run_id: str = args.run_id
    _require_run_dir(run_id)
    exchanges, spans = _load_run_exchanges_and_spans(run_id)

    print(f"Inspecting run: {run_id}")
    print(f"Exchanges: {len(exchanges)}  Spans: {len(spans)}")

    results = ins.run_all_exchange_checks(exchanges)
    for check_name, flags in results.items():
        _print_flags(check_name, flags)

    _print_flags(
        "known_error_pattern",
        ins.match_known_error_patterns(spans),
    )
    _print_flags(
        "reserved_kwarg_collision",
        ins.check_reserved_kwarg_collision(spans),
    )
    _print_flags(
        "near_duplicate_sibling_content",
        ins.find_near_duplicate_sibling_content(spans),
    )

    if args.registered_tools:
        registered = set(args.registered_tools.split(","))
        _print_flags(
            "tool_call_name_fuzzy_match",
            ins.check_tool_call_name_fuzzy_match(exchanges, registered),
        )
        _print_flags(
            "tool_call_name_dotted_compound",
            ins.check_tool_call_name_dotted_compound(exchanges, registered),
        )
        _print_flags(
            "action_name_not_registered",
            ins.check_action_name_not_registered(exchanges, registered),
        )

    if args.configured_host:
        _print_flags(
            "endpoint_host_mismatch",
            ins.check_endpoint_host_mismatch(exchanges, args.configured_host),
        )

    if args.check_kwarg:
        _print_flags(
            "missing_extra_kwarg",
            ins.check_missing_extra_kwarg(exchanges, args.check_kwarg),
        )

    if args.diff_field:
        _print_flags(
            "field_present_on_wire_absent_downstream",
            ins.field_present_on_wire_absent_downstream(
                exchanges, spans, args.diff_field
            ),
        )

    if args.diff_get_post_field:
        _print_flags(
            "get_post_field_mismatch",
            ins.check_get_post_field_mismatch(
                exchanges,
                args.diff_get_post_field,
                get_id_field=args.diff_get_post_id_field,
                post_id_field=args.diff_get_post_post_id_field,
            ),
        )

    if not results and not spans:
        print()
        print("No anomalies flagged (or nothing recorded for this run).")


# ---------------------------------------------------------------------------
# Subcommand: diff
# ---------------------------------------------------------------------------
#
# `agent-trace diff <run_id_a> <run_id_b>` — the CLI never previously
# exposed raw exchange bodies at all, let alone a comparison between two
# runs. cmd_show only pretty-prints trace.json span metadata; cmd_replay
# only prints exchange counts and the span tree.
# ---------------------------------------------------------------------------


def _exchanges_by_url(
    exchanges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_url: dict[str, list[dict[str, Any]]] = {}
    for exchange in exchanges:
        by_url.setdefault(str(exchange.get("url")), []).append(exchange)
    return by_url


def _diff_text(label_a: str, text_a: str, label_b: str, text_b: str) -> list[str]:
    return list(
        difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
            lineterm="",
        )
    )


def cmd_diff(args: argparse.Namespace) -> None:
    """Print a structured diff of two recorded runs' exchanges, matched by
    URL, highlighting field-level differences between request/response
    bodies."""
    run_id_a: str = args.run_id_a
    run_id_b: str = args.run_id_b
    _require_run_dir(run_id_a)
    _require_run_dir(run_id_b)

    exchanges_a, _ = _load_run_exchanges_and_spans(run_id_a)
    exchanges_b, _ = _load_run_exchanges_and_spans(run_id_b)

    by_url_a = _exchanges_by_url(exchanges_a)
    by_url_b = _exchanges_by_url(exchanges_b)

    print(f"Diffing {run_id_a} ({len(exchanges_a)} exchanges) vs "
          f"{run_id_b} ({len(exchanges_b)} exchanges)")

    all_urls = sorted(set(by_url_a) | set(by_url_b))
    if not all_urls:
        print("No exchanges recorded in either run.")
        return

    any_diff = False
    for url in all_urls:
        rows_a = by_url_a.get(url, [])
        rows_b = by_url_b.get(url, [])
        if not rows_a:
            print(f"\n{url}: only present in {run_id_b} ({len(rows_b)} exchange(s))")
            any_diff = True
            continue
        if not rows_b:
            print(f"\n{url}: only present in {run_id_a} ({len(rows_a)} exchange(s))")
            any_diff = True
            continue

        for i in range(max(len(rows_a), len(rows_b))):
            row_a = rows_a[i] if i < len(rows_a) else None
            row_b = rows_b[i] if i < len(rows_b) else None
            if row_a is None or row_b is None:
                print(f"\n{url} [{i}]: exchange count differs between runs")
                any_diff = True
                continue
            for field in ("request_body", "response_body"):
                text_a = str(row_a.get(field, ""))
                text_b = str(row_b.get(field, ""))
                if text_a == text_b:
                    continue
                any_diff = True
                print(f"\n{url} [{i}] {field}:")
                for line in _diff_text(f"{run_id_a}", text_a, f"{run_id_b}", text_b):
                    print(f"  {line}")

    if not any_diff:
        print("\nNo differences found — every matched exchange is byte-identical.")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------
#
# Wraps an arbitrary child process (e.g. `langgraph dev`) with recording
# pre-enabled process-wide, via the AGENT_TRACE_AUTO_RECORD env var
# (Tracer.start_auto_record() / agent_trace._activate_auto_record_from_env())
# — the supported mechanism for capturing a framework-managed dev server
# process whose entrypoint the developer doesn't own, and doesn't want to
# hand-instrument with a `with tracer.start_trace(...)` block.
# ---------------------------------------------------------------------------


def _strip_leading_separator(command: list[str]) -> list[str]:
    """`agent-trace run -- langgraph dev` — argparse.REMAINDER captures the
    `--` separator verbatim as the first token of the remainder. Strip it
    (when present) so the child process is exec'd with the real command
    only. A bare `agent-trace run langgraph dev` (no `--`) works too — there
    is simply nothing to strip in that case."""
    if command and command[0] == "--":
        return command[1:]
    return command


def cmd_run(args: argparse.Namespace) -> None:
    """Exec a child process with recording pre-enabled process-wide.

    Sets ``AGENT_TRACE_AUTO_RECORD=1`` (plus ``AGENT_TRACE_RUN_ID``/
    ``AGENT_TRACE_AUTO_RECORD_NAME``/``AGENT_TRACE_TRACE_DIR``) in the
    child's environment before launching it, so the *first* `import
    agent_trace` anywhere inside that process — including one owned
    entirely by a third-party CLI like `langgraph dev` that the developer
    never touches — activates process-wide recording with zero code
    changes required in the developer's own graph/agent code. See
    ``agent_trace.Tracer.start_auto_record`` and
    ``agent_trace._activate_auto_record_from_env`` for the activation path
    this env var feeds into.

    Exits with the child process's own exit code.
    """
    command = _strip_leading_separator(args.child_command)
    if not command:
        sys.exit(
            "error: no command given.\n"
            "Usage: agent-trace run -- <command> [args...]\n"
            "Example: agent-trace run -- langgraph dev"
        )

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:12]}"
    trace_dir = _trace_dir()

    env = os.environ.copy()
    env["AGENT_TRACE_AUTO_RECORD"] = "1"
    env["AGENT_TRACE_RUN_ID"] = run_id
    env["AGENT_TRACE_AUTO_RECORD_NAME"] = args.name
    env["AGENT_TRACE_TRACE_DIR"] = str(trace_dir)

    print("agent-trace: recording enabled (AGENT_TRACE_AUTO_RECORD=1)")
    print(f"agent-trace: run_id:    {run_id}")
    print(f"agent-trace: trace_dir: {trace_dir}")
    print(f"agent-trace: command:   {' '.join(command)}")
    print()

    try:
        result = subprocess.run(command, env=env, check=False)  # noqa: S603
    except FileNotFoundError as exc:
        sys.exit(f"error: could not exec {command[0]!r}: {exc}")
    except KeyboardInterrupt:
        sys.exit(130)

    print()
    print(f"agent-trace: child process exited with code {result.returncode}")
    print(f"agent-trace: inspect this run with: agent-trace show {run_id}")
    sys.exit(result.returncode)


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
            "  agent-trace show run_abc123def456 --errors-only\n"
            "  agent-trace replay run_abc123def456\n"
            "  agent-trace inspect run_abc123def456\n"
            "  agent-trace diff run_a run_b\n"
            "  agent-trace run -- langgraph dev\n"
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
    show_p.add_argument(
        "--errors-only",
        dest="errors_only",
        action="store_true",
        help="Only print ERROR-status spans with their captured exception text",
    )

    # replay
    replay_p = sub.add_parser(
        "replay",
        help="Enter replay mode for a run and print the span tree",
    )
    replay_p.add_argument("run_id", help="Run ID (e.g. run_abc123def456)")

    # inspect
    inspect_p = sub.add_parser(
        "inspect",
        help="Auto-flag/search raw request-response bodies for known malformed shapes",
    )
    inspect_p.add_argument("run_id", help="Run ID (e.g. run_abc123def456)")
    inspect_p.add_argument(
        "--registered-tools",
        dest="registered_tools",
        default=None,
        help="Comma-separated list of registered tool names, enables tool-call "
        "name fuzzy-match/dotted-compound/ReAct-action-name checks",
    )
    inspect_p.add_argument(
        "--configured-host",
        dest="configured_host",
        default=None,
        help="Framework's configured LLM endpoint host, enables the "
        "endpoint-host-mismatch check",
    )
    inspect_p.add_argument(
        "--check-kwarg",
        dest="check_kwarg",
        default=None,
        help="Dotted kwarg path (e.g. extra_body.chat_template_kwargs.thinking) "
        "expected to be present on the wire; flags requests where it's absent",
    )
    inspect_p.add_argument(
        "--diff-field",
        dest="diff_field",
        default=None,
        help="Top-level response field to check for wire-present-but-"
        "downstream-absent (e.g. usage)",
    )
    inspect_p.add_argument(
        "--diff-get-post-field",
        dest="diff_get_post_field",
        default=None,
        help="Dotted field path (e.g. instructions) to compare between an "
        "earlier GET response and a later, causally-related POST request "
        "body referencing the same resource id (#2620); flags stale-value "
        "mismatches such as a GPTAssistantAgent POST /runs still sending "
        "instructions that no longer match what GET /assistants/{id} "
        "returns",
    )
    inspect_p.add_argument(
        "--diff-get-post-id-field",
        dest="diff_get_post_id_field",
        default="id",
        help="Field name the GET response uses for the resource id "
        "(default: id)",
    )
    inspect_p.add_argument(
        "--diff-get-post-post-id-field",
        dest="diff_get_post_post_id_field",
        default=None,
        help="Field name the POST request body uses to reference the same "
        "resource id, if different from --diff-get-post-id-field (e.g. "
        "assistant_id)",
    )

    # diff
    diff_p = sub.add_parser(
        "diff",
        help="Diff two recorded runs' exchanges (matched by URL)",
    )
    diff_p.add_argument("run_id_a", help="First run ID")
    diff_p.add_argument("run_id_b", help="Second run ID")

    # run
    run_p = sub.add_parser(
        "run",
        help=(
            "Exec a child process with recording pre-enabled process-wide "
            "(e.g. agent-trace run -- langgraph dev)"
        ),
    )
    run_p.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Explicit run ID (default: random, printed on start)",
    )
    run_p.add_argument(
        "--name",
        dest="name",
        default="auto-record",
        help="Trace name recorded in trace.json metadata (default: auto-record)",
    )
    run_p.add_argument(
        "child_command",
        nargs=argparse.REMAINDER,
        help="Command to exec, e.g. -- langgraph dev",
    )

    args = parser.parse_args()

    dispatch = {
        "version": cmd_version,
        "list": cmd_list,
        "show": cmd_show,
        "replay": cmd_replay,
        "inspect": cmd_inspect,
        "diff": cmd_diff,
        "run": cmd_run,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
