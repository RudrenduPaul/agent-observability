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
    except Exception:
        exchange_count = 0

    print(f"Recorded exchanges: {exchange_count}")
    print()

    # Enter replay mode
    from agent_trace import replay as _replay

    with _replay(run_id) as ctx:
        print(f"Replay active — {ctx.fixture.exchange_count()} exchange(s) available")
        print("(No HTTP requests were made — fixture is ready for agent code)")
        print()

    # Load and print the original trace
    if trace_json_path.exists():
        try:
            from agent_trace.core.trace import Trace
            from agent_trace.exporters.stdout import StdoutExporter

            trace_obj = Trace.from_dict(
                json.loads(trace_json_path.read_text(encoding="utf-8"))
            )
            print("--- Original span tree (from trace.json) ---")
            StdoutExporter().export(trace_obj)
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
