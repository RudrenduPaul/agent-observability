"""
File exporter — writes traces as JSON or JSONL files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent_trace import Trace

__all__ = [
    "FileExporter",
    "export",
]


class FileExporter:
    """Write a :class:`~agent_trace.Trace` to disk as JSON or JSONL.

    Parameters
    ----------
    output_dir:
        Directory where trace files are written.  Created if it does not exist.
    format:
        ``"json"`` writes a single pretty-printed JSON file with the full
        trace structure.  ``"jsonl"`` writes one span dict per line.
    """

    def __init__(
        self,
        output_dir: Path,
        format: Literal["json", "jsonl"] = "json",
    ) -> None:
        self.output_dir: Path = Path(output_dir)
        self.format: Literal["json", "jsonl"] = format

    def export(self, trace: Trace) -> Path:
        """Write *trace* to ``output_dir/{run_id}.{format}`` and return the path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir.resolve()
        out_path = (base / f"{trace.run_id}.{self.format}").resolve()
        try:
            out_path.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Invalid run_id {trace.run_id!r}: path traversal detected"
            ) from None

        if self.format == "jsonl":
            lines = [
                json.dumps(span.to_dict(), separators=(",", ":"))
                for span in trace.spans
            ]
            out_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        else:
            out_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

        return out_path


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def export(
    trace: Trace,
    output_dir: Path | None = None,
) -> Path:
    """Export *trace* as JSON to *output_dir* (default: current directory).

    Returns the path of the written file.
    """
    effective_dir = output_dir or Path.cwd()
    return FileExporter(effective_dir).export(trace)
