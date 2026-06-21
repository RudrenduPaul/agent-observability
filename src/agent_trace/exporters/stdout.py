"""
Console exporter — prints a trace as a colored span tree using rich.
Falls back to plain text if rich is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_trace import Span, Trace

__all__ = [
    "StdoutExporter",
    "export",
]

# Colour map: SpanStatus value -> rich colour name
_STATUS_COLOUR: dict[str, str] = {
    "OK": "green",
    "ERROR": "red",
    "UNSET": "yellow",
}

# Plain-text symbol map
_STATUS_SYMBOL: dict[str, str] = {
    "OK": "[OK]",
    "ERROR": "[ERR]",
    "UNSET": "[---]",
}


def _trace_header_info(trace: Trace) -> tuple[str, str]:
    """Return (duration_string, display_name) for trace header lines."""
    ended = [s.end_time for s in trace.spans if s.end_time is not None]
    started = [s.start_time for s in trace.spans]
    dur_str = ""
    if started and ended:
        total_ms = (max(ended) - min(started)) * 1_000
        dur_str = f" ({total_ms:.1f} ms total)"
    display_name = str(trace.metadata.get("name", trace.run_id))
    return dur_str, display_name


class StdoutExporter:
    """Export a :class:`~agent_trace.Trace` as a human-readable span tree.

    Uses ``rich`` for coloured output when available; otherwise falls back to
    plain indented ASCII.
    """

    def export(self, trace: Trace) -> None:
        """Print the full span tree for *trace* to stdout."""
        try:
            self._export_rich(trace)
        except ImportError:
            self._export_plain(trace)

    def export_span(self, span: Span, depth: int = 0) -> None:
        """Export a single span at the given indent depth (plain-text only)."""
        indent = "  " * depth
        sym = _STATUS_SYMBOL.get(span.status.value, "[---]")
        dur = f" ({span.duration_ms:.1f} ms)" if span.duration_ms is not None else ""
        print(f"{indent}{sym} {span.name}{dur}")

    # ------------------------------------------------------------------
    # Rich implementation
    # ------------------------------------------------------------------

    def _export_rich(self, trace: Trace) -> None:
        from rich.console import Console
        from rich.tree import Tree

        console = Console()
        dur_str, display_name = _trace_header_info(trace)
        rich_dur = f"  [dim]({dur_str.strip()})[/dim]" if dur_str else ""

        root_label = (
            f"[bold cyan]Trace:[/bold cyan] {display_name}"
            f"  [dim]{trace.run_id}[/dim]{rich_dur}"
        )
        tree = Tree(root_label)

        # Build parent->children map first so tree construction is order-independent.
        children_map: dict[str | None, list[Any]] = {}
        for span in trace.spans:
            children_map.setdefault(span.parent_id, []).append(span)

        id_to_node: dict[str, Any] = {}

        def _add_children(parent_node: Any, parent_id: str | None) -> None:
            for span in children_map.get(parent_id, []):
                colour = _STATUS_COLOUR.get(span.status.value, "yellow")
                dur = (
                    f"  [dim]({span.duration_ms:.1f} ms)[/dim]"
                    if span.duration_ms is not None
                    else ""
                )
                label = (
                    f"[{colour}]{span.name}[/{colour}]"
                    f"  [{colour}]{span.status.value}[/{colour}]"
                    f"{dur}"
                )
                node = parent_node.add(label)
                id_to_node[span.span_id] = node
                _add_children(node, span.span_id)

        _add_children(tree, None)
        console.print(tree)

    # ------------------------------------------------------------------
    # Plain-text fallback
    # ------------------------------------------------------------------

    def _export_plain(self, trace: Trace) -> None:
        dur_str, display_name = _trace_header_info(trace)
        print(f"Trace: {display_name}  [{trace.run_id}]{dur_str}")

        # Build parent->children mapping for tree printing
        children: dict[str | None, list[Span]] = {}
        for span in trace.spans:
            children.setdefault(span.parent_id, []).append(span)

        def _print_subtree(parent_id: str | None, depth: int) -> None:
            for span in children.get(parent_id, []):
                self.export_span(span, depth=depth)
                _print_subtree(span.span_id, depth + 1)

        _print_subtree(None, depth=1)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_exporter: StdoutExporter = StdoutExporter()


def export(trace: Trace) -> None:
    """Export *trace* to stdout using the default :class:`StdoutExporter`."""
    _default_exporter.export(trace)
