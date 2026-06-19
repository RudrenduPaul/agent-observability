"""
Trace — a collection of Spans sharing a single trace_id.

A Trace is the top-level container returned by the tracer and persisted to
disk.  It intentionally holds no locks or I/O handles; thread-safety is the
responsibility of the Tracer that creates and mutates it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from agent_trace.core.span import Span

__all__ = ["Trace"]

_AttrValue = str | int | float | bool


@dataclass
class Trace:
    """Ordered collection of Spans that belong to one logical agent run.

    ``run_id`` identifies the *execution* (re-running the same code produces
    a new run_id) while ``trace_id`` identifies the *trace structure* — the
    same fixture can be replayed many times with different run_ids but the
    same trace_id so that tooling can diff runs.
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, _AttrValue] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_span(self, span: Span) -> None:
        """Append *span* to this trace, enforcing trace_id consistency.

        If the span carries a different trace_id it means it was created
        before the trace existed (e.g. during library initialisation).  We
        silently correct the mismatch so callers don't need to thread
        trace_id through every call-site.
        """
        if span.trace_id != self.trace_id:
            span.trace_id = self.trace_id
        self.spans.append(span)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_span(self, span_id: str) -> Span | None:
        """Return the first span whose span_id matches, or None."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def root_spans(self) -> list[Span]:
        """Return spans that have no parent (top-level operations)."""
        return [s for s in self.spans if s.parent_id is None]

    def children_of(self, span_id: str) -> list[Span]:
        """Return all direct children of *span_id* in insertion order."""
        return [s for s in self.spans if s.parent_id == span_id]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (no datetime objects, enums by value)."""
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Deserialise from a dict produced by to_dict()."""
        spans = [Span.from_dict(s) for s in data.get("spans", [])]
        return cls(
            trace_id=str(data["trace_id"]),
            run_id=str(data["run_id"]),
            spans=spans,
            metadata={k: v for k, v in data.get("metadata", {}).items()},
        )
