"""
Span and SpanEvent data models for agent-trace.

A Span represents a single unit of work within a trace.  Spans are cheap
value objects — they hold no locks, no background threads, and no file
handles.  All mutation goes through the public methods below so that
to_dict() remains consistent with the live state.
"""

from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_trace.core.clock import get_time

__all__ = [
    "Span",
    "SpanEvent",
    "SpanStatus",
]

# Attribute value types accepted everywhere in this module.
_AttrValue = str | int | float | bool


class SpanStatus(str, Enum):
    """Lifecycle status of a Span, modelled after OpenTelemetry's StatusCode.

    CANCELLED is deliberately distinct from ERROR: a span ended because the
    run/task it belonged to was cancelled (e.g. ``asyncio.CancelledError``)
    did not fail — it was cut off mid-flight. Collapsing both into ERROR
    makes a genuine application failure indistinguishable from a cancelled
    run when reading a trace, which matters for diagnosing
    cancellation-triggered data-loss bugs (e.g. unpersisted checkpoints).
    """

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


@dataclass
class SpanEvent:
    """A timestamped annotation attached to a Span.

    Events capture point-in-time occurrences (e.g. "tool call returned") that
    are too fine-grained for their own span.
    """

    name: str
    timestamp: float
    attributes: dict[str, _AttrValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanEvent:
        return cls(
            name=str(data["name"]),
            timestamp=float(data["timestamp"]),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class Span:
    """Immutable-by-convention record of a single traced operation.

    Spans are created open (end_time is None) and closed via end().  After
    end() is called, the span should be treated as read-only — mutating a
    closed span produces undefined behaviour in exporters.

    All timestamps come from core.clock.get_time() so the replay engine can
    substitute pre-recorded values without touching any other code.
    """

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    name: str = ""
    start_time: float = field(default_factory=get_time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, _AttrValue] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def end(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Close the span, recording end_time and setting status."""
        if self.end_time is not None:
            return
        self.end_time = get_time()
        self.status = status

    def add_event(
        self,
        name: str,
        attributes: dict[str, _AttrValue] | None = None,
    ) -> None:
        """Append a named event at the current clock time."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=get_time(),
                attributes=attributes or {},
            )
        )

    def set_attribute(self, key: str, value: _AttrValue) -> None:
        """Set or overwrite a single attribute."""
        self.attributes[key] = value

    def record_exception(
        self, exc: BaseException, status: SpanStatus = SpanStatus.ERROR
    ) -> None:
        """Capture an exception as a SpanEvent and mark the span's status.

        Follows OpenTelemetry's exception semantic conventions so downstream
        exporters (e.g. the OTLP exporter) can surface the stack trace.

        ``status`` defaults to ``SpanStatus.ERROR`` (the historical
        behavior). Pass ``status=SpanStatus.CANCELLED`` for a span ended by
        run/task cancellation so a reader of the trace can tell "this
        failed" apart from "this was cut off mid-flight".
        """
        self.add_event(
            "exception",
            attributes={
                "exception.type": type(exc).__qualname__,
                "exception.message": str(exc),
                "exception.stacktrace": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )
        self.status = status

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def duration_ms(self) -> float | None:
        """Wall duration in milliseconds, or None if the span is still open."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1_000

    # ------------------------------------------------------------------
    # Serialisation — all primitives, no datetime objects
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Enums are stored by value so the output is readable without importing
        this module.  Timestamps are plain floats (Unix seconds) to avoid
        timezone ambiguity and datetime import overhead.
        """
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status.value,
            "attributes": dict(self.attributes),
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        """Deserialise from a dict produced by to_dict()."""
        events = [SpanEvent.from_dict(e) for e in data.get("events", [])]
        return cls(
            span_id=str(data["span_id"]),
            trace_id=str(data["trace_id"]),
            parent_id=data.get("parent_id"),
            name=str(data["name"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"])
            if data.get("end_time") is not None
            else None,
            status=SpanStatus(data.get("status", SpanStatus.UNSET.value)),
            attributes=dict(data.get("attributes", {})),
            events=events,
        )
