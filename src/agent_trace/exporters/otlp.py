"""
OTLP exporter — sends spans to an OpenTelemetry collector endpoint.

Requires: pip install opentelemetry-exporter-otlp-proto-grpc

This exporter converts agent-trace Span objects to OTLP format and
sends them via gRPC. Use this to integrate with Jaeger, Grafana Tempo,
or any OTLP-compatible backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import SpanStatus

if TYPE_CHECKING:
    from agent_trace import Span, Trace

__all__ = [
    "OTLPExporter",
]

logger = logging.getLogger(__name__)

_OTLP_INSTALL_HINT = (
    "The OTLP exporter requires the OpenTelemetry gRPC package.\n"
    "Install it with:\n\n"
    "    pip install opentelemetry-exporter-otlp-proto-grpc\n"
)

# Nanoseconds per second — OTLP uses int nanoseconds for timestamps
_NS_PER_SEC: int = 1_000_000_000


class OTLPExporter:
    """Export a :class:`~agent_trace.Trace` to an OTLP-compatible backend.

    Converts agent-trace spans to OpenTelemetry ``ResourceSpan`` format and
    ships them to *endpoint* via gRPC.

    Parameters
    ----------
    endpoint:
        gRPC endpoint of the OpenTelemetry collector, e.g.
        ``"http://localhost:4317"`` (the default).
    """

    def __init__(self, endpoint: str = "http://localhost:4317") -> None:
        self.endpoint: str = endpoint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, trace: Trace) -> None:
        """Send all spans in *trace* to the configured OTLP endpoint."""
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.trace import (
                NonRecordingSpan,
                SpanContext,
                StatusCode,
                TraceFlags,
            )
        except ImportError as exc:
            raise ImportError(_OTLP_INSTALL_HINT) from exc

        # Build the status map once per export call rather than per span.
        status_map: dict[SpanStatus, Any] = {
            SpanStatus.OK: StatusCode.OK,
            SpanStatus.ERROR: StatusCode.ERROR,
            SpanStatus.UNSET: StatusCode.UNSET,
        }

        service_name = str(trace.metadata.get("name", "agent-trace"))
        exporter = OTLPSpanExporter(endpoint=self.endpoint)
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_tracer = provider.get_tracer("agent-trace")
        failed = 0

        try:
            for span in trace.spans:
                otlp_data = self._span_to_otlp(span, status_map)
                try:
                    trace_id_int = (
                        int(span.trace_id, 16)
                        if _is_hex(span.trace_id)
                        else hash(span.trace_id) & ((1 << 128) - 1)
                    )
                    parent_ctx = otel_context.get_current()
                    if span.parent_id is not None:
                        parent_span_id_int = (
                            int(span.parent_id, 16)
                            if _is_hex(span.parent_id)
                            else hash(span.parent_id) & ((1 << 64) - 1)
                        )
                        parent_span_context = SpanContext(
                            trace_id=trace_id_int,
                            span_id=parent_span_id_int,
                            is_remote=True,
                            trace_flags=TraceFlags(TraceFlags.SAMPLED),
                        )
                        parent_ctx = otel_trace.set_span_in_context(
                            NonRecordingSpan(parent_span_context)
                        )

                    with otel_tracer.start_as_current_span(
                        span.name,
                        context=parent_ctx,
                        start_time=otlp_data["start_time_unix_nano"],
                    ) as otel_span:
                        for k, v in span.attributes.items():
                            attr_v = (
                                v if isinstance(v, (bool, int, float, str)) else str(v)
                            )
                            otel_span.set_attribute(k, attr_v)
                        for event in span.events:
                            otel_span.add_event(
                                event.name,
                                attributes={
                                    k: str(v) for k, v in event.attributes.items()
                                },
                            )
                        otel_span.set_status(otlp_data["status_code"])

                        if span.end_time is not None:
                            otel_span.end(end_time=otlp_data["end_time_unix_nano"])
                except Exception:
                    failed += 1
                    logger.debug(
                        "agent-trace: failed to export span %r to OTLP",
                        span.span_id,
                        exc_info=True,
                    )
        finally:
            provider.shutdown()

        if failed:
            logger.warning(
                "agent-trace: %d/%d span(s) failed to export to %s",
                failed,
                len(trace.spans),
                self.endpoint,
            )

    def _span_to_otlp(
        self, span: Span, status_map: dict[SpanStatus, Any]
    ) -> dict[str, Any]:
        """Convert a :class:`~agent_trace.Span` to an OTLP-compatible dict.

        The returned dict contains::

            {
                "name": str,
                "trace_id": str,
                "span_id": str,
                "parent_span_id": str | None,
                "start_time_unix_nano": int,
                "end_time_unix_nano": int | None,
                "status_code": StatusCode,
                "attributes": list[KeyValue],
            }
        """
        end_ns: int | None = (
            int(span.end_time * _NS_PER_SEC) if span.end_time is not None else None
        )
        return {
            "name": span.name,
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_id,
            "start_time_unix_nano": int(span.start_time * _NS_PER_SEC),
            "end_time_unix_nano": end_ns,
            "status_code": status_map.get(span.status, status_map[SpanStatus.UNSET]),
            "attributes": [{"key": k, "value": v} for k, v in span.attributes.items()],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_hex(s: str) -> bool:
    try:
        int(s, 16)
        return True
    except ValueError:
        return False
