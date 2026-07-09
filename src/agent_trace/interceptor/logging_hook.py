"""
Logging-capture layer for non-HTTP, non-callback runtime warnings.

Some real, reproducible agent failures never raise an exception and never
touch the network — e.g. LangGraph's pregel scheduler logging ``"wrote to
unknown channel X, ignoring it"`` when ``push_ui_message`` targets an
unregistered state channel (langgraph#5464). Neither the HTTP interceptor
(``agent_trace.interceptor.httpx_hook``/``requests_patch``) nor
``LangGraphTracer``'s callback handlers
(``agent_trace.integrations.langgraph``) can ever see this class of
failure — nothing in agent-trace previously attached to Python's own
``logging`` module at all.

Usage::

    from agent_trace import tracer
    from agent_trace.interceptor.logging_hook import capture_logging

    with tracer.start_trace("my_graph") as trace:
        with capture_logging(tracer, logger_names=["langgraph"]):
            graph.invoke(state, config={"callbacks": [...]})

Every ``WARNING``-or-above record emitted by any of the named loggers while
the ``with`` block is active is persisted as a ``runtime_log`` event on a
dedicated ``logging:capture`` span — independent of, and in addition to,
whatever HTTP/callback capture is also active.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Tracer

__all__ = ["capture_logging"]

logger = logging.getLogger(__name__)

# Bounds — this is diagnostic capture, not a full log mirror. A pathological
# hot loop logging thousands of warnings must not grow a single span's event
# list unboundedly; logging.captured_count keeps counting past the cap.
_MAX_ATTR_LEN = 4_000
_MAX_LOG_EVENTS = 200

# Loggers known to emit real, reproducible non-exception runtime warnings
# relevant to agent failures — e.g. langgraph#5464's pregel-scheduler
# "wrote to unknown channel" line. Callers can pass their own logger_names
# to widen or narrow this.
_DEFAULT_LOGGER_NAMES: tuple[str, ...] = ("langgraph", "langchain_core")


class _SpanLoggingHandler(logging.Handler):
    """logging.Handler that persists each record as a SpanEvent on *span*.

    Formatting/attribute-extraction failures on a single record must never
    break the caller's real logging pipeline — every step is best-effort.
    """

    def __init__(self, span: Span, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._span = span
        self.captured_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.captured_count += 1
        if self.captured_count > _MAX_LOG_EVENTS:
            return
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        try:
            self._span.add_event(
                "runtime_log",
                attributes={
                    "log.level": record.levelname,
                    "log.logger": record.name,
                    "log.message": str(message)[:_MAX_ATTR_LEN],
                    "log.filename": str(record.pathname),
                    "log.lineno": int(record.lineno),
                },
            )
        except Exception:
            logger.debug(
                "agent-trace: failed to record captured log event",
                exc_info=True,
            )


@contextlib.contextmanager
def capture_logging(
    tracer: Tracer,
    *,
    logger_names: Sequence[str] = _DEFAULT_LOGGER_NAMES,
    level: int = logging.WARNING,
    span_name: str = "logging:capture",
) -> Generator[Span, None, None]:
    """Attach a logging.Handler to *logger_names* for the lifetime of the
    ``with`` block, persisting each captured record onto a dedicated
    ``logging:capture`` span (opened via *tracer*).

    *logger_names* defaults to LangGraph/LangChain's own logger namespaces
    — the loggers a pregel-scheduler warning like langgraph#5464's actually
    goes through. Pass an explicit list to widen/narrow the set.

    The span closes ``OK`` when the block exits normally (log capture is a
    diagnostic side-channel, not a success/failure signal in itself) or
    ``ERROR`` if the block itself raises (the exception is recorded onto the
    span, then re-raised unchanged).
    """
    span = tracer.start_span(span_name)
    handler = _SpanLoggingHandler(span, level=level)
    attached: list[logging.Logger] = []
    try:
        for name in logger_names:
            target = logging.getLogger(name)
            target.addHandler(handler)
            attached.append(target)
        yield span
    except Exception as exc:
        span.record_exception(exc)
        if span.end_time is None:
            span.end(SpanStatus.ERROR)
        raise
    else:
        span.set_attribute("logging.captured_count", handler.captured_count)
        if span.end_time is None:
            span.end(SpanStatus.OK)
    finally:
        for target in attached:
            target.removeHandler(handler)
