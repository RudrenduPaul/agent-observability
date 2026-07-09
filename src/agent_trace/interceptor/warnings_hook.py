"""
Python `warnings`-capture layer for library-raised runtime warnings.

Some real, reproducible failures never raise an exception and never touch
the network at all — e.g. langgraph#5628's ``RuntimeWarning: Failed to trim
messages to fit within max_tokens limit before summarization``, raised
entirely inside local token-counting logic before any HTTP call is
attempted. Nothing in agent-trace previously captured arbitrary
library-raised ``warnings.warn()`` calls (the only prior ``warnings.warn()``
call sites in the codebase were agent-trace's own internal warnings in
``httpx_hook.py``/``requests_patch.py``) — this is a related but distinct
gap from ``agent_trace.interceptor.logging_hook`` (that one targets
Python's ``logging`` module for a different subsystem's warning path —
LangGraph's pregel-scheduler log line for langgraph#5464; this one targets
Python's ``warnings`` module, requiring a ``warnings.catch_warnings()``/
``showwarning`` override instead of a ``logging.Handler``).

Usage::

    from agent_trace import tracer
    from agent_trace.interceptor.warnings_hook import capture_warnings

    with tracer.start_trace("my_graph") as trace:
        with capture_warnings(tracer):
            graph.invoke(state, config={"callbacks": [...]})

Every ``UserWarning``/``RuntimeWarning`` (or whatever *categories* is
passed) raised anywhere during the ``with`` block is persisted as a
``runtime_warning`` event on a dedicated ``warnings:capture`` span —
independent of, and in addition to, whatever HTTP/callback capture is also
active.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Tracer

__all__ = ["capture_warnings"]

logger = logging.getLogger(__name__)

# Bounds — this is diagnostic capture, not a full warnings mirror. A
# pathological hot loop emitting thousands of warnings must not grow a
# single span's event list unboundedly; warnings.captured_count keeps
# counting past the cap.
_MAX_ATTR_LEN = 4_000
_MAX_WARNING_EVENTS = 200

_DEFAULT_CATEGORIES: tuple[type[Warning], ...] = (UserWarning, RuntimeWarning)


@contextlib.contextmanager
def capture_warnings(
    tracer: Tracer,
    *,
    categories: tuple[type[Warning], ...] = _DEFAULT_CATEGORIES,
    span_name: str = "warnings:capture",
) -> Generator[Span, None, None]:
    """Install a ``warnings.catch_warnings()`` context with a custom
    ``showwarning`` override for the lifetime of the ``with`` block,
    persisting every caught warning matching *categories* onto a dedicated
    ``warnings:capture`` span (opened via *tracer*).

    Uses ``warnings.simplefilter("always")`` inside the ``catch_warnings()``
    context so a warning that Python would otherwise only show once per
    location (the default ``"default"`` filter action) is still captured on
    every occurrence, without changing any filter state outside this
    ``with`` block — ``catch_warnings()`` saves and restores
    ``warnings.filters``/``showwarning`` on exit regardless of how the block
    exits.

    The span closes ``OK`` when the block exits normally (warning capture
    is a diagnostic side-channel, not a success/failure signal in itself)
    or ``ERROR`` if the block itself raises (the exception is recorded onto
    the span, then re-raised unchanged).
    """
    span = tracer.start_span(span_name)
    count = 0

    def _showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        nonlocal count
        if not (isinstance(category, type) and issubclass(category, categories)):
            return
        count += 1
        if count > _MAX_WARNING_EVENTS:
            return
        try:
            span.add_event(
                "runtime_warning",
                attributes={
                    "warning.category": category.__name__,
                    "warning.message": str(message)[:_MAX_ATTR_LEN],
                    "warning.filename": str(filename),
                    "warning.lineno": int(lineno),
                },
            )
        except Exception:
            logger.debug(
                "agent-trace: failed to record captured warning event",
                exc_info=True,
            )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = _showwarning
            yield span
    except Exception as exc:
        span.record_exception(exc)
        if span.end_time is None:
            span.end(SpanStatus.ERROR)
        raise
    else:
        span.set_attribute("warnings.captured_count", count)
        if span.end_time is None:
            span.end(SpanStatus.OK)
