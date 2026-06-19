"""
Clock abstraction for deterministic replay.

All spans use get_time() from this module — never time.time() directly.
During replay, set_clock(FixtureClock()) swaps the time source so all
recorded timestamps are replayed exactly without wall-clock drift.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token

__all__ = [
    "Clock",
    "FixtureClock",
    "WallClock",
    "get_clock",
    "get_time",
    "restore_clock",
    "set_clock",
]


class Clock(ABC):
    """Abstract base for all time sources used by agent-trace spans."""

    @abstractmethod
    def now(self) -> float:
        """Return current time as a Unix timestamp (seconds since epoch)."""
        ...


class WallClock(Clock):
    """Production clock that delegates to time.time()."""

    def now(self) -> float:
        return time.time()


class FixtureClock(Clock):
    """Replay clock driven by pre-recorded timestamps.

    Initialises to ``time.time()`` so replay spans have meaningful timestamps
    by default.  Call :meth:`advance` with each recorded timestamp to reproduce
    original execution times exactly.
    """

    def __init__(self, initial: float | None = None) -> None:
        self._current: float = initial if initial is not None else time.time()

    def advance(self, timestamp: float) -> None:
        """Set the clock to *timestamp* (seconds since epoch)."""
        self._current = timestamp

    def now(self) -> float:
        return self._current


# One clock per async context (or thread).  Default is the real wall clock so
# that code works without any setup in production.
_clock_var: ContextVar[Clock] = ContextVar("_agent_trace_clock", default=WallClock())  # noqa: B039


def get_clock() -> Clock:
    """Return the active Clock for the current context."""
    return _clock_var.get()


def set_clock(clock: Clock) -> Token[Clock]:
    """Replace the active clock and return a Token for later restoration.

    Always pair with restore_clock() in a finally block to avoid leaking the
    override into sibling async tasks.
    """
    return _clock_var.set(clock)


def restore_clock(token: Token[Clock]) -> None:
    """Undo a previous set_clock() using the Token it returned."""
    _clock_var.reset(token)


def get_time() -> float:
    """Return current time from the active clock.

    This is the only call-site for time inside agent-trace core code.  Using
    this instead of time.time() directly is what makes replay deterministic.
    """
    return _clock_var.get().now()
