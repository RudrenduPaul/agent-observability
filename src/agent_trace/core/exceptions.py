"""
Shared exception types for agent-trace.

Centralised here so that callers can catch a single NetworkGuardError
regardless of which HTTP client (httpx or requests) raised it.
"""

from __future__ import annotations

import os

__all__ = ["NetworkGuardError", "RunawayToolCallLoopError", "guard_active"]


class NetworkGuardError(RuntimeError):
    """Raised when a live network call is attempted during guarded replay.

    Set ``AGENT_TRACE_NETWORK_GUARD=1`` to activate the guard.  The guard
    exists so that test suites relying on fixtures blow up loudly rather than
    silently hitting real endpoints, which would cause non-deterministic
    results and unexpected API costs.

    Both the httpx and requests interceptors raise this same class, so a
    single ``except NetworkGuardError`` catches either.
    """


class RunawayToolCallLoopError(RuntimeError):
    """Raised by RecordingTransport/AsyncRecordingTransport's optional
    live loop guard (``loop_guard_threshold=``) once a configurable number
    of *consecutive* tool-call-only responses have been recorded for the
    same host during an active recording session.

    This is the live-recording signal for issue #3097 — a model that never
    stops emitting ``tool_calls`` and never returns a final tool-call-free
    message, silently burning through the full context window (and API
    budget) before anyone notices. Replay-after-the-fact only helps
    diagnose a loop that already happened and already cost the tokens; this
    guard is what actually catches it early, during the run that is still
    burning the budget.

    Raised (rather than merely warned) only when the transport was
    constructed with ``on_loop_detected=agent_trace.interceptor.httpx_hook.
    raise_on_loop_detected`` (or an equivalent custom callback) — the
    default ``on_loop_detected`` only emits a ``UserWarning``.
    """


def guard_active() -> bool:
    """Return True when AGENT_TRACE_NETWORK_GUARD=1 is set in the environment."""
    return os.environ.get("AGENT_TRACE_NETWORK_GUARD", "0") == "1"
