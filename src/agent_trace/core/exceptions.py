"""
Shared exception types for agent-trace.

Centralised here so that callers can catch a single NetworkGuardError
regardless of which HTTP client (httpx or requests) raised it.
"""

from __future__ import annotations

import os

__all__ = ["NetworkGuardError", "guard_active"]


class NetworkGuardError(RuntimeError):
    """Raised when a live network call is attempted during guarded replay.

    Set ``AGENT_TRACE_NETWORK_GUARD=1`` to activate the guard.  The guard
    exists so that test suites relying on fixtures blow up loudly rather than
    silently hitting real endpoints, which would cause non-deterministic
    results and unexpected API costs.

    Both the httpx and requests interceptors raise this same class, so a
    single ``except NetworkGuardError`` catches either.
    """


def guard_active() -> bool:
    """Return True when AGENT_TRACE_NETWORK_GUARD=1 is set in the environment."""
    return os.environ.get("AGENT_TRACE_NETWORK_GUARD", "0") == "1"
