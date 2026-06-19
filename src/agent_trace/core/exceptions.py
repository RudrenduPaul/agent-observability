"""
Shared exception types for agent-trace.

Centralised here so that callers can catch a single NetworkGuardError
regardless of which HTTP client (httpx or requests) raised it.
"""

from __future__ import annotations

__all__ = ["NetworkGuardError"]


class NetworkGuardError(RuntimeError):
    """Raised when a live network call is attempted during guarded replay.

    Set ``AGENT_TRACE_NETWORK_GUARD=1`` to activate the guard.  The guard
    exists so that test suites relying on fixtures blow up loudly rather than
    silently hitting real endpoints, which would cause non-deterministic
    results and unexpected API costs.

    Both the httpx and requests interceptors raise this same class, so a
    single ``except NetworkGuardError`` catches either.
    """
