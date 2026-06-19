"""
agent-trace — AI agent observability with deterministic record/replay.

Quick start:
    from agent_trace import tracer

    @tracer.instrument(record=True)
    def my_agent(query: str) -> str:
        ...

    result = my_agent("debug this")
    # Trace saved to ~/.agent-trace/runs/run_<id>/

Replay offline:
    from agent_trace import replay

    with replay("run_<id>") as ctx:
        result = my_agent(ctx.get_metadata("input"))
"""

from __future__ import annotations

import functools
import inspect
import json
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

# Re-export canonical model types from their authoritative modules
from agent_trace.core.exceptions import NetworkGuardError
from agent_trace.core.span import Span, SpanStatus
from agent_trace.core.trace import Trace

if TYPE_CHECKING:
    from agent_trace._replay.fixture import Fixture

__version__ = "0.1.0"

__all__ = [
    "NetworkGuardError",
    "ReplayContext",
    "Span",
    "SpanStatus",
    "Trace",
    "Tracer",
    "replay",
    "tracer",
]

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class Tracer:
    """Central orchestrator for trace collection.

    Create one global instance (``tracer = Tracer()``) and use it across your
    application.  All methods are thread-safe and async-safe: each coroutine
    or thread has its own active trace via ContextVar.
    """

    def __init__(self, trace_dir: Path | None = None) -> None:
        self._trace_dir: Path = trace_dir or (Path.home() / ".agent-trace" / "runs")
        # ContextVar gives each async task / thread its own active trace.
        # This replaces the previous threading.Lock + single attribute approach
        # which was not safe for concurrent asyncio agents.
        self._active_trace_var: ContextVar[Trace | None] = ContextVar(
            "agent_trace_active_trace", default=None
        )

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def start_trace(
        self,
        name: str,
        record: bool = False,
        run_id: str | None = None,
    ) -> Generator[Trace, None, None]:
        """Start a trace, yield it, then save trace.json on exit.

        Nested calls are supported — the inner trace saves/restores the outer
        one in ``_active_trace_var``.  If *record* is True, all outbound HTTP
        calls during the context are captured into a SQLite fixture at
        ``run_dir/fixture.db``.
        """
        effective_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        base = self._trace_dir.resolve()
        run_dir = (base / effective_run_id).resolve()
        try:
            run_dir.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Invalid run_id {effective_run_id!r}: path traversal detected"
            ) from None
        run_dir.mkdir(parents=True, exist_ok=True)

        trace = Trace(trace_id=effective_run_id, run_id=effective_run_id)
        trace.metadata["name"] = name

        fixture: Any = None
        if record:
            from agent_trace._replay.fixture import Fixture as _Fixture

            fixture = _Fixture(run_dir / "fixture.db", trace_id=effective_run_id)
            self._install_recording_transport(fixture)

        # ContextVar.set() returns a Token that restores the previous value on reset().
        token: Token[Trace | None] = self._active_trace_var.set(trace)

        try:
            yield trace
        except Exception as exc:
            # Mark any open spans as ERROR
            for span in trace.spans:
                if span.end_time is None:
                    span.record_exception(exc)
                    span.end(SpanStatus.ERROR)
            raise
        finally:
            if record:
                self._uninstall_recording_transport()
            if fixture is not None:
                fixture.close()

            # Persist trace JSON
            trace_json_path = run_dir / "trace.json"
            try:
                trace_json_path.write_text(
                    json.dumps(trace.to_dict(), indent=2), encoding="utf-8"
                )
            except OSError:
                pass

            self._active_trace_var.reset(token)

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def instrument(
        self,
        record: bool = False,
        name: str | None = None,
    ) -> Callable[[F], F]:
        """Decorator that wraps a function in :meth:`start_trace`.

        Works for both sync and async functions::

            @tracer.instrument(record=True)
            async def my_agent(query: str) -> str:
                ...
        """

        def decorator(fn: F) -> F:
            trace_name = name or fn.__name__

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.start_trace(trace_name, record=record):
                        return await fn(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_trace(trace_name, record=record):
                    return fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    # ------------------------------------------------------------------
    # Span management
    # ------------------------------------------------------------------

    @contextmanager
    def span(
        self,
        name: str,
        parent_id: str | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager that creates a span and auto-calls :meth:`Span.end`.

        On success the span is closed with ``SpanStatus.OK``; on exception it
        is closed with ``SpanStatus.ERROR`` and the exception is re-raised.
        """
        s = self.start_span(name, parent_id=parent_id)
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            # record_exception already sets status=ERROR; call end explicitly
            # to capture end_time.
            if s.end_time is None:
                s.end(SpanStatus.ERROR)
            raise
        else:
            if s.end_time is None:
                s.end(SpanStatus.OK)

    def start_span(
        self,
        name: str,
        parent_id: str | None = None,
    ) -> Span:
        """Create and register a :class:`Span` on the active trace.

        If there is no active trace this is a no-op that returns a detached
        span (it will not appear in any serialised output).
        """
        active = self._active_trace_var.get()

        span_id = uuid.uuid4().hex[:16]
        trace_id = active.trace_id if active is not None else uuid.uuid4().hex

        s = Span(
            name=name,
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
        )

        if active is not None:
            active.add_span(s)

        return s

    # ------------------------------------------------------------------
    # Active trace accessor
    # ------------------------------------------------------------------

    @property
    def active_trace(self) -> Trace | None:
        """The currently active :class:`Trace`, or None outside a trace context."""
        return self._active_trace_var.get()

    # ------------------------------------------------------------------
    # Recording transport patching
    # ------------------------------------------------------------------

    def _install_recording_transport(self, fixture: Any) -> None:
        """Monkey-patch httpx and requests to record all HTTP traffic.

        Uses a nesting counter so that nested start_trace(record=True) calls
        don't overwrite the saved original with an already-patched method.
        Only the outermost call saves + installs; inner calls are no-ops.
        """
        depth = getattr(self, "_transport_depth", 0)
        self._transport_depth: int = depth + 1
        if depth > 0:
            # Already patched by an outer trace — don't double-patch.
            return

        try:
            import httpx

            from agent_trace.interceptor.httpx_hook import RecordingTransport

            _orig_init = httpx.Client.__init__

            def _patched_init(client_self: Any, *args: Any, **kwargs: Any) -> None:
                kwargs.setdefault("transport", RecordingTransport(fixture))
                _orig_init(client_self, *args, **kwargs)

            self._original_httpx_init: Any = _orig_init
            setattr(httpx.Client, "__init__", _patched_init)
        except ImportError:
            pass

        try:
            import requests

            from agent_trace.interceptor.requests_patch import RecordingAdapter

            _orig_get_adapter = requests.Session.get_adapter

            def _patched_get_adapter(session_self: Any, url: str, **kwargs: Any) -> Any:
                return RecordingAdapter(fixture)

            self._original_requests_get_adapter: Any = _orig_get_adapter
            setattr(requests.Session, "get_adapter", _patched_get_adapter)
        except ImportError:
            pass

    def _uninstall_recording_transport(self) -> None:
        """Restore the original ``__init__`` / ``get_adapter`` methods.

        Only the outermost trace uninstalls; inner traces decrement the counter.
        """
        depth = getattr(self, "_transport_depth", 1)
        self._transport_depth = max(0, depth - 1)
        if self._transport_depth > 0:
            return  # Still nested — leave patch in place.

        orig_httpx = getattr(self, "_original_httpx_init", None)
        if orig_httpx is not None:
            try:
                import httpx

                setattr(httpx.Client, "__init__", orig_httpx)
            except ImportError:
                pass
            self._original_httpx_init = None

        orig_requests = getattr(self, "_original_requests_get_adapter", None)
        if orig_requests is not None:
            try:
                import requests

                setattr(requests.Session, "get_adapter", orig_requests)
            except ImportError:
                pass
            self._original_requests_get_adapter = None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

tracer: Tracer = Tracer()


# ---------------------------------------------------------------------------
# ReplayContext
# ---------------------------------------------------------------------------


class ReplayContext:
    """Context manager returned by :func:`replay`.

    Delegates to :func:`agent_trace._replay.engine.replay_context` so that the
    :class:`~agent_trace.core.clock.FixtureClock` is installed and network
    calls are served from the fixture without touching real endpoints.
    """

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path: Path = fixture_path
        self._fixture: Fixture | None = None
        self._ctx_manager: Any = None

    def __enter__(self) -> ReplayContext:
        from agent_trace._replay.engine import replay_context

        # replay_context yields the Fixture; we capture it via a one-shot
        # generator wrapper so we can expose it as self._fixture.
        cm = replay_context(self._fixture_path)
        self._fixture = cm.__enter__()
        self._ctx_manager = cm
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool | None:
        result: bool | None = None
        if self._ctx_manager is not None:
            result = self._ctx_manager.__exit__(exc_type, exc_val, exc_tb)
        return result

    @property
    def fixture(self) -> Fixture:
        """The underlying :class:`~agent_trace._replay.fixture.Fixture`."""
        if self._fixture is None:
            raise RuntimeError("ReplayContext must be used as a context manager.")
        return self._fixture

    def get_metadata(self, key: str) -> str | None:
        """Look up a metadata value stored in the fixture."""
        return self.fixture.get_metadata(key)


# ---------------------------------------------------------------------------
# replay() factory
# ---------------------------------------------------------------------------


def replay(
    run_id_or_path: str | Path,
    trace_dir: Path | None = None,
) -> ReplayContext:
    """Return a :class:`ReplayContext` for the given run ID or path.

    *run_id_or_path* may be either:

    - A run directory path (absolute or relative) containing ``fixture.db``.
    - A run-ID string like ``run_abc123`` that is resolved relative to
      *trace_dir* (default: ``~/.agent-trace/runs``).

    Example::

        with replay("run_abc123") as ctx:
            value = ctx.get_metadata("input")
    """
    p = Path(run_id_or_path)
    if not p.is_absolute():
        base = Path(trace_dir or (Path.home() / ".agent-trace" / "runs")).resolve()
        p = (base / p).resolve()
        try:
            p.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Invalid run path {run_id_or_path!r}: path traversal detected"
            ) from None
    else:
        p = p.resolve()

    fixture_path = p / "fixture.db"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"No fixture.db found at {fixture_path}. "
            "Did you record this run with record=True?"
        )

    return ReplayContext(fixture_path)
