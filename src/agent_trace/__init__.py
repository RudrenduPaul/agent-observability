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
import logging
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, TypeVar

# Re-export canonical model types from their authoritative modules
from agent_trace._replay.fixture import Fixture
from agent_trace.core.exceptions import NetworkGuardError
from agent_trace.core.span import Span, SpanStatus
from agent_trace.core.trace import Trace
from agent_trace.plugins.base import Plugin, PluginBase, SpanPlugin, TracePlugin

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

__all__ = [
    "Fixture",
    "NetworkGuardError",
    "Plugin",
    "PluginBase",
    "ReplayContext",
    "Span",
    "SpanPlugin",
    "SpanStatus",
    "Trace",
    "TracePlugin",
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
        # Transport-patch nesting counter and saved originals are initialised
        # here so their types are declared once and getattr-with-default is
        # never needed.
        self._transport_depth: int = 0
        # Stored as a (sync, async) tuple when patched; None otherwise.
        self._original_httpx_transport_for_url: tuple[Any, Any] | None = None
        self._original_requests_get_adapter: Any = None
        # The fixture that HTTP calls made *in the current context* should be
        # recorded into.  ContextVar (not a plain attribute) so that each
        # asyncio Task / thread gets its own independent value: two
        # concurrently active start_trace(record=True) contexts (e.g. two
        # in-flight requests in a server process) each see only their own
        # fixture here, never each other's.  The class-level httpx/requests
        # patches installed by _patch_httpx/_patch_requests read this at
        # request-dispatch time rather than closing over a single fixture
        # captured whenever the patch happened to first be installed.
        self._active_fixture_var: ContextVar[Fixture | None] = ContextVar(
            "agent_trace_active_fixture", default=None
        )
        # Registered plugins — called on span and trace lifecycle events.
        self._plugins: list[Plugin] = []

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def start_trace(
        self,
        name: str,
        record: bool = False,
        run_id: str | None = None,
        trace_id: str | None = None,
        remote_backend: Any = None,
    ) -> Generator[Trace, None, None]:
        """Start a trace, yield it, then save trace.json on exit.

        Nested calls are supported — the inner trace saves/restores the outer
        one in ``_active_trace_var``.  If *record* is True, all outbound HTTP
        calls during the context are captured into a SQLite fixture at
        ``run_dir/fixture.db``.

        *trace_id*, when supplied, overrides the default random
        ``uuid.uuid4().hex``. Pass a value derived from a stable external
        identity — e.g. a LangGraph run's ``thread_id``/checkpoint id via
        ``agent_trace.integrations.langgraph.derive_trace_id()`` — so that
        two worker processes independently recording "the same" logical
        operation (e.g. an original long-running tool call and its
        checkpoint-swept re-dispatch on a managed platform) produce
        traces sharing one ``trace_id`` and can be recognized/diffed as the
        same logical run after the fact, instead of two unrelated,
        un-linkable random UUIDs.

        *remote_backend*, when supplied (a
        ``agent_trace.exporters.remote_fixture.RemoteFixtureBackend``,
        requires *record* to also be True), durably uploads each HTTP
        exchange to remote storage as it's recorded, and syncs the final
        ``trace.json``/``fixture.db`` on exit — so a worker killed or swept
        mid-run on a managed platform (issue #7417) still has its recording
        recoverable from remote storage, instead of only the worker's own
        ephemeral, developer-inaccessible local filesystem.
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

        # trace_id must be 128-bit hex for OTLP; run_id is the human-readable
        # directory name ("run_abc123").  Always generate them independently
        # unless the caller supplied a deterministic trace_id explicitly.
        trace = Trace(trace_id=trace_id or uuid.uuid4().hex, run_id=effective_run_id)
        trace.metadata["name"] = name
        token: Token[Trace | None] = self._active_trace_var.set(trace)

        self._call_plugin("on_trace_start", trace)

        on_exchange_recorded = self._remote_exchange_callback(
            record, remote_backend, effective_run_id
        )

        # Use Fixture as a context manager when recording; nullcontext() when
        # not, so fixture lifecycle and transport patching are always balanced.
        fixture_ctx: Any = (
            Fixture(
                run_dir / "fixture.db",
                trace_id=trace.trace_id,
                on_exchange_recorded=on_exchange_recorded,
            )
            if record
            else nullcontext()
        )
        try:
            with fixture_ctx as fixture:
                fixture_token: Token[Fixture | None] | None = None
                if fixture is not None:
                    # Set the ContextVar *before* installing the patch so
                    # that any HTTP call made anywhere in this context (or a
                    # nested one) during the patch's lifetime resolves back
                    # to this trace's fixture, even under overlapping
                    # concurrent recordings.
                    fixture_token = self._active_fixture_var.set(fixture)
                    self._install_recording_transport()
                try:
                    yield trace
                except Exception as exc:
                    for span in trace.spans:
                        if span.end_time is None:
                            span.record_exception(exc)
                            span.end(SpanStatus.ERROR)
                    raise
                finally:
                    if fixture is not None:
                        self._uninstall_recording_transport()
                    if fixture_token is not None:
                        self._active_fixture_var.reset(fixture_token)
        finally:
            trace_json_path = run_dir / "trace.json"
            try:
                trace_json_path.write_text(
                    json.dumps(trace.to_dict(), indent=2), encoding="utf-8"
                )
            except OSError as _write_err:
                logger.warning(
                    "agent-trace: could not write trace.json to %s: %s",
                    trace_json_path,
                    _write_err,
                )
            self._sync_run_to_remote(remote_backend, run_dir, effective_run_id)
            self._call_plugin("on_trace_end", trace)
            self._active_trace_var.reset(token)

    @staticmethod
    def _remote_exchange_callback(
        record: bool, remote_backend: Any, run_id: str
    ) -> Any:
        """Return an on_exchange_recorded callback wired to *remote_backend*
        (durably uploading each exchange as it's recorded — see
        agent_trace.exporters.remote_fixture), or None when recording isn't
        active or no remote backend was supplied. Split out of start_trace()
        purely to keep that method's own branch/statement count low."""
        if not record or remote_backend is None:
            return None
        from agent_trace.exporters.remote_fixture import remote_sync_callback

        return remote_sync_callback(remote_backend, run_id)

    @staticmethod
    def _sync_run_to_remote(remote_backend: Any, run_dir: Path, run_id: str) -> None:
        """Best-effort final sync of trace.json/fixture.db to *remote_backend*
        on start_trace() exit. Split out purely to keep start_trace()'s own
        branch/statement count low."""
        if remote_backend is None:
            return
        try:
            from agent_trace.exporters.remote_fixture import sync_run_to_remote

            sync_run_to_remote(run_dir, remote_backend, run_id)
        except Exception:
            logger.warning(
                "agent-trace: failed to sync run %s to remote backend",
                run_id,
                exc_info=True,
            )

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

        Registered plugins receive ``on_span_start`` immediately and
        ``on_span_end`` when ``span.end()`` is called.
        """
        active = self._active_trace_var.get()
        span_id = uuid.uuid4().hex[:16]
        trace_id = active.trace_id if active is not None else uuid.uuid4().hex
        s = Span(name=name, span_id=span_id, trace_id=trace_id, parent_id=parent_id)
        if active is not None:
            active.add_span(s)
        self._call_plugin("on_span_start", s)
        tracer_ref = self
        original_end = s.end

        def _plugin_end(status: SpanStatus = SpanStatus.OK) -> None:
            original_end(status)
            tracer_ref._call_plugin("on_span_end", s)

        s.end = _plugin_end  # type: ignore[method-assign]
        return s

    # ------------------------------------------------------------------
    # Active trace accessor
    # ------------------------------------------------------------------

    @property
    def active_trace(self) -> Trace | None:
        """The currently active :class:`Trace`, or None outside a trace context."""
        return self._active_trace_var.get()

    # ------------------------------------------------------------------
    # Plugin API
    # ------------------------------------------------------------------

    def add_plugin(self, plugin: Plugin) -> None:
        """Register a plugin to receive span and trace lifecycle callbacks.

        Plugins are called synchronously in registration order.  Exceptions
        inside a plugin hook are caught and logged so a buggy plugin cannot
        silently break the caller.

        Example::

            from agent_trace.plugins import PluginBase

            class LogPlugin(PluginBase):
                def on_span_end(self, span):
                    print(span.name, span.duration_ms)

            tracer.add_plugin(LogPlugin())
        """
        if plugin not in self._plugins:
            self._plugins.append(plugin)

    def remove_plugin(self, plugin: Plugin) -> None:
        """Unregister a previously added plugin."""
        try:
            self._plugins.remove(plugin)
        except ValueError:
            pass

    def _call_plugin(self, method: str, arg: Any) -> None:
        """Call *method* on every registered plugin, swallowing exceptions."""
        for plugin in self._plugins:
            fn = getattr(plugin, method, None)
            if fn is not None:
                try:
                    fn(arg)
                except Exception:
                    logger.warning(
                        "agent-trace: plugin %r raised in %s — skipping",
                        plugin,
                        method,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Recording transport patching
    # ------------------------------------------------------------------

    def _install_recording_transport(self) -> None:
        """Monkey-patch httpx and requests to record HTTP traffic.

        Uses a nesting counter so that overlapping/nested
        start_trace(record=True) calls install the class-level patch exactly
        once and only remove it once the *last* active recording exits.
        Only the outermost call installs; inner/overlapping calls are no-ops
        on the patch itself.

        The installed patches do not close over a single fixture — each one
        resolves ``self._active_fixture_var.get()`` fresh at request-dispatch
        time, so this is safe even when two recordings are simultaneously
        active (see the ContextVar comment on ``_active_fixture_var``).
        """
        self._transport_depth += 1
        if self._transport_depth > 1:
            return
        self._patch_httpx()
        self._patch_requests()

    def _uninstall_recording_transport(self) -> None:
        """Restore the original patched methods.

        Only the outermost trace uninstalls; inner traces decrement the counter.
        """
        self._transport_depth = max(0, self._transport_depth - 1)
        if self._transport_depth > 0:
            return
        self._unpatch_httpx()
        self._unpatch_requests()

    def _patch_httpx(self) -> None:
        try:
            import httpx

            from agent_trace.interceptor.httpx_hook import (
                AsyncRecordingTransport,
                RecordingTransport,
            )

            # Patch at request-dispatch time (`_transport_for_url`, called by
            # httpx internally on every single request/redirect hop) rather
            # than at Client.__init__ time.  This fixes two problems with the
            # old __init__-time patch:
            #
            # 1. A client constructed *before* recording activates (e.g. an
            #    LLM client built once at module-import time, as
            #    `langgraph dev`/`make_graph()` entry points typically do)
            #    permanently kept its original transport under the old
            #    design, so recording could never see its traffic no matter
            #    when `start_trace(record=True)` was later entered.
            #    `_transport_for_url` is looked up fresh on every send(), so
            #    pre-existing clients are captured too.
            # 2. A client constructed with an explicit `transport=` (e.g.
            #    langchain-openai's TCP-keepalive transport, or any SDK that
            #    pre-populates the `transport` kwarg) defeated the old
            #    `kwargs.setdefault("transport", ...)` silently — setdefault
            #    never fires when the key is already present.  Here we always
            #    wrap whatever transport httpx would have used (default or
            #    caller-supplied, including per-URL `mounts=` transports)
            #    as `inner`, so nothing bypasses recording.
            active_fixture_var = self._active_fixture_var
            orig_sync = httpx.Client._transport_for_url
            orig_async = httpx.AsyncClient._transport_for_url

            def _patched_sync(client_self: Any, url: Any) -> Any:
                base_transport = orig_sync(client_self, url)
                fixture = active_fixture_var.get()
                if fixture is None or isinstance(base_transport, RecordingTransport):
                    return base_transport
                return RecordingTransport(fixture, inner=base_transport)

            def _patched_async(client_self: Any, url: Any) -> Any:
                base_transport = orig_async(client_self, url)
                fixture = active_fixture_var.get()
                if fixture is None or isinstance(
                    base_transport, AsyncRecordingTransport
                ):
                    return base_transport
                return AsyncRecordingTransport(fixture, inner=base_transport)

            self._original_httpx_transport_for_url = (orig_sync, orig_async)
            setattr(httpx.Client, "_transport_for_url", _patched_sync)
            setattr(httpx.AsyncClient, "_transport_for_url", _patched_async)
        except ImportError:
            pass

    def _patch_requests(self) -> None:
        try:
            import requests

            from agent_trace.interceptor.requests_patch import RecordingAdapter

            # requests.Session.get_adapter(url) is already resolved fresh on
            # every request (not at Session-construction time), so — unlike
            # the old httpx.Client.__init__ patch — this already covers
            # pre-existing Sessions and caller-mounted custom adapters
            # correctly.  The only change here is resolving the fixture from
            # the ContextVar at call time instead of a closed-over value, so
            # concurrent recordings route correctly too.
            active_fixture_var = self._active_fixture_var
            orig = requests.Session.get_adapter

            def _patched(session_self: Any, url: str, **kwargs: Any) -> Any:
                # Call the original dispatch so custom/mounted adapters are
                # respected, then wrap the returned adapter to record the
                # exchange for whichever trace is active in this context.
                inner = orig(session_self, url, **kwargs)
                fixture = active_fixture_var.get()
                if fixture is None or isinstance(inner, RecordingAdapter):
                    return inner
                return RecordingAdapter(fixture, inner=inner)

            self._original_requests_get_adapter = orig
            setattr(requests.Session, "get_adapter", _patched)
        except ImportError:
            pass

    def _unpatch_httpx(self) -> None:
        orig = self._original_httpx_transport_for_url
        if orig is None:
            return
        try:
            import httpx

            orig_sync, orig_async = orig
            setattr(httpx.Client, "_transport_for_url", orig_sync)
            setattr(httpx.AsyncClient, "_transport_for_url", orig_async)
        except ImportError:
            pass
        self._original_httpx_transport_for_url = None

    def _unpatch_requests(self) -> None:
        orig = self._original_requests_get_adapter
        if orig is None:
            return
        try:
            import requests

            setattr(requests.Session, "get_adapter", orig)
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

    - A path to a ``fixture.db`` file directly (e.g. ``Path("fixtures/fixture.db")``).
    - A run directory path (absolute or relative) containing ``fixture.db``.
    - A run-ID string like ``run_abc123`` that is resolved relative to
      *trace_dir* (default: ``~/.agent-trace/runs``).

    Example::

        with replay("run_abc123") as ctx:
            value = ctx.get_metadata("input")

        with replay(Path("fixtures/fixture.db")) as ctx:
            result = my_agent()
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

    fixture_path = p if p.suffix == ".db" else p / "fixture.db"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"No fixture.db found at {fixture_path}. "
            "Did you record this run with record=True?"
        )

    return ReplayContext(fixture_path)
