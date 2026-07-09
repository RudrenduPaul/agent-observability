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

import atexit
import functools
import inspect
import json
import logging
import os
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
        # Honours AGENT_TRACE_TRACE_DIR (same env var agent_trace._cli's
        # _trace_dir() reads) so that a tracer constructed anonymously in a
        # process started via `agent-trace run` — most importantly the
        # global `tracer` singleton, which AGENT_TRACE_AUTO_RECORD activates
        # on below — writes to the same location the CLI will later look in
        # by default, without requiring the caller to thread trace_dir
        # through by hand.
        env_trace_dir = os.environ.get("AGENT_TRACE_TRACE_DIR")
        default_trace_dir = Path.home() / ".agent-trace" / "runs"
        self._trace_dir: Path = trace_dir or (
            Path(env_trace_dir) if env_trace_dir else default_trace_dir
        )
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
        # Stored as a (insecure, secure) tuple when patched; None otherwise.
        # aio variants are stored separately since grpc.aio is a lazy import
        # (importing it eagerly would force asyncio C-extension loading for
        # every agent-trace user, even ones who never touch gRPC).
        self._original_grpc_channel_fns: tuple[Any, Any] | None = None
        self._original_grpc_aio_channel_fns: tuple[Any, Any] | None = None
        self._original_aiohttp_request: Any = None
        # Registered plugins — called on span and trace lifecycle events.
        self._plugins: list[Plugin] = []
        # Set by start_auto_record()/AGENT_TRACE_AUTO_RECORD when this
        # tracer is recording process-wide, outside any `with
        # start_trace(...)` block the caller owns — see start_auto_record()
        # docstring. None when no auto-record session is active.
        self._auto_record_state: dict[str, Any] | None = None

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
    # Auto-record — process-wide activation with no enclosing `with` block
    # ------------------------------------------------------------------
    #
    # start_trace()/instrument() both require the caller to own the
    # top-level invocation so they have somewhere to put a `with
    # tracer.start_trace(record=True):` block. That assumption breaks for
    # any framework-managed server process the developer doesn't control
    # the entrypoint of — e.g. `langgraph dev`/LangGraph Studio, which
    # imports the developer's `make_graph()` once at server startup and
    # then owns every subsequent invocation's lifecycle itself. The methods
    # below (and the AGENT_TRACE_AUTO_RECORD env var read at import time,
    # below the class) are the supported mechanism for that case: recording
    # activates for the remaining lifetime of the process instead of one
    # caller-scoped block.

    def start_auto_record(
        self,
        name: str = "auto-record",
        run_id: str | None = None,
    ) -> Path:
        """Activate process-wide recording with no enclosing `with` block.

        Unlike :meth:`start_trace`, this has no natural "end" the caller is
        expected to reach — it's meant for a process whose top-level
        invocation the caller does not own (e.g. a `langgraph dev`/
        LangGraph Studio server process). Everything that happens for the
        remainder of the process — every HTTP exchange, every span opened
        via :meth:`start_span`/:meth:`span` — is captured into one Trace
        and one Fixture, exactly as :meth:`start_trace` does, except the
        capture window is "until the process exits or
        :meth:`stop_auto_record` is called" instead of "for the duration of
        one `with` block".

        An ``atexit`` hook is registered so ``trace.json``/``fixture.db``
        are flushed on ordinary interpreter shutdown even if the caller
        never calls :meth:`stop_auto_record` explicitly (the common case —
        the process is killed by its supervisor, not shut down by the
        developer's own code).

        Idempotent: calling this while an auto-record session is already
        active logs a warning and returns the existing session's run
        directory unchanged, rather than leaking a second Fixture/patch
        layer.

        Returns the run directory (``trace_dir/<run_id>``) recording is
        being written to.

        Coarser-grained than :meth:`start_trace` by design: because there's
        no well-defined caller-owned "end", a long-lived process
        accumulates every exchange/span for its entire remaining lifetime
        into a single trace/fixture pair, not one logical run per
        invocation. Prefer :meth:`start_trace` whenever the caller genuinely
        owns the top-level invocation.
        """
        if self._auto_record_state is not None:
            logger.warning(
                "agent-trace: start_auto_record() called while an auto-record "
                "session is already active (run_dir=%s) — ignoring.",
                self._auto_record_state["run_dir"],
            )
            return self._auto_record_state["run_dir"]  # type: ignore[no-any-return]

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

        trace = Trace(trace_id=uuid.uuid4().hex, run_id=effective_run_id)
        trace.metadata["name"] = name
        trace.metadata["auto_record"] = True
        trace_token = self._active_trace_var.set(trace)

        fixture = Fixture(run_dir / "fixture.db", trace_id=trace.trace_id)
        fixture_token = self._active_fixture_var.set(fixture)
        self._install_recording_transport()

        atexit_callback = self.stop_auto_record
        atexit.register(atexit_callback)

        self._auto_record_state = {
            "run_dir": run_dir,
            "trace": trace,
            "trace_token": trace_token,
            "fixture": fixture,
            "fixture_token": fixture_token,
            "atexit_callback": atexit_callback,
        }
        self._call_plugin("on_trace_start", trace)
        logger.info("agent-trace: auto-record active — writing to %s", run_dir)
        return run_dir

    def stop_auto_record(self) -> None:
        """Stop a :meth:`start_auto_record` session, flushing
        ``trace.json``/closing ``fixture.db``. A no-op when no auto-record
        session is active (safe to call from an ``atexit`` hook even after
        an explicit call already ran)."""
        state = self._auto_record_state
        if state is None:
            return
        self._auto_record_state = None

        try:
            self._uninstall_recording_transport()
        finally:
            fixture: Fixture = state["fixture"]
            try:
                fixture.close()
            except Exception:
                logger.warning(
                    "agent-trace: error closing auto-record fixture", exc_info=True
                )
            try:
                self._active_fixture_var.reset(state["fixture_token"])
            except ValueError:
                # reset() requires the same Context the token's set() ran
                # in; an atexit callback can run in a different Context
                # than the original start_auto_record() call. Best-effort —
                # the ContextVar's process-lifetime value no longer matters
                # once the process is shutting down.
                pass

        trace: Trace = state["trace"]
        run_dir: Path = state["run_dir"]
        try:
            (run_dir / "trace.json").write_text(
                json.dumps(trace.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError as _write_err:
            logger.warning(
                "agent-trace: could not write trace.json to %s: %s",
                run_dir / "trace.json",
                _write_err,
            )
        self._call_plugin("on_trace_end", trace)

        try:
            self._active_trace_var.reset(state["trace_token"])
        except ValueError:
            pass

        # atexit.unregister() is documented to never raise, even when the
        # callback was never registered or was already removed.
        atexit.unregister(state["atexit_callback"])

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
        self._patch_grpc()
        self._patch_aiohttp()

    def _uninstall_recording_transport(self) -> None:
        """Restore the original patched methods.

        Only the outermost trace uninstalls; inner traces decrement the counter.
        """
        self._transport_depth = max(0, self._transport_depth - 1)
        if self._transport_depth > 0:
            return
        self._unpatch_httpx()
        self._unpatch_requests()
        self._unpatch_grpc()
        self._unpatch_aiohttp()

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

    def _patch_aiohttp(self, fixture: Any) -> None:
        try:
            import aiohttp

            from agent_trace.interceptor.aiohttp_hook import make_recording_request

            orig_request = aiohttp.ClientSession._request

            self._original_aiohttp_request = orig_request
            setattr(
                aiohttp.ClientSession,
                "_request",
                make_recording_request(fixture, orig_request),
            )
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

    def _patch_grpc(self) -> None:
        """Monkey-patch grpc's channel factories to record every RPC.

        grpc.insecure_channel/secure_channel are plain module-level
        functions (not a shared base-class method the way httpx.Client is),
        so we patch the module attribute itself. See
        agent_trace/interceptor/grpc_hook.py's module docstring for why this
        is the correct interception point for google-api-core-backed SDKs.

        Like _patch_httpx/_patch_requests, the fixture is resolved from
        self._active_fixture_var at call time (not closed over at patch-
        install time) so concurrently active start_trace(record=True)
        contexts each record into their own fixture.
        """
        active_fixture_var = self._active_fixture_var
        try:
            import grpc

            from agent_trace.interceptor.grpc_hook import GRPCRecordingInterceptor

            orig_insecure = grpc.insecure_channel
            orig_secure = grpc.secure_channel

            def _patched_insecure(
                target: str, options: Any = None, compression: Any = None
            ) -> Any:
                channel = orig_insecure(
                    target, options=options, compression=compression
                )
                fixture = active_fixture_var.get()
                if fixture is None:
                    return channel
                return grpc.intercept_channel(
                    channel, GRPCRecordingInterceptor(fixture, target)
                )

            def _patched_secure(
                target: str,
                credentials: Any,
                options: Any = None,
                compression: Any = None,
            ) -> Any:
                channel = orig_secure(
                    target, credentials, options=options, compression=compression
                )
                fixture = active_fixture_var.get()
                if fixture is None:
                    return channel
                return grpc.intercept_channel(
                    channel, GRPCRecordingInterceptor(fixture, target)
                )

            self._original_grpc_channel_fns = (orig_insecure, orig_secure)
            grpc.insecure_channel = _patched_insecure
            grpc.secure_channel = _patched_secure
        except ImportError:
            pass

        try:
            from grpc import aio

            from agent_trace.interceptor.grpc_hook import AsyncGRPCRecordingInterceptor

            orig_aio_insecure = aio.insecure_channel
            orig_aio_secure = aio.secure_channel

            def _patched_aio_insecure(target: str, **kwargs: Any) -> Any:
                fixture = active_fixture_var.get()
                if fixture is None:
                    return orig_aio_insecure(target, **kwargs)
                interceptors = list(kwargs.pop("interceptors", None) or [])
                interceptors.append(AsyncGRPCRecordingInterceptor(fixture, target))
                return orig_aio_insecure(target, interceptors=interceptors, **kwargs)

            def _patched_aio_secure(
                target: str, credentials: Any, **kwargs: Any
            ) -> Any:
                fixture = active_fixture_var.get()
                if fixture is None:
                    return orig_aio_secure(target, credentials, **kwargs)
                interceptors = list(kwargs.pop("interceptors", None) or [])
                interceptors.append(AsyncGRPCRecordingInterceptor(fixture, target))
                return orig_aio_secure(
                    target, credentials, interceptors=interceptors, **kwargs
                )

            self._original_grpc_aio_channel_fns = (orig_aio_insecure, orig_aio_secure)
            aio.insecure_channel = _patched_aio_insecure
            aio.secure_channel = _patched_aio_secure
        except ImportError:
            pass

    def _unpatch_grpc(self) -> None:
        orig = self._original_grpc_channel_fns
        if orig is not None:
            try:
                import grpc

                orig_insecure, orig_secure = orig
                grpc.insecure_channel = orig_insecure
                grpc.secure_channel = orig_secure
            except ImportError:
                pass
            self._original_grpc_channel_fns = None

        orig_aio = self._original_grpc_aio_channel_fns
        if orig_aio is not None:
            try:
                from grpc import aio

                orig_aio_insecure, orig_aio_secure = orig_aio
                aio.insecure_channel = orig_aio_insecure
                aio.secure_channel = orig_aio_secure
            except ImportError:
                pass
            self._original_grpc_aio_channel_fns = None

    def _patch_aiohttp(self) -> None:
        """Monkey-patch aiohttp.ClientSession._request to record every call.

        Like _patch_httpx/_patch_requests/_patch_grpc, the fixture is
        resolved from self._active_fixture_var at call time (not closed
        over at patch-install time) so concurrently active
        start_trace(record=True) contexts each record into their own
        fixture. See agent_trace/interceptor/aiohttp_hook.py's module
        docstring for why this interceptor exists alongside the
        httpx/requests ones.
        """
        active_fixture_var = self._active_fixture_var
        try:
            import aiohttp

            from agent_trace.interceptor.aiohttp_hook import _record_exchange

            orig_request = aiohttp.ClientSession._request

            async def _patched_request(
                session_self: Any, method: str, str_or_url: Any, **kwargs: Any
            ) -> Any:
                response = await orig_request(
                    session_self, method, str_or_url, **kwargs
                )
                fixture = active_fixture_var.get()
                if fixture is not None:
                    await _record_exchange(fixture, method, str_or_url, kwargs, response)
                return response

            self._original_aiohttp_request = orig_request
            setattr(aiohttp.ClientSession, "_request", _patched_request)
        except ImportError:
            pass

    def _unpatch_aiohttp(self) -> None:
        orig = self._original_aiohttp_request
        if orig is None:
            return
        try:
            import aiohttp

            setattr(aiohttp.ClientSession, "_request", orig)
        except ImportError:
            pass
        self._original_aiohttp_request = None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

tracer: Tracer = Tracer()


# ---------------------------------------------------------------------------
# AGENT_TRACE_AUTO_RECORD — process-wide auto-record activation, read once at
# import time. This is the supported mechanism for attaching agent-trace to
# an externally-managed server process (e.g. `langgraph dev`/LangGraph
# Studio) that the developer does not own the top-level invocation of: set
# the env var (directly, or via `agent-trace run -- <command>`, see
# agent_trace._cli.cmd_run) before the process that imports `agent_trace`
# starts, and recording activates on the global `tracer` singleton the
# moment this module is first imported — no `with tracer.start_trace(...)`
# block required anywhere in the developer's own code.
#
# AGENT_TRACE_AUTO_RECORD: "1"/"true"/"yes"/"on" (case-insensitive) enables.
# Anything else (unset, "0", "false", ...) leaves the tracer untouched —
# this whole block is then a no-op with zero overhead.
# AGENT_TRACE_RUN_ID: optional explicit run_id (default: random).
# AGENT_TRACE_AUTO_RECORD_NAME: optional trace name (default: "auto-record").
# AGENT_TRACE_TRACE_DIR: honoured indirectly — the CLI's `agent-trace run`
# sets it explicitly; the module-level `tracer` singleton otherwise uses its
# own default (~/.agent-trace/runs), same as every other entry point.
# ---------------------------------------------------------------------------

_AUTO_RECORD_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _auto_record_enabled_from_env() -> bool:
    raw = os.environ.get("AGENT_TRACE_AUTO_RECORD", "")
    return raw.strip().lower() in _AUTO_RECORD_TRUE_VALUES


def _activate_auto_record_from_env() -> None:
    """Best-effort AGENT_TRACE_AUTO_RECORD activation on the global
    `tracer` singleton. Never raises — a misconfigured env var must not
    break importing `agent_trace` itself."""
    if not _auto_record_enabled_from_env():
        return
    try:
        tracer.start_auto_record(
            name=os.environ.get("AGENT_TRACE_AUTO_RECORD_NAME", "auto-record"),
            run_id=os.environ.get("AGENT_TRACE_RUN_ID") or None,
        )
    except Exception:
        logger.warning(
            "agent-trace: AGENT_TRACE_AUTO_RECORD activation failed", exc_info=True
        )


_activate_auto_record_from_env()


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
