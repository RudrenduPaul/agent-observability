"""
Checkpointer + node-cache instrumentation for the LangGraph integration.

Background
----------

``LangGraphTracer`` (``agent_trace.integrations.langgraph``) only implements
the standard ``BaseCallbackHandler`` chain/LLM/tool lifecycle
(``on_chain_*``/``on_llm_*``/``on_tool_*``). Checkpointer persistence
(``BaseCheckpointSaver.put``/``put_writes``, and the serializer boundary
those go through) and per-node ``CachePolicy`` hit/miss decisions
(``BaseCache.get``/``set``) are structurally different interfaces with no
callback hook at all — confirmed via a repo-wide grep (zero hits for
``checkpoint|serde|dumps_typed|loads_typed|CachePolicy`` anywhere in
``src/agent_trace/`` before this module). Bugs in state persistence (loss on
cancellation, storage bloat, a write silently attributed to the wrong node,
a non-deterministic cache key) are therefore invisible to agent-trace today
regardless of recording mode.

This module wraps three *public, stable* LangGraph interfaces — confirmed by
direct inspection of the installed ``langgraph`` package (see docstrings on
each class/function below), not private/internal modules — so it stays
correct across LangGraph versions the same way the rest of this codebase's
checkpointer-adjacent code does:

  - ``BaseCheckpointSaver`` (``langgraph.checkpoint.base``): the ABC every
    checkpointer implementation (``InMemorySaver``, ``SqliteSaver``,
    ``PostgresSaver``, ...) subclasses. ``TracingCheckpointSaver`` wraps one
    instance of it, recording a span per ``put``/``aput``/``put_writes``/
    ``aput_writes`` call (timestamp via span start/end, payload size,
    whether the call actually completed), and delegates every other method
    unchanged.
  - ``SerializerProtocol`` (``langgraph.checkpoint.serde.base``): the
    ``dumps_typed``/``loads_typed`` boundary every checkpointer calls to
    turn a ``Checkpoint``/write value into bytes and back.
    ``TracingCheckpointSaver`` installs a ``TracingSerde`` wrapper directly
    onto the *wrapped* checkpointer's own ``.serde`` attribute (not just its
    own), so serde calls made internally by the checkpointer's own
    ``put``/``get`` implementations are captured too, not just calls this
    module happens to make directly.
  - ``BaseCache`` (``langgraph.cache.base``): the ABC every node-cache
    backend (``InMemoryCache``, ``RedisCache``, ...) subclasses.
    ``TracingCache`` wraps one instance, recording hit/miss per
    ``get``/``aget`` call and key/count per ``set``/``aset`` call.
  - ``CachePolicy.key_func`` (``langgraph.types``): the function LangGraph
    actually calls to hash a node's input into the cache key. ``BaseCache``
    itself only ever sees the *already-computed* key, never the state
    object that produced it — ``wrap_cache_policy`` traces the key_func
    itself so the actual hashed input is captured, not just the resulting
    key.

Plus two wrapper functions, ``traced_update_state``/``traced_aupdate_state``,
around the public ``CompiledStateGraph.update_state``/``.aupdate_state``
methods: they record which node an external state write was attributed to
(the ``as_node`` argument, when the caller supplied one explicitly) and the
pregel scheduler's resulting ``next`` task list immediately afterward (via
the public ``.get_state()``/``.aget_state()`` call), flagging
``checkpoint.zero_tasks_scheduled=True`` when that list comes back empty —
the exact silent-no-op-resume shape behind issue #4217.

All instrumentation here is strictly additive and best-effort: any failure
recording span data is swallowed (logged at DEBUG) so a bug in this module
can never change the behavior — or break the exception propagation — of the
real checkpointer/cache/update_state call it wraps.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus
from agent_trace.integrations.langgraph import _stringify, _to_attr_string

if TYPE_CHECKING:
    from agent_trace import Tracer

__all__ = [
    "TracingCache",
    "TracingCheckpointSaver",
    "TracingSerde",
    "traced_aupdate_state",
    "traced_update_state",
    "wrap_cache_policy",
]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Checkpointer/cache tracing requires langgraph.\n"
    "Install it with:\n\n"
    "    pip install langgraph\n"
)


def _require_checkpoint_base() -> Any:
    """Lazy import of langgraph.checkpoint.base — raises a clear error if
    langgraph is absent, mirroring _require_langchain_core() in langgraph.py."""
    try:
        from langgraph.checkpoint import base

        return base
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _require_cache_base() -> Any:
    """Lazy import of langgraph.cache.base."""
    try:
        from langgraph.cache import base

        return base
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _extract_thread_id(config: Any) -> str | None:
    """Best-effort thread_id extraction from a LangGraph RunnableConfig."""
    try:
        configurable = (config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id")
        return str(thread_id) if thread_id is not None else None
    except Exception:
        return None


def _estimate_size(value: Any) -> int | None:
    """Best-effort byte-size estimate of *value* — a UTF-8 length of its
    bounded JSON serialization, not a precise wire size. Good enough to spot
    order-of-magnitude storage bloat (the failure class behind issue #7714)
    without depending on any particular checkpointer's own encoding."""
    try:
        return len(_to_attr_string(value, max_len=2_000_000).encode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TracingSerde — checkpointer serde-boundary capture
# ---------------------------------------------------------------------------


class TracingSerde:
    """Wraps a ``SerializerProtocol`` implementation, recording per-call
    payload byte size and duration as a span per ``dumps_typed``/
    ``loads_typed`` call.

    Confirmed against the installed langgraph's
    ``langgraph.checkpoint.serde.base.SerializerProtocol``: the only two
    methods every real serde implementation must provide are
    ``dumps_typed(obj) -> (type_name, bytes)`` and
    ``loads_typed((type_name, bytes)) -> obj``. ``dumps``/``loads`` (the
    legacy untyped shape) are also forwarded for serde implementations that
    still expose them, but are not the primary capture point since
    ``BaseCheckpointSaver`` upgrades any untyped serde to a typed one via
    ``maybe_add_typed_methods`` before ever calling it.
    """

    def __init__(self, inner: Any, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        t0 = time.monotonic()
        type_name, data = self._inner.dumps_typed(obj)
        self._record("dumps_typed", type_name, len(data), time.monotonic() - t0)
        return type_name, data

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        t0 = time.monotonic()
        result = self._inner.loads_typed(data)
        type_name, raw = data
        self._record("loads_typed", type_name, len(raw), time.monotonic() - t0)
        return result

    def dumps(self, obj: Any) -> bytes:
        t0 = time.monotonic()
        data: bytes = self._inner.dumps(obj)
        self._record("dumps", type(obj).__name__, len(data), time.monotonic() - t0)
        return data

    def loads(self, data: bytes) -> Any:
        t0 = time.monotonic()
        result = self._inner.loads(data)
        self._record("loads", type(result).__name__, len(data), time.monotonic() - t0)
        return result

    def __getattr__(self, name: str) -> Any:
        # Anything not explicitly wrapped above (e.g. a serde-specific
        # extension method) passes through to the real serde unchanged.
        return getattr(self._inner, name)

    def _record(
        self, operation: str, type_name: str, byte_size: int, elapsed_secs: float
    ) -> None:
        try:
            span = self._tracer.start_span(f"checkpoint:serde:{operation}")
            span.set_attribute("serde.operation", operation)
            span.set_attribute("serde.type", type_name)
            span.set_attribute("serde.byte_size", byte_size)
            span.set_attribute("serde.duration_ms", elapsed_secs * 1000)
            span.end(SpanStatus.OK)
        except Exception:
            logger.debug(
                "agent-trace: failed to record serde %s call", operation, exc_info=True
            )


# ---------------------------------------------------------------------------
# TracingCheckpointSaver — checkpointer write instrumentation
# ---------------------------------------------------------------------------


def _make_checkpoint_saver_base() -> type:
    """Return langgraph.checkpoint.base.BaseCheckpointSaver (lazy import).

    TracingCheckpointSaver must genuinely subclass BaseCheckpointSaver:
    LangGraph's own Pregel._defaults() gates checkpoint behavior on
    isinstance(checkpointer, BaseCheckpointSaver) (confirmed via direct
    inspection of the installed langgraph.pregel.main), so a duck-typed
    wrapper that merely implements the same methods without the real base
    class would be silently treated as "no checkpointer" by LangGraph.
    """
    base = _require_checkpoint_base()
    return base.BaseCheckpointSaver  # type: ignore[no-any-return]


def _build_tracing_checkpoint_saver_class() -> type:
    base_cls = _make_checkpoint_saver_base()

    class _TracingCheckpointSaverImpl(base_cls):  # type: ignore[misc, valid-type]
        """Concrete implementation — see TracingCheckpointSaver for public docs."""

        def __init__(self, inner: Any, tracer: Tracer) -> None:
            # Deliberately skip BaseCheckpointSaver.__init__ (it would
            # overwrite self.serde with maybe_add_typed_methods(None or
            # class-level JsonPlusSerializer()) rather than the wrapped
            # checkpointer's own serde).
            self._inner = inner
            self._tracer = tracer
            # Wrap the *inner* checkpointer's own serde in place so
            # serde-boundary calls made internally by inner.put()/.get()
            # (not just calls this wrapper happens to make directly) are
            # captured too — this is what actually closes the "checkpointer
            # serde-boundary capture" gap, not just a wrapper method here.
            if not isinstance(inner.serde, TracingSerde):
                inner.serde = TracingSerde(inner.serde, tracer)
            self.serde = inner.serde

        # -- delegated read/admin surface (BaseCheckpointSaver defines all of
        # these as real methods that raise NotImplementedError by default, so
        # plain attribute lookup would find the base class's version rather
        # than falling through to __getattr__ — each has to be delegated
        # explicitly). ------------------------------------------------------

        @property
        def config_specs(self) -> list[Any]:
            return self._inner.config_specs  # type: ignore[no-any-return]

        def get(self, config: Any) -> Any:
            return self._inner.get(config)

        def get_tuple(self, config: Any) -> Any:
            return self._inner.get_tuple(config)

        def list(self, config: Any, **kwargs: Any) -> Any:
            return self._inner.list(config, **kwargs)

        def delete_thread(self, thread_id: str) -> None:
            self._inner.delete_thread(thread_id)

        def delete_for_runs(self, run_ids: Any) -> None:
            self._inner.delete_for_runs(run_ids)

        def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
            self._inner.copy_thread(source_thread_id, target_thread_id)

        def prune(self, thread_ids: Any, **kwargs: Any) -> None:
            self._inner.prune(thread_ids, **kwargs)

        def get_next_version(self, current: Any, channel: Any = None) -> Any:
            return self._inner.get_next_version(current, channel)

        async def aget(self, config: Any) -> Any:
            return await self._inner.aget(config)

        async def aget_tuple(self, config: Any) -> Any:
            return await self._inner.aget_tuple(config)

        async def alist(self, config: Any, **kwargs: Any) -> Any:
            async for item in self._inner.alist(config, **kwargs):
                yield item

        async def adelete_thread(self, thread_id: str) -> None:
            await self._inner.adelete_thread(thread_id)

        async def adelete_for_runs(self, run_ids: Any) -> None:
            await self._inner.adelete_for_runs(run_ids)

        async def acopy_thread(
            self, source_thread_id: str, target_thread_id: str
        ) -> None:
            await self._inner.acopy_thread(source_thread_id, target_thread_id)

        async def aprune(self, thread_ids: Any, **kwargs: Any) -> None:
            await self._inner.aprune(thread_ids, **kwargs)

        # -- instrumented write surface --------------------------------------

        def put(
            self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any
        ) -> Any:
            return _record_sync_write(
                self._tracer,
                "checkpoint:put",
                config,
                checkpoint,
                lambda: self._inner.put(config, checkpoint, metadata, new_versions),
            )

        async def aput(
            self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any
        ) -> Any:
            return await _record_async_write(
                self._tracer,
                "checkpoint:aput",
                config,
                checkpoint,
                lambda: self._inner.aput(config, checkpoint, metadata, new_versions),
            )

        def put_writes(
            self, config: Any, writes: Any, task_id: str, task_path: str = ""
        ) -> None:
            _record_sync_write(
                self._tracer,
                "checkpoint:put_writes",
                config,
                writes,
                lambda: self._inner.put_writes(config, writes, task_id, task_path),
                task_id=task_id,
                channels=writes,
            )

        async def aput_writes(
            self, config: Any, writes: Any, task_id: str, task_path: str = ""
        ) -> None:
            await _record_async_write(
                self._tracer,
                "checkpoint:aput_writes",
                config,
                writes,
                lambda: self._inner.aput_writes(config, writes, task_id, task_path),
                task_id=task_id,
                channels=writes,
            )

    return _TracingCheckpointSaverImpl


def _channel_names(writes: Any) -> list[str] | None:
    """Best-effort: [channel for (channel, value) in writes] for a
    put_writes()-shaped `writes` argument, else None."""
    try:
        return [str(w[0]) for w in writes]
    except Exception:
        return None


def _annotate_write_span(
    span: Span,
    config: Any,
    payload: Any,
    *,
    task_id: str | None,
    channels: Any | None,
) -> None:
    thread_id = _extract_thread_id(config)
    if thread_id is not None:
        span.set_attribute("checkpoint.thread_id", thread_id)
    if task_id is not None:
        span.set_attribute("checkpoint.task_id", str(task_id))
    if channels is not None:
        names = _channel_names(channels)
        if names:
            span.set_attribute("checkpoint.channels", ",".join(names))
    size = _estimate_size(payload)
    if size is not None:
        span.set_attribute("checkpoint.payload_size_bytes", size)


def _record_sync_write(
    tracer: Tracer,
    span_name: str,
    config: Any,
    payload: Any,
    fn: Any,
    *,
    task_id: str | None = None,
    channels: Any | None = None,
) -> Any:
    span = tracer.start_span(span_name)
    _annotate_write_span(span, config, payload, task_id=task_id, channels=channels)
    try:
        result = fn()
    except BaseException as exc:
        span.set_attribute("checkpoint.completed", False)
        _close_write_span_on_error(span, exc)
        raise
    span.set_attribute("checkpoint.completed", True)
    span.end(SpanStatus.OK)
    return result


async def _record_async_write(
    tracer: Tracer,
    span_name: str,
    config: Any,
    payload: Any,
    fn: Any,
    *,
    task_id: str | None = None,
    channels: Any | None = None,
) -> Any:
    span = tracer.start_span(span_name)
    _annotate_write_span(span, config, payload, task_id=task_id, channels=channels)
    try:
        result = await fn()
    except BaseException as exc:
        span.set_attribute("checkpoint.completed", False)
        _close_write_span_on_error(span, exc)
        raise
    span.set_attribute("checkpoint.completed", True)
    span.end(SpanStatus.OK)
    return result


def _close_write_span_on_error(span: Span, exc: BaseException) -> None:
    """Distinguish a cancelled write (cut off mid-flight — the exact
    cancellation-triggered data-loss shape behind issue #5672) from a
    genuine write failure, mirroring _close_span_with_exception's
    CANCELLED-vs-ERROR split in the callback integration."""
    if isinstance(exc, asyncio.CancelledError):
        span.record_exception(exc, status=SpanStatus.CANCELLED)
        span.end(SpanStatus.CANCELLED)
    else:
        span.record_exception(exc)
        span.end(SpanStatus.ERROR)


_TracingCheckpointSaverClass: type | None = None


def TracingCheckpointSaver(inner: Any, tracer: Tracer) -> Any:  # noqa: N802
    """Wrap *inner* (any real ``BaseCheckpointSaver`` instance) so every
    ``put``/``aput``/``put_writes``/``aput_writes`` call records a span
    (timestamp via span start/end, payload size, whether the call actually
    completed) while every other method — reads, admin operations, the
    serde boundary — is transparently delegated (with the serde boundary
    itself also instrumented; see ``TracingSerde``).

    Drop-in replacement for the real checkpointer::

        real = InMemorySaver()
        traced = TracingCheckpointSaver(real, tracer)
        graph = builder.compile(checkpointer=traced)

    Returns a genuine ``BaseCheckpointSaver`` subclass instance (required —
    LangGraph's own ``Pregel._defaults()`` gates checkpoint behavior on
    ``isinstance(checkpointer, BaseCheckpointSaver)``), built lazily so
    importing this module never requires langgraph to be installed.
    """
    global _TracingCheckpointSaverClass  # noqa: PLW0603
    if _TracingCheckpointSaverClass is None:
        _TracingCheckpointSaverClass = _build_tracing_checkpoint_saver_class()
    return _TracingCheckpointSaverClass(inner, tracer)


# ---------------------------------------------------------------------------
# traced_update_state / traced_aupdate_state — as_node + task-scheduling
# ---------------------------------------------------------------------------


def _record_post_update_schedule(
    span: Span, graph: Any, new_config: Any, requested_as_node: str | None
) -> None:
    """Best-effort: read the post-write task schedule via the public
    get_state()/.next surface and flag an empty schedule.

    Wrapped entirely in try/except: a shape mismatch here (e.g. a future
    LangGraph version renaming StateSnapshot.next) must degrade to "no
    schedule captured", never break the update_state() call it's
    piggybacking on.
    """
    if requested_as_node is not None:
        span.set_attribute("checkpoint.as_node", str(requested_as_node))
        span.set_attribute("checkpoint.as_node_provided", True)
    else:
        # LangGraph infers as_node internally (e.g. "the last node that
        # updated the state") when the caller doesn't supply one — that
        # inferred value isn't observable from the public update_state()/
        # get_state() surface, so this is honestly reported as "not
        # provided" rather than guessed at.
        span.set_attribute("checkpoint.as_node_provided", False)
    try:
        snapshot = graph.get_state(new_config)
        next_tasks = tuple(getattr(snapshot, "next", None) or ())
        span.set_attribute("checkpoint.next_task_count", len(next_tasks))
        if next_tasks:
            span.set_attribute("checkpoint.next_tasks", ",".join(next_tasks))
        span.set_attribute("checkpoint.zero_tasks_scheduled", len(next_tasks) == 0)
    except Exception:
        logger.debug(
            "agent-trace: failed to read post-update-state task schedule",
            exc_info=True,
        )


def traced_update_state(
    tracer: Tracer,
    graph: Any,
    config: Any,
    values: Any,
    as_node: str | None = None,
    task_id: str | None = None,
) -> Any:
    """Wrap ``graph.update_state(...)``, recording the ``as_node`` the write
    was attributed to (when the caller supplied one explicitly) and the
    pregel scheduler's resulting ``next`` task list immediately afterward —
    flagging ``checkpoint.zero_tasks_scheduled=True`` when that list comes
    back empty, the exact silent-no-op-resume shape behind issue #4217.

    Usage::

        new_config = traced_update_state(tracer, graph, config, values,
                                          as_node="my_node")
    """
    span = tracer.start_span("checkpoint:update_state")
    try:
        new_config = graph.update_state(
            config, values, as_node=as_node, task_id=task_id
        )
    except BaseException as exc:
        span.record_exception(exc)
        span.end(
            SpanStatus.CANCELLED
            if isinstance(exc, asyncio.CancelledError)
            else SpanStatus.ERROR
        )
        raise
    _record_post_update_schedule(span, graph, new_config, as_node)
    span.end(SpanStatus.OK)
    return new_config


async def traced_aupdate_state(
    tracer: Tracer,
    graph: Any,
    config: Any,
    values: Any,
    as_node: str | None = None,
    task_id: str | None = None,
) -> Any:
    """Async equivalent of :func:`traced_update_state`."""
    span = tracer.start_span("checkpoint:update_state")
    try:
        new_config = await graph.aupdate_state(
            config, values, as_node=as_node, task_id=task_id
        )
    except BaseException as exc:
        span.record_exception(exc)
        span.end(
            SpanStatus.CANCELLED
            if isinstance(exc, asyncio.CancelledError)
            else SpanStatus.ERROR
        )
        raise
    try:
        snapshot = await graph.aget_state(new_config)
    except Exception:
        snapshot = None
    _record_post_update_schedule_async(span, snapshot, as_node)
    span.end(SpanStatus.OK)
    return new_config


def _record_post_update_schedule_async(
    span: Span, snapshot: Any, requested_as_node: str | None
) -> None:
    """Same bookkeeping as _record_post_update_schedule, but for the async
    path where the snapshot was already fetched via graph.aget_state()."""
    if requested_as_node is not None:
        span.set_attribute("checkpoint.as_node", str(requested_as_node))
        span.set_attribute("checkpoint.as_node_provided", True)
    else:
        span.set_attribute("checkpoint.as_node_provided", False)
    try:
        next_tasks = tuple(getattr(snapshot, "next", None) or ())
        span.set_attribute("checkpoint.next_task_count", len(next_tasks))
        if next_tasks:
            span.set_attribute("checkpoint.next_tasks", ",".join(next_tasks))
        span.set_attribute("checkpoint.zero_tasks_scheduled", len(next_tasks) == 0)
    except Exception:
        logger.debug(
            "agent-trace: failed to read post-update-state task schedule (async)",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# TracingCache / wrap_cache_policy — CachePolicy hit/miss + key input capture
# ---------------------------------------------------------------------------


def _format_full_keys(keys: Any) -> str:
    """Render a sequence of BaseCache FullKey = (Namespace, str) tuples as a
    compact, human-readable string."""
    try:
        return ",".join(f"{'/'.join(ns)}:{key}" for ns, key in keys)
    except Exception:
        return _stringify(keys, max_len=500)


def _build_tracing_cache_class() -> type:
    base = _require_cache_base()
    base_cls = base.BaseCache

    class _TracingCacheImpl(base_cls):  # type: ignore[misc, valid-type]
        """Concrete implementation — see TracingCache for public docs."""

        def __init__(self, inner: Any, tracer: Tracer) -> None:
            self._inner = inner
            self._tracer = tracer
            self.serde = inner.serde

        def get(self, keys: Any) -> Any:
            result = self._inner.get(keys)
            self._record_get(keys, result)
            return result

        async def aget(self, keys: Any) -> Any:
            result = await self._inner.aget(keys)
            self._record_get(keys, result)
            return result

        def set(self, pairs: Any) -> None:
            self._record_set(pairs)
            self._inner.set(pairs)

        async def aset(self, pairs: Any) -> None:
            self._record_set(pairs)
            await self._inner.aset(pairs)

        def clear(self, namespaces: Any = None) -> None:
            self._inner.clear(namespaces)

        async def aclear(self, namespaces: Any = None) -> None:
            await self._inner.aclear(namespaces)

        def _record_get(self, keys: Any, result: dict[Any, Any]) -> None:
            try:
                hits = [k for k in keys if k in result]
                misses = [k for k in keys if k not in result]
                span = self._tracer.start_span("cache:get")
                span.set_attribute("cache.hit_count", len(hits))
                span.set_attribute("cache.miss_count", len(misses))
                if hits:
                    span.set_attribute("cache.hit_keys", _format_full_keys(hits))
                if misses:
                    span.set_attribute("cache.miss_keys", _format_full_keys(misses))
                span.end(SpanStatus.OK)
            except Exception:
                logger.debug(
                    "agent-trace: failed to record cache get() call", exc_info=True
                )

        def _record_set(self, pairs: Any) -> None:
            try:
                keys = list(pairs.keys())
                span = self._tracer.start_span("cache:set")
                span.set_attribute("cache.set_count", len(keys))
                if keys:
                    span.set_attribute("cache.set_keys", _format_full_keys(keys))
                span.end(SpanStatus.OK)
            except Exception:
                logger.debug(
                    "agent-trace: failed to record cache set() call", exc_info=True
                )

    return _TracingCacheImpl


_TracingCacheClass: type | None = None


def TracingCache(inner: Any, tracer: Tracer) -> Any:  # noqa: N802
    """Wrap *inner* (any real ``BaseCache`` instance) so every ``get``/
    ``aget`` call records a hit/miss-count span and every ``set``/``aset``
    call records a span naming the keys written — the cache hit/miss half of
    the "Hook LangGraph CachePolicy cache hit/miss decisions" gap. Combine
    with :func:`wrap_cache_policy` to also capture the state object/bytes
    LangGraph hashed to compute the cache key.

    Usage::

        graph = builder.compile(cache=TracingCache(InMemoryCache(), tracer))

    Returns a genuine ``BaseCache`` subclass instance, built lazily so
    importing this module never requires langgraph to be installed.
    """
    global _TracingCacheClass  # noqa: PLW0603
    if _TracingCacheClass is None:
        _TracingCacheClass = _build_tracing_cache_class()
    return _TracingCacheClass(inner, tracer)


def wrap_cache_policy(policy: Any, tracer: Tracer) -> Any:
    """Return a copy of *policy* (a ``langgraph.types.CachePolicy``) whose
    ``key_func`` records the state object it was actually called with — the
    real input LangGraph hashes to compute a node's cache key — alongside
    the resulting key, before delegating to the real ``key_func``.

    ``BaseCache.get()``/``.set()`` (see ``TracingCache`` above) only ever
    see the *already-computed* key; the state object that produced it is
    only observable at the ``key_func`` call site itself, which is exactly
    what this wraps (a copy of the CachePolicy dataclass — a public,
    documented field — not a private pregel internal).

    Usage::

        builder.add_node("my_node", my_fn,
                          cache_policy=wrap_cache_policy(CachePolicy(), tracer))

    Returns *policy* unchanged if it is None (mirrors add_node's own
    cache_policy=None default meaning "no caching").
    """
    if policy is None:
        return None
    import dataclasses

    original_key_func = policy.key_func

    def _traced_key_func(*args: Any, **kwargs: Any) -> Any:
        key = original_key_func(*args, **kwargs)
        try:
            span = tracer.start_span("cache:key_func")
            hashed_input = args[0] if args else kwargs
            span.set_attribute("cache.key_input", _to_attr_string(hashed_input))
            span.set_attribute(
                "cache.key",
                key if isinstance(key, str) else _stringify(key, max_len=500),
            )
            span.end(SpanStatus.OK)
        except Exception:
            logger.debug(
                "agent-trace: failed to record cache key_func call", exc_info=True
            )
        return key

    return dataclasses.replace(policy, key_func=_traced_key_func)
