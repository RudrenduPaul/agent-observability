"""
Per-superstep state-merge diagnostics for parallel Command(graph=PARENT)
routing.

Wraps any ``BaseCheckpointSaver`` so that when N parallel tasks in the same
Pregel superstep each propose a write to the same channel — the exact shape
behind issue #7129, where 2 of 3 parallel tool calls each returning
``Command(graph=Command.PARENT, update={...})`` had their update silently
discarded by LangGraph's own parent-graph merge — the drop shows up as an
explicit, countable fact on a span instead of leaving the developer to
already suspect LangGraph's internal Pregel/``Command.PARENT`` semantics
before they can even look for it.

Why the checkpointer, not a BaseCallbackHandler hook: the actual merge
happens inside Pregel's own scheduler, a layer no ``on_chain_*``/``on_llm_*``/
``on_tool_*`` callback (what ``LangGraphTracer`` implements) can ever
observe — confirmed via reading ``langgraph/graph/state.py``
(``_control_branch``) and ``langgraph/errors.py`` (``ParentCommand``), which
show the parent-graph update is delivered via an internal control-flow
exception raised per task, not via any traced Runnable. The checkpointer is
the one place *every* superstep's proposed writes (``put_writes``, one call
per task) and finalized state (``put``, once per superstep) both pass
through, regardless of which internal mechanism produced them.

Usage::

    from agent_trace import tracer
    from agent_trace.integrations.langgraph_state_diff import wrap_checkpointer
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = wrap_checkpointer(InMemorySaver(), tracer=tracer)
    graph = builder.compile(checkpointer=checkpointer)

    with tracer.start_trace("my-graph") as trace:
        graph.invoke(input, config={"configurable": {"thread_id": "t1"}})
    # trace now carries a "checkpoint:superstep_merge" span (if any superstep
    # had >1 task propose a write to the same channel) with a
    # "superstep_state_merge" event per affected channel.

Heuristic, not a full re-implementation of Pregel's channel-merge algorithm:
after a superstep's checkpoint is persisted, a proposed value counts as
"survived" if it equals the persisted channel value outright, or if the
persisted value is a list/tuple containing it (covers reducer channels like
``add_messages`` that append rather than overwrite). Anything else is
reported as dropped. False negatives are possible for exotic custom
reducers — this module's job is to turn "developer has to already suspect
this" into "here's a countable fact to check", not to replace reading the
persisted state yourself for a novel reducer.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import SpanStatus

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from agent_trace import Tracer

__all__ = ["wrap_checkpointer"]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "wrap_checkpointer() requires langgraph.\nInstall it with:\n\n"
    "    pip install langgraph\n"
)

_SUPERSTEP_MERGE_SPAN_NAME = "checkpoint:superstep_merge"


def _superstep_key(config: Any) -> tuple[Any, Any, Any] | None:
    """(thread_id, checkpoint_ns, checkpoint_id) — the same key
    ``BaseCheckpointSaver.put_writes``/``put`` implementations use to scope
    a superstep's proposed writes (confirmed against the real
    ``InMemorySaver.put_writes``/``put`` implementations). Returns None if
    ``config`` doesn't carry a ``thread_id`` (e.g. a checkpointer used
    outside a real Pregel run)."""
    conf = (config or {}).get("configurable") or {}
    thread_id = conf.get("thread_id")
    if thread_id is None:
        return None
    checkpoint_ns = conf.get("checkpoint_ns", "")
    checkpoint_id = conf.get("checkpoint_id")
    return (thread_id, checkpoint_ns, checkpoint_id)


def _safe_eq(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def _value_survived(proposed: Any, final: Any) -> bool:
    """True if *proposed* is reflected in the persisted *final* channel
    value — either directly (last-value-wins channel) or as a member of a
    list/tuple (an append/reducer-style channel)."""
    if _safe_eq(proposed, final):
        return True
    if isinstance(final, (list, tuple)):
        return any(_safe_eq(proposed, item) for item in final)
    return False


class _TracingCheckpointSaverMixin:
    """Shared diagnostic logic for the sync and async wrapped methods.

    Not usable on its own — combined with a real ``BaseCheckpointSaver``
    subclass by :func:`wrap_checkpointer`, which builds the concrete class
    lazily once ``BaseCheckpointSaver`` is importable (mirrors the
    ``_get_tracer_class`` pattern in ``integrations/langgraph.py``).
    """

    def _diagnostic_init(self, inner: Any, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer
        self._pending: dict[tuple[Any, Any, Any], list[tuple[str, str, Any]]] = {}
        self._pending_lock = threading.Lock()
        # Forward every method this wrapper doesn't itself override straight
        # to the wrapped saver, bound to its own instance — avoids having to
        # hand-reimplement BaseCheckpointSaver's full surface area (get,
        # list, delete_thread, copy_thread, prune, ... and their async
        # counterparts) just to pass isinstance(checkpointer,
        # BaseCheckpointSaver) checks.
        for attr_name in (
            "get",
            "aget",
            "get_tuple",
            "aget_tuple",
            "list",
            "alist",
            "delete_thread",
            "adelete_thread",
            "delete_for_runs",
            "adelete_for_runs",
            "copy_thread",
            "acopy_thread",
            "prune",
            "aprune",
            "get_next_version",
            "config_specs",
        ):
            value = getattr(inner, attr_name, None)
            if value is None:
                continue
            try:
                object.__setattr__(self, attr_name, value)
            except AttributeError:
                # A handful of BaseCheckpointSaver members (e.g.
                # config_specs) are read-only @property descriptors on the
                # class — they can't be shadowed by an instance attribute.
                # Falling back to the class's own (default) behavior for
                # those is fine; only put/put_writes/aput/aput_writes need
                # this wrapper's own logic.
                logger.debug(
                    "agent-trace: could not forward checkpointer.%s "
                    "(read-only property) — using the wrapper's own "
                    "default implementation instead",
                    attr_name,
                )

    def _record_proposed_writes(
        self, config: Any, writes: Any, task_id: str
    ) -> None:
        key = _superstep_key(config)
        if key is None:
            return
        with self._pending_lock:
            bucket = self._pending.setdefault(key, [])
            for channel, value in writes:
                bucket.append((task_id, channel, value))

    def _diagnose_superstep(self, config: Any, checkpoint: Any) -> None:
        """Called after the wrapped saver's put() persists a superstep's
        checkpoint. Diffs every channel that had >1 distinct task propose a
        write against the persisted value, emitting a span event for any
        channel where at least one proposal didn't survive."""
        conf = (config or {}).get("configurable") or {}
        thread_id = conf.get("thread_id")
        if thread_id is None:
            return
        # The incoming `config`'s checkpoint_id is the *parent* checkpoint —
        # the same id put_writes() calls for this superstep were scoped
        # under (confirmed against InMemorySaver.put/put_writes: both key
        # off config["configurable"]["checkpoint_id"], and put()'s config
        # argument is the pre-finalization config carrying the parent id).
        key = (thread_id, conf.get("checkpoint_ns", ""), conf.get("checkpoint_id"))
        with self._pending_lock:
            proposals = self._pending.pop(key, None)
        if not proposals:
            return

        channel_values = (checkpoint or {}).get("channel_values") or {}
        by_channel: dict[str, list[tuple[str, Any]]] = {}
        for task_id, channel, value in proposals:
            by_channel.setdefault(channel, []).append((task_id, value))

        affected: list[dict[str, Any]] = []
        for channel, entries in by_channel.items():
            distinct_tasks = {task_id for task_id, _ in entries}
            if len(distinct_tasks) < 2:
                continue  # only one task proposed this channel — no merge to diagnose
            final_value = channel_values.get(channel)
            dropped_tasks = [
                task_id
                for task_id, value in entries
                if not _value_survived(value, final_value)
            ]
            if dropped_tasks:
                affected.append(
                    {
                        "channel": channel,
                        "proposed_count": len(entries),
                        "survived_count": len(entries) - len(dropped_tasks),
                        "dropped_count": len(dropped_tasks),
                        "dropped_task_ids": dropped_tasks,
                    }
                )

        if not affected:
            return

        try:
            span = self._tracer.start_span(_SUPERSTEP_MERGE_SPAN_NAME)
            span.set_attribute("checkpoint.thread_id", str(thread_id))
            for fact in affected:
                span.add_event(
                    "superstep_state_merge",
                    attributes={
                        "channel": str(fact["channel"]),
                        "proposed_count": int(fact["proposed_count"]),
                        "survived_count": int(fact["survived_count"]),
                        "dropped_count": int(fact["dropped_count"]),
                        "dropped_task_ids": ",".join(fact["dropped_task_ids"]),
                    },
                )
            span.end(SpanStatus.OK)
        except Exception:
            logger.debug(
                "agent-trace: failed to record superstep state-merge "
                "diagnostic",
                exc_info=True,
            )

    # -- sync -----------------------------------------------------------

    def put(
        self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any
    ) -> Any:
        result = self._inner.put(config, checkpoint, metadata, new_versions)
        try:
            self._diagnose_superstep(config, checkpoint)
        except Exception:
            logger.debug(
                "agent-trace: superstep diagnostic failed in put()", exc_info=True
            )
        return result

    def put_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        self._inner.put_writes(config, writes, task_id, task_path)
        try:
            self._record_proposed_writes(config, writes, task_id)
        except Exception:
            logger.debug(
                "agent-trace: failed to record proposed superstep writes",
                exc_info=True,
            )

    # -- async ------------------------------------------------------------

    async def aput(
        self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any
    ) -> Any:
        result = await self._inner.aput(config, checkpoint, metadata, new_versions)
        try:
            self._diagnose_superstep(config, checkpoint)
        except Exception:
            logger.debug(
                "agent-trace: superstep diagnostic failed in aput()", exc_info=True
            )
        return result

    async def aput_writes(
        self, config: Any, writes: Any, task_id: str, task_path: str = ""
    ) -> None:
        await self._inner.aput_writes(config, writes, task_id, task_path)
        try:
            self._record_proposed_writes(config, writes, task_id)
        except Exception:
            logger.debug(
                "agent-trace: failed to record proposed superstep writes",
                exc_info=True,
            )


_TracingCheckpointSaverClass: type | None = None
_class_lock = threading.Lock()


def _get_wrapper_class() -> type:
    """Lazily build the concrete wrapper class, with ``BaseCheckpointSaver``
    as a genuine base at definition time (LangGraph's ``ensure_valid_checkpointer``
    requires ``isinstance(checkpointer, BaseCheckpointSaver)``)."""
    global _TracingCheckpointSaverClass  # noqa: PLW0603
    if _TracingCheckpointSaverClass is not None:
        return _TracingCheckpointSaverClass

    with _class_lock:
        if _TracingCheckpointSaverClass is not None:
            return _TracingCheckpointSaverClass

        try:
            from langgraph.checkpoint.base import BaseCheckpointSaver
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        class _TracingCheckpointSaver(
            _TracingCheckpointSaverMixin, BaseCheckpointSaver[Any]
        ):
            def __init__(self, inner: Any, tracer: Tracer) -> None:
                super().__init__(serde=getattr(inner, "serde", None))
                self._diagnostic_init(inner, tracer)

        _TracingCheckpointSaverClass = _TracingCheckpointSaver
        return _TracingCheckpointSaverClass


def wrap_checkpointer(checkpointer: BaseCheckpointSaver[Any], *, tracer: Tracer) -> Any:
    """Wrap *checkpointer* so agent-trace records per-superstep state-merge
    diagnostics onto *tracer*'s active trace.

    The returned object is itself a ``BaseCheckpointSaver`` (satisfies
    LangGraph's ``ensure_valid_checkpointer`` isinstance check) that forwards
    every call to *checkpointer* unchanged — recording is purely additive
    and never alters what gets persisted or returned.
    """
    wrapper_cls = _get_wrapper_class()
    return wrapper_cls(checkpointer, tracer)
