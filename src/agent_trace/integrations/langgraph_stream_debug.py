"""
Optional deep instrumentation of LangGraph's internal
``langgraph.pregel._messages.StreamMessagesHandler`` — the private class
that actually implements ``stream_mode="messages"``.

Background
----------

``StreamMessagesHandler`` decides, per chat-model call, whether that call's
tokens get pushed into a ``stream_mode="messages"`` iterator at all —
governed entirely by whether ``TAG_NOSTREAM`` ("nostream") is present in the
*runtime* ``tags`` list ``on_chat_model_start`` receives for that call (see
``StreamMessagesHandler.on_chat_model_start``: ``if metadata and (not tags
or (TAG_NOSTREAM not in tags))``). This is a LangGraph-internal mechanism,
not reachable via any public configuration surface — a developer who
declares ``tags=["nostream"]`` on a node's own action (see
``agent_trace.integrations.langgraph._get_declared_node_tags``) is trusting
that the tag actually survives, unmodified, all the way to that runtime
check. When it doesn't (a tag-propagation bug elsewhere in
LangChain/LangGraph strips or fails to merge it before the callback manager
fires), the symptom is silent: content the developer explicitly tried to
suppress from the stream shows up anyway (issue #7509).

This module patches ``StreamMessagesHandler.on_chat_model_start`` to record,
for every chat-model call it processes, the node name and the exact
suppress/allow decision it made — the identical signal
``StreamMessagesHandler`` itself acts on — so that decision becomes
inspectable instead of buried inside a private class with no logging and no
public hook. Cross-referenced against a node's *declared* tags (captured
separately via ``LangGraphTracer``'s ``langgraph.declared_tags`` span
attribute, when a ``graph=`` was supplied), :func:`flag_inconsistencies`
turns "declared nostream but LangGraph still decided to stream it" into an
automated flag rather than something a developer discovers by reading
private LangGraph source and comparing it against their own graph
definition by hand.

Usage::

    from agent_trace.integrations.langgraph_stream_debug import (
        install_stream_debug_patch,
        get_stream_decisions,
        flag_inconsistencies,
    )

    # Call once, before any graph.stream(..., stream_mode="messages") run:
    install_stream_debug_patch()
    ...
    decisions = get_stream_decisions()
    flagged = flag_inconsistencies(decisions, declared_nostream_nodes={"my_node"})

Best-effort by design, matching the monkeypatch pattern already used in
``agent_trace.integrations.langgraph`` (``_install_runtime_capture_patch``,
``_install_branch_exception_capture_patch``): if this LangGraph version's
private internals don't match what this patch expects,
``install_stream_debug_patch()`` logs at DEBUG and returns False rather than
raising — activating this instrumentation must never be able to break a
real graph run.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "StreamDecision",
    "flag_inconsistencies",
    "get_stream_decisions",
    "install_stream_debug_patch",
    "reset_stream_decisions",
]

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_installed = False
_decisions: list[StreamDecision] = []
_MAX_DECISIONS = 5_000  # bounded, matching the defensive-cap pattern elsewhere


@dataclass(frozen=True)
class StreamDecision:
    """One ``StreamMessagesHandler.on_chat_model_start`` suppress/allow
    decision, as it actually happened at runtime."""

    node_name: str | None
    run_id: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    suppressed: bool = False
    """True if TAG_NOSTREAM was present in the runtime tags list — i.e.
    StreamMessagesHandler decided NOT to register this call for streaming,
    so none of its tokens will reach a stream_mode="messages" consumer."""


def get_stream_decisions() -> list[StreamDecision]:
    """Return every recorded streaming suppress/allow decision so far."""
    with _lock:
        return list(_decisions)


def reset_stream_decisions() -> None:
    """Clear recorded decisions — call between separate runs/tests sharing
    one process so results from an earlier run don't bleed into the next."""
    with _lock:
        _decisions.clear()


def flag_inconsistencies(
    decisions: list[StreamDecision],
    declared_nostream_nodes: set[str],
) -> list[StreamDecision]:
    """Given recorded per-call streaming decisions and the set of node names
    known (from a separate source — e.g. ``LangGraphTracer``'s
    ``langgraph.declared_tags`` span attribute) to have declared a
    ``"nostream"`` tag at graph-construction time, return every decision
    where that declared intent was NOT honored — the node declared
    ``nostream`` but ``StreamMessagesHandler`` still decided to stream that
    call's tokens (``suppressed=False``). This is the exact tag/stream
    inconsistency behind issue #7509."""
    return [
        d
        for d in decisions
        if d.node_name in declared_nostream_nodes and not d.suppressed
    ]


def install_stream_debug_patch() -> bool:
    """Monkeypatch ``StreamMessagesHandler.on_chat_model_start`` to record
    every suppress/allow decision it makes. Idempotent — safe to call more
    than once; only the first call actually installs anything.

    Returns True if the patch installed (this LangGraph version's private
    internals matched what was expected), False if it didn't and this
    instrumentation is a no-op on the installed version.
    """
    global _installed  # noqa: PLW0603
    if _installed:
        return True
    with _lock:
        if _installed:
            return True
        try:
            from langgraph.constants import TAG_NOSTREAM
            from langgraph.pregel._messages import StreamMessagesHandler
        except Exception:
            logger.debug(
                "agent-trace: StreamMessagesHandler deep-instrumentation "
                "unavailable on this LangGraph version (private module "
                "shape not as expected); stream/tag inconsistency "
                "detection will not be available.",
                exc_info=True,
            )
            _installed = True  # don't retry every call
            return False

        original_on_chat_model_start = StreamMessagesHandler.on_chat_model_start

        def _patched_on_chat_model_start(
            self: Any,
            serialized: dict[str, Any],
            messages: Any,
            *,
            run_id: Any,
            parent_run_id: Any = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            try:
                node_name = (metadata or {}).get("langgraph_node")
                decision = StreamDecision(
                    node_name=str(node_name) if node_name is not None else None,
                    run_id=str(run_id),
                    tags=tuple(tags or ()),
                    suppressed=bool(tags and TAG_NOSTREAM in tags),
                )
                with _lock:
                    if len(_decisions) < _MAX_DECISIONS:
                        _decisions.append(decision)
            except Exception:
                logger.debug(
                    "agent-trace: failed to record stream suppress/allow "
                    "decision for run %r",
                    run_id,
                    exc_info=True,
                )
            return original_on_chat_model_start(
                self,
                serialized,
                messages,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )

        StreamMessagesHandler.on_chat_model_start = _patched_on_chat_model_start
        _installed = True
        return True
