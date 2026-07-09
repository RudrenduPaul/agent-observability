"""
LangGraph integration — callback handler and graph-aware span enrichment.

Instruments a LangGraph StateGraph to emit spans for each node execution,
edge traversal, and LLM call within the graph.

Usage:
    from agent_trace.integrations.langgraph import LangGraphTracer
    from agent_trace import tracer

    with tracer.start_trace("my_graph", record=True) as trace:
        result = graph.invoke(
            input_state,
            config={"callbacks": [LangGraphTracer(tracer=tracer, trace=trace)]}
        )
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import threading
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Generator, Iterable
from typing import TYPE_CHECKING, Any

from agent_trace.core.clock import get_time
from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = [
    "LangGraphTracer",
    "derive_trace_id",
    "find_tool_params_shaped_like_state",
    "instrument_graph_factory",
    "traced_astream",
    "traced_stream",
]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LangGraphTracer requires langchain-core and langgraph.\n"
    "Install them with:\n\n"
    "    pip install langchain-core langgraph\n"
)

# Span attributes only accept str/int/float/bool (see agent_trace.core.span).
# Anything structured (BaseMessage, dicts full of BaseMessages, dataclasses,
# etc.) has to be flattened to a bounded string before it can be stored.
_MAX_ATTR_LEN = 8_000

# _deep_serialize() bounds — deliberately small. This is span-attribute
# summary data, not a full-fidelity dump, and the bound also protects against
# pathological/self-referential objects (e.g. an unconfigured
# unittest.mock.MagicMock, whose attribute access and method calls always
# yield a brand-new child Mock) that would otherwise recurse indefinitely.
_MAX_SERIALIZE_DEPTH = 6
_MAX_COLLECTION_ITEMS = 200

# Cap on the number of llm_stream_delta SpanEvents recorded per LLM span —
# a long stream (thousands of tokens) would otherwise grow a single span
# unboundedly. llm.stream_token_count keeps counting past this cap; only
# new SpanEvents stop being appended.
_MAX_STREAM_EVENTS_PER_SPAN = 500


def _safe_str(value: Any) -> str:
    """str(value), degrading to a placeholder instead of raising."""
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def _serialize_mapping(
    value: dict[Any, Any], *, _depth: int, _seen: frozenset[int]
) -> dict[str, Any]:
    """_deep_serialize() branch for dict-like values, capped at
    _MAX_COLLECTION_ITEMS keys."""
    out: dict[str, Any] = {}
    items = list(value.items())
    for k, v in items[:_MAX_COLLECTION_ITEMS]:
        out[str(k)] = _deep_serialize(v, _depth=_depth + 1, _seen=_seen)
    if len(items) > _MAX_COLLECTION_ITEMS:
        out["..."] = f"<{len(items) - _MAX_COLLECTION_ITEMS} more items truncated>"
    return out


def _serialize_sequence(
    value: list[Any] | tuple[Any, ...], *, _depth: int, _seen: frozenset[int]
) -> list[Any]:
    """_deep_serialize() branch for list/tuple values, capped at
    _MAX_COLLECTION_ITEMS items."""
    items = list(value)
    out_list = [
        _deep_serialize(item, _depth=_depth + 1, _seen=_seen)
        for item in items[:_MAX_COLLECTION_ITEMS]
    ]
    if len(items) > _MAX_COLLECTION_ITEMS:
        out_list.append(f"<{len(items) - _MAX_COLLECTION_ITEMS} more items truncated>")
    return out_list


def _serialize_container(
    value: dict[Any, Any] | list[Any] | tuple[Any, ...],
    *,
    _depth: int,
    _seen: frozenset[int],
) -> Any:
    """Dispatch a dict/list/tuple to its dedicated serializer.

    Split out purely to keep _deep_serialize's own branch/return count low.
    """
    if isinstance(value, dict):
        return _serialize_mapping(value, _depth=_depth, _seen=_seen)
    return _serialize_sequence(value, _depth=_depth, _seen=_seen)


def _dump_via_attr(value: Any, attr_name: str) -> Any:
    """Call value.<attr_name>() if present and callable, else return None.

    Used for both the pydantic-v2 ``model_dump()`` shape and the pydantic-v1
    / dict-like ``dict()`` shape — a failure or absence of the method is not
    an error, just "this strategy doesn't apply to this object".
    """
    method = getattr(value, attr_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _deep_serialize(
    value: Any,
    *,
    _depth: int = 0,
    _seen: frozenset[int] = frozenset(),
) -> Any:
    """Recursively convert *value* into JSON-primitive-only data.

    Tries, in order, per non-primitive object: a pydantic-v2-style
    ``model_dump()`` (what ``langchain_core.messages.BaseMessage`` exposes),
    a ``dict()`` method (pydantic v1 / other dict-likes), and finally
    ``str()``. Bounded on both depth and collection size (rather than
    delegating recursion to ``json.dumps``'s own ``default`` callback) so a
    pathological or infinitely-self-generating object cannot spin this into
    a multi-thousand-frame stack walk — every recursive branch strictly
    increments ``_depth`` and stops at ``_MAX_SERIALIZE_DEPTH`` regardless of
    what the object's own methods return.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    obj_id = id(value)
    if obj_id in _seen:
        return "<circular-reference>"
    if _depth >= _MAX_SERIALIZE_DEPTH:
        return _safe_str(value)

    seen_here = _seen | {obj_id}

    if isinstance(value, (dict, list, tuple)):
        return _serialize_container(value, _depth=_depth, _seen=seen_here)

    for attr_name in ("model_dump", "dict"):
        dumped = _dump_via_attr(value, attr_name)
        if dumped is not None:
            return _deep_serialize(dumped, _depth=_depth + 1, _seen=seen_here)

    return _safe_str(value)


def _to_attr_string(value: Any, *, max_len: int = _MAX_ATTR_LEN) -> str:
    """Defensively serialize *value* into a bounded, span-safe string.

    Used for anything structured (message lists, chain inputs/outputs,
    metadata dicts, response_metadata, ...) that can't be stored as a raw
    span attribute directly.
    """
    try:
        text = json.dumps(_deep_serialize(value), ensure_ascii=False, default=str)
    except Exception:
        try:
            text = str(value)
        except Exception:
            text = "<unserializable>"
    if len(text) > max_len:
        text = text[:max_len] + "...<truncated>"
    return text


def _stringify(value: Any, *, max_len: int = _MAX_ATTR_LEN) -> str:
    """Turn *value* into a span-safe string.

    Plain strings are truncated as-is (so raw tool I/O isn't wrapped in JSON
    quotes); anything else goes through :func:`_to_attr_string`.
    """
    if isinstance(value, str):
        text = value
    else:
        return _to_attr_string(value, max_len=max_len)
    if len(text) > max_len:
        text = text[:max_len] + "...<truncated>"
    return text


# ---------------------------------------------------------------------------
# trace_id derivation from LangGraph thread_id/checkpoint identity
# ---------------------------------------------------------------------------
#
# Tracer.start_trace() always mints a random trace_id (uuid.uuid4().hex) and
# LangGraphTracer never reads config["configurable"]["thread_id"] or any
# checkpoint identity. Two worker processes independently recording "the
# same" logical operation (e.g. an original long-running tool call and its
# checkpoint-swept re-dispatch on a managed platform — see issue #7417)
# therefore produce two unrelated, un-linkable traces today: there is no way
# to recognize or diff them as the same logical run after the fact.
#
# derive_trace_id() lets a caller opt into a *deterministic* trace_id (still
# a 128-bit hex string, so it stays valid for the OTLP exporter) computed
# from LangGraph's own thread_id (+ an optional checkpoint id), instead of a
# fresh random UUID:
#
#     thread_id = config["configurable"]["thread_id"]
#     with tracer.start_trace("my_graph", record=True,
#                              trace_id=derive_trace_id(thread_id)) as trace:
#         graph.invoke(inputs, config=config)
#
# Two processes deriving from the same thread_id (+ checkpoint id, when one
# resumes from a specific checkpoint) always produce the identical trace_id,
# letting tooling recognize/diff them as the same logical run — exactly the
# "tooling to diff runs" use case Trace's own docstring already describes
# trace_id as existing for.


def derive_trace_id(thread_id: str, checkpoint_id: str | None = None) -> str:
    """Deterministically derive a 128-bit-hex trace_id from a LangGraph
    ``thread_id`` (and, optionally, a checkpoint id).

    Same (thread_id, checkpoint_id) always produces the same trace_id,
    regardless of which process or machine computes it — the mechanism
    needed to correlate an original run and its checkpoint-swept
    re-dispatch (issue #7417) as one logical run after the fact.
    """
    material = thread_id if checkpoint_id is None else f"{thread_id}:{checkpoint_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Pre-invocation (graph-construction-phase) instrumentation entry point
# ---------------------------------------------------------------------------
#
# LangGraphTracer only ever attaches at graph.invoke()/graph.stream() time —
# a `config["callbacks"]=[LangGraphTracer(...)]` argument passed to a call
# that must already exist. `langgraph dev`/LangGraph Platform inverts that:
# it imports the developer's own `make_graph()`-style factory function once
# at server startup and calls it *before* any invocation exists to attach a
# callback to. A bug in that construction phase itself (MCP client setup,
# tool loading, config parsing — issue #4798) is therefore unreachable by
# LangGraphTracer no matter how it's wired into the eventual invoke() call.
#
# instrument_graph_factory() closes that gap: it wraps the factory function
# itself, so recording/tracing is active for the duration of the call that
# *builds* the graph, not only the calls that later run it.


def instrument_graph_factory(
    tracer: Tracer,
    factory: Callable[..., Any] | None = None,
    *,
    name: str = "graph-construction",
) -> Any:
    """Give a LangGraph Platform ``make_graph()``-style entry point a
    construction-phase instrumentation hook, so tracing/recording is active
    while the graph is being *built* — MCP client setup, tool loading,
    config parsing — not only while it is later invoked.

    Usable as a decorator directly on the factory::

        from agent_trace import tracer
        from agent_trace.integrations.langgraph import instrument_graph_factory

        @instrument_graph_factory(tracer)
        def make_graph(config: dict) -> CompiledStateGraph:
            mcp_client = MultiServerMCPClient(...)   # now captured
            tools = mcp_client.get_tools()
            return build_graph(tools)

    ...or wrapping an existing factory inline::

        graph = instrument_graph_factory(tracer, make_graph)(config)

    Behavior depends on whether a trace is already active in the calling
    context:

      - If :attr:`Tracer.active_trace` is already set (e.g.
        :meth:`Tracer.start_auto_record` activated via
        ``AGENT_TRACE_AUTO_RECORD``, or an enclosing
        ``start_trace(record=True)`` block), the factory call is wrapped in
        its own child ``graph-construction`` span, nested under whatever is
        already active, so construction-phase work is visible as a
        distinct span in the trace.
      - If no trace is active at all, this activates a scoped
        ``start_trace(name, record=True)`` for the duration of the factory
        call only — so at minimum, HTTP calls made during construction
        (e.g. an MCP ``streamable_http`` client's tool-listing request) are
        captured, closing the alternative of *zero* visibility into
        construction-phase bugs.

    Works for both sync and async factories (``async def make_graph(...)``
    is detected via :func:`inspect.iscoroutinefunction` and awaited
    correctly).
    """

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if tracer.active_trace is not None:
                    with tracer.span(name):
                        return await fn(*args, **kwargs)
                with tracer.start_trace(name, record=True):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if tracer.active_trace is not None:
                with tracer.span(name):
                    return fn(*args, **kwargs)
            with tracer.start_trace(name, record=True):
                return fn(*args, **kwargs)

        return wrapper

    if factory is not None:
        return _decorate(factory)
    return _decorate


# ---------------------------------------------------------------------------
# Runtime/context capture
# ---------------------------------------------------------------------------
#
# LangGraph injects a per-run ``Runtime``/context object into node functions
# via a private config key (``CONFIG_KEY_RUNTIME`` = "__pregel_runtime")
# that is never exposed to the public ``BaseCallbackHandler`` interface —
# on_chain_start only ever receives (serialized, inputs, run_id, ...), with
# no way to see the runtime object at all. The only way to capture it is to
# read it out of the RunnableConfig at the point LangGraph itself resolves
# it, which happens inside ``RunnableCallable.invoke``/``ainvoke`` — a
# private module. This is stashed into a ContextVar immediately before
# LangGraph calls into the callback manager, so LangGraphTracer.on_chain_start
# (which fires synchronously inside that same call) can read it back out.

_current_runtime: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "agent_trace_langgraph_runtime", default=None
)

_runtime_patch_lock = threading.Lock()
_runtime_patch_installed = False


def _install_runtime_capture_patch() -> None:
    """Best-effort monkeypatch that makes the LangGraph Runtime/context object
    observable to LangGraphTracer.

    This touches a private LangGraph module
    (``langgraph._internal._runnable``/``langgraph._internal._constants``)
    that may change shape across versions without notice. Every step here is
    wrapped so a mismatch degrades to "no runtime captured" (the pre-existing
    behavior) rather than breaking tracing or import of this module.
    """
    global _runtime_patch_installed  # noqa: PLW0603
    if _runtime_patch_installed:
        return
    with _runtime_patch_lock:
        if _runtime_patch_installed:
            return
        try:
            from langgraph._internal._constants import (
                CONF,
                CONFIG_KEY_RUNTIME,
            )
            from langgraph._internal._runnable import (
                RunnableCallable,
            )
        except Exception:
            logger.debug(
                "agent-trace: LangGraph Runtime-capture patch unavailable "
                "(internal module shape not as expected on this LangGraph "
                "version); Runtime/context objects will not be captured on "
                "spans.",
                exc_info=True,
            )
            _runtime_patch_installed = True  # don't retry every call
            return

        original_invoke = RunnableCallable.invoke
        original_ainvoke = RunnableCallable.ainvoke

        def _capture_runtime_from_config(config: Any) -> None:
            """Best-effort: stash config's Runtime object into the ContextVar.

            Swallows everything — a shape mismatch here must never break the
            actual LangGraph invocation it's piggybacking on.
            """
            try:
                if config is not None:
                    runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)
                    if runtime is not None:
                        _current_runtime.set(runtime)
            except Exception:
                logger.debug(
                    "agent-trace: failed to read Runtime/context off config",
                    exc_info=True,
                )

        def _patched_invoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            _capture_runtime_from_config(config)
            return original_invoke(self, input, config, **kwargs)

        async def _patched_ainvoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            _capture_runtime_from_config(config)
            return await original_ainvoke(self, input, config, **kwargs)

        RunnableCallable.invoke = _patched_invoke  # type: ignore[method-assign]
        RunnableCallable.ainvoke = _patched_ainvoke  # type: ignore[method-assign]
        _runtime_patch_installed = True


# ---------------------------------------------------------------------------
# Exception classification — origin layer + known error signatures
# ---------------------------------------------------------------------------
#
# agent-trace's callback layer already captures exception.type/message onto
# error spans, but nothing classifies it: a developer has to read the raw
# trace.json and independently recognize a known LangGraph error code (e.g.
# ErrorCode.INVALID_CHAT_HISTORY) or figure out whether the failure came from
# the LLM provider's SDK or from application/framework code. This section
# tags every error span with a best-effort "error.origin" and, where the
# message matches a known pattern, an "error.known_pattern" attribute.

# Top-level package names recognized as an LLM-provider SDK. type(exc).__module__
# starts with one of these -> the exception originated inside the provider's
# client library (a 4xx/5xx wire error, a provider-side validation error),
# not inside the developer's own chain/application code.
_PROVIDER_MODULE_PREFIXES: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "groq",
        "google",
        "genai",
        "cohere",
        "mistralai",
        "boto3",
        "botocore",
    }
)

# Top-level package names recognized as framework/orchestration code, as
# distinct from the developer's own application code.
_CHAIN_MODULE_PREFIXES: frozenset[str] = frozenset(
    {"langgraph", "langchain_core", "langchain", "langchain_community"}
)

# (message substring, label) pairs for exception messages that match a known,
# previously root-caused failure signature. Matched case-insensitively
# against str(exc). Order matters only in that the first match wins, but
# entries are written to be mutually exclusive in practice.
_KNOWN_ERROR_SIGNATURES: tuple[tuple[str, str], ...] = (
    (
        "invalid_chat_history",
        "langgraph_invalid_chat_history",
    ),
    (
        "selected invalid tool",
        "middleware_invalid_tool_selection",
    ),
)


def _classify_exception_origin(error: BaseException) -> str:
    """Best-effort "which layer did this exception come from" tag.

    Returns "provider" (an LLM-SDK-raised exception — e.g. an OpenAI/
    Anthropic/Groq 4xx), "chain" (LangGraph/LangChain framework code), or
    "application" (everything else — the developer's own node/tool code, the
    most common case for a genuine bug).
    """
    module = type(error).__module__ or ""
    top_level = module.split(".", 1)[0]
    if top_level in _PROVIDER_MODULE_PREFIXES:
        return "provider"
    if top_level in _CHAIN_MODULE_PREFIXES:
        return "chain"
    return "application"


def _match_known_error_signature(message: str) -> str | None:
    """Return a short label if *message* matches a known, previously
    root-caused failure signature, else None."""
    if not message:
        return None
    lowered = message.lower()
    for needle, label in _KNOWN_ERROR_SIGNATURES:
        if needle in lowered:
            return label
    return None


def _classify_and_tag_exception(span: Span, error: BaseException) -> None:
    """Apply error.origin + error.known_pattern attributes to *span*.

    Called after ``span.record_exception`` for any span closing ERROR, so
    the classification is available directly on the span instead of
    requiring a developer to read exception.message out of raw trace.json
    and recognize the pattern themselves.
    """
    span.set_attribute("error.origin", _classify_exception_origin(error))
    known_pattern = _match_known_error_signature(str(error))
    if known_pattern:
        span.set_attribute("error.known_pattern", known_pattern)


# ---------------------------------------------------------------------------
# LangGraph internal control-flow signals — not application errors
# ---------------------------------------------------------------------------
#
# LangGraph raises its own exceptions internally to implement control flow
# that has nothing to do with application failure:
#   - ParentCommand: raised when a node returns Command(graph=Command.PARENT,
#     ...) to implement a multi-agent handoff jump up to the parent graph.
#   - GraphInterrupt: raised when a node calls interrupt() to pause a run
#     for human-in-the-loop resumption.
# Both subclass langgraph.errors.GraphBubbleUp (confirmed against the
# installed langgraph package). Without special-casing these, on_chain_error
# marks the node span ERROR with exception.type=ParentCommand/GraphInterrupt,
# identical in shape to a genuine application exception — a developer has to
# manually filter these out before finding the real error in a trace.

_control_flow_exception_types: tuple[type[BaseException], ...] | None = None
_control_flow_types_lock = threading.Lock()


def _get_control_flow_exception_types() -> tuple[type[BaseException], ...]:
    """Lazily resolve LangGraph's internal control-flow exception types.

    Tries the shared ``GraphBubbleUp`` base first (covers both
    ``ParentCommand`` and ``GraphInterrupt`` in one isinstance check on
    LangGraph versions that have it). Falls back to importing the two known
    concrete types individually for versions where no shared base exists.
    Every import is wrapped so a version mismatch degrades to "nothing
    special-cased" (the pre-existing behavior) rather than breaking tracing.
    """
    global _control_flow_exception_types  # noqa: PLW0603
    if _control_flow_exception_types is not None:
        return _control_flow_exception_types
    with _control_flow_types_lock:
        if _control_flow_exception_types is not None:
            return _control_flow_exception_types

        found: list[type[BaseException]] = []
        try:
            from langgraph.errors import GraphBubbleUp

            found.append(GraphBubbleUp)
        except Exception:
            for type_name in ("ParentCommand", "GraphInterrupt"):
                try:
                    from langgraph import errors as _lg_errors

                    found.append(getattr(_lg_errors, type_name))
                except Exception:
                    logger.debug(
                        "agent-trace: could not import langgraph.errors.%s "
                        "on this LangGraph version; it will not be "
                        "special-cased as a control-flow signal.",
                        type_name,
                        exc_info=True,
                    )
        _control_flow_exception_types = tuple(found)
        return _control_flow_exception_types


def _is_langgraph_control_flow_signal(error: BaseException) -> bool:
    """True if *error* is LangGraph's own internal control-flow signal
    (a Command/ParentCommand handoff jump or a GraphInterrupt pause) rather
    than an application-level exception."""
    types_ = _get_control_flow_exception_types()
    return bool(types_) and isinstance(error, types_)


def _record_control_flow_signal(span: Span, error: BaseException) -> None:
    """Close *span* OK with an informational attribute instead of ERROR.

    Distinguishes a GraphInterrupt (run paused, not failed) from a
    ParentCommand/other handoff jump via separate boolean attributes, since
    the two mean different things to a developer reading the trace.
    """
    type_name = type(error).__name__
    span.set_attribute("langgraph.control_flow_signal", type_name)
    if type_name == "GraphInterrupt":
        span.set_attribute("langgraph.interrupted", True)
    else:
        span.set_attribute("langgraph.handoff", True)
    span.add_event(
        "langgraph_control_flow",
        attributes={
            "control_flow.type": type_name,
            "control_flow.message": _safe_str(error)[:_MAX_ATTR_LEN],
        },
    )


# ---------------------------------------------------------------------------
# Branch (conditional-edge) dispatch exception capture
# ---------------------------------------------------------------------------
#
# LangGraph builds a conditional edge's routing dispatch as a RunnableCallable
# constructed with trace=False (langgraph/graph/_branch.py, BranchSpec.run()):
# a deliberate choice by LangGraph itself so the dispatch step doesn't show up
# as its own chain span. RunnableCallable.invoke/ainvoke skip the callback
# manager entirely when self.trace is falsy (langgraph/_internal/_runnable.py)
# — no on_chain_start/on_chain_error ever fires for this component, so an
# exception raised inside it (e.g. a KeyError from BranchSpec._finish() when a
# router's return value doesn't match a registered destination) produces zero
# agent-trace spans or callback events today.
#
# The capture point has to be RunnableCallable.invoke/ainvoke themselves (the
# same class the Runtime-capture patch above already wraps), NOT
# BranchSpec._route/_aroute directly: BranchSpec.run() captures
# `func=self._route`/`afunc=self._aroute` as bound-method values baked into a
# RunnableCallable instance at *graph-compile time* (builder.compile()) —
# typically long before any LangGraphTracer is ever constructed. Patching
# BranchSpec._route/_aroute as class attributes only affects bound-method
# lookups that happen *after* the patch installs; a RunnableCallable compiled
# earlier already holds a direct reference to the pre-patch function and would
# never observe a later patch. RunnableCallable.invoke/ainvoke, by contrast,
# are resolved fresh via normal method lookup every time `.invoke()`/
# `.ainvoke()` is called on any instance — patching them here (regardless of
# when any given RunnableCallable was constructed) reliably intercepts every
# call, exactly like the Runtime-capture patch above already relies on.

_current_langgraph_tracer: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "agent_trace_langgraph_current_tracer", default=None
)

_branch_patch_lock = threading.Lock()
_branch_patch_installed = False


def _is_branch_dispatch_callable(runnable_callable: Any, branch_spec_cls: type) -> bool:
    """True if *runnable_callable* wraps a BranchSpec._route/_aroute bound
    method — i.e. this is LangGraph's conditional-edge dispatch step, not one
    of the other unrelated trace=False RunnableCallables LangGraph also
    constructs internally (channel writes, ToolNode, etc.)."""
    for attr_name in ("func", "afunc"):
        bound_method = getattr(runnable_callable, attr_name, None)
        bound_owner = getattr(bound_method, "__self__", None)
        if isinstance(bound_owner, branch_spec_cls):
            return True
    return False


def _branch_spec_instance(runnable_callable: Any) -> Any | None:
    """Return the BranchSpec instance a dispatch RunnableCallable wraps, or
    None. Shared by the router-name and registered-destinations lookups
    below so there's exactly one place that reaches into
    ``.func``/``.afunc.__self__``."""
    bound_method = getattr(runnable_callable, "func", None) or getattr(
        runnable_callable, "afunc", None
    )
    return getattr(bound_method, "__self__", None)


def _resolve_router_name(runnable_callable: Any) -> str:
    """Best-effort: the user's own router function name (e.g.
    ``should_continue``) for a Branch dispatch RunnableCallable —
    ``BranchSpec.path`` is the Runnable wrapping whatever callable was
    passed to ``add_conditional_edges``, typically a ``RunnableLambda``
    whose ``.func``/``.afunc`` is the original function object. Falls back
    to ``"<anonymous>"`` for a lambda or when the shape doesn't match
    (private LangGraph/LangChain internals, may change across versions).
    """
    branch_self = _branch_spec_instance(runnable_callable)
    path = getattr(branch_self, "path", None)
    underlying = getattr(path, "func", None) or getattr(path, "afunc", None) or path
    name = getattr(underlying, "__name__", None)
    return str(name) if name else "<anonymous>"


def _registered_destinations(runnable_callable: Any) -> str:
    """Comma-joined set of destinations this Branch could route to (its
    ``path_map``/``ends``), or "" if unavailable."""
    branch_self = _branch_spec_instance(runnable_callable)
    ends = getattr(branch_self, "ends", None) or {}
    return ",".join(str(v) for v in ends.values())


def _record_branch_dispatch(
    runnable_callable: Any, *, error: BaseException | None
) -> None:
    """Open a standalone ``branch:dispatch`` span recording which
    conditional-edge/router function made a routing decision — on both the
    success and failure path — since LangGraph deliberately builds this
    component with ``trace=False`` and no ``on_chain_start``/``on_chain_end``/
    ``on_chain_error`` ever fires for it otherwise.

    Without this, a tool-call span's downstream exception (e.g. a
    ``ValidationError`` on a missing injected field) looks identical
    regardless of whether the cause was a routing bug that dispatched to the
    wrong node, or genuinely malformed model output — a developer has no
    record of which router ran, or that it ran at all.

    Best-effort: swallows every exception itself (an instrumentation bug
    here must never break — or change the exception raised by — the real
    LangGraph routing dispatch it's piggybacking on). No-ops entirely if no
    LangGraphTracer instance is active in this context (e.g. the graph was
    invoked without one).
    """
    handler = _current_langgraph_tracer.get()
    if handler is None:
        return
    try:
        span = handler._tracer.start_span("branch:dispatch")
        span.set_attribute("langgraph.branch_dispatch", True)
        router_name = _resolve_router_name(runnable_callable)
        span.set_attribute("branch.router_name", router_name)
        destinations = _registered_destinations(runnable_callable)
        if destinations:
            span.set_attribute("branch.registered_destinations", destinations)
        if error is not None:
            span.record_exception(error)
            _classify_and_tag_exception(span, error)
            span.end(SpanStatus.ERROR)
        else:
            span.end(SpanStatus.OK)
    except Exception:
        logger.debug(
            "agent-trace: failed to record Branch dispatch",
            exc_info=True,
        )


def _install_branch_dispatch_capture_patch() -> None:
    """Best-effort monkeypatch making LangGraph's trace=False Branch dispatch
    (conditional-edge routing) observable to agent-trace — both which router
    ran and, on failure, why it raised.

    Touches private-ish LangGraph modules (``langgraph.graph._branch``,
    ``langgraph._internal._runnable``) that may change shape across versions
    without notice; every step here is wrapped so a mismatch degrades to
    "dispatch not captured" (the pre-existing behavior) rather than breaking
    tracing or import.
    """
    global _branch_patch_installed  # noqa: PLW0603
    if _branch_patch_installed:
        return
    with _branch_patch_lock:
        if _branch_patch_installed:
            return
        try:
            from langgraph._internal._runnable import RunnableCallable
            from langgraph.graph._branch import BranchSpec
        except Exception:
            logger.debug(
                "agent-trace: LangGraph Branch dispatch capture patch "
                "unavailable (internal module shape not as expected on "
                "this LangGraph version); conditional-edge routing "
                "dispatch will not be captured.",
                exc_info=True,
            )
            _branch_patch_installed = True  # don't retry every call
            return

        original_invoke = RunnableCallable.invoke
        original_ainvoke = RunnableCallable.ainvoke

        def _patched_invoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            is_branch = not self.trace and _is_branch_dispatch_callable(
                self, BranchSpec
            )
            try:
                result = original_invoke(self, input, config, **kwargs)
            except BaseException as exc:
                if is_branch:
                    _record_branch_dispatch(self, error=exc)
                raise
            if is_branch:
                _record_branch_dispatch(self, error=None)
            return result

        async def _patched_ainvoke(
            self: Any, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            is_branch = not self.trace and _is_branch_dispatch_callable(
                self, BranchSpec
            )
            try:
                result = await original_ainvoke(self, input, config, **kwargs)
            except BaseException as exc:
                if is_branch:
                    _record_branch_dispatch(self, error=exc)
                raise
            if is_branch:
                _record_branch_dispatch(self, error=None)
            return result

        RunnableCallable.invoke = _patched_invoke  # type: ignore[method-assign]
        RunnableCallable.ainvoke = _patched_ainvoke  # type: ignore[method-assign]
        _branch_patch_installed = True


# ---------------------------------------------------------------------------
# Tool-argument injection (InjectedState/InjectedStore) capture
# ---------------------------------------------------------------------------
#
# ToolNode._inject_tool_args() resolves Annotated[..., InjectedState]/
# InjectedStore/InjectedToolCallId tool arguments right before the tool is
# actually invoked — confirmed via direct inspection of the installed
# langgraph.prebuilt.tool_node module: it is called via normal instance
# method lookup (self._inject_tool_args(...)) from ToolNode's own dispatch
# code, so patching the class method (like RunnableCallable.invoke/ainvoke
# above) reliably intercepts every call regardless of when any given
# ToolNode instance was constructed. Nothing in agent_trace previously
# observed this step at all (confirmed via repo-wide grep: zero hits for
# InjectedState/InjectedStore/inject_tool_args before this patch) — a
# ValidationError raised later inside the tool call itself looks identical
# whether the cause was a routing bug that bypassed injection, a caller
# passing a malformed tool schema, or genuinely malformed model output.
#
# _inject_tool_args runs *before* the tool's own .invoke()/.ainvoke() call,
# which is what fires on_tool_start — so there is no open tool: span yet to
# attach to at the point this patch observes injection. Recorded as its own
# standalone span instead (the same pattern as branch:dispatch above), whose
# start_time necessarily precedes the corresponding tool: span's.

_tool_injection_patch_lock = threading.Lock()
_tool_injection_patch_installed = False


def _record_tool_arg_injection(tool_name: str, injected_keys: list[str]) -> None:
    """Open a standalone ``tool_inject:<tool_name>`` span recording whether
    InjectedState/InjectedStore/InjectedToolCallId resolution ran for
    *tool_name*, and which argument names it resolved.

    Deliberately named with a ``tool_inject:`` prefix rather than
    ``tool:...`` — code/tests filtering spans by ``name.startswith("tool:")``
    to find the *actual* tool-call span must not also match this one, since
    this span always precedes (and is distinct from) the real ``tool:<name>``
    span for the same call (injection runs before on_tool_start fires).

    Best-effort: swallows every exception itself. No-ops entirely if no
    LangGraphTracer instance is active in this context.
    """
    handler = _current_langgraph_tracer.get()
    if handler is None:
        return
    try:
        span = handler._tracer.start_span(f"tool_inject:{tool_name}")
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.injection_ran", bool(injected_keys))
        if injected_keys:
            span.set_attribute("tool.injected_arg_keys", ",".join(injected_keys))
        span.end(SpanStatus.OK)
    except Exception:
        logger.debug(
            "agent-trace: failed to record tool-argument injection for "
            "tool %r",
            tool_name,
            exc_info=True,
        )


def _install_tool_arg_injection_capture_patch() -> None:
    """Best-effort monkeypatch making ``ToolNode``'s tool-argument injection
    step observable to agent-trace.

    Touches ``langgraph.prebuilt.tool_node.ToolNode._inject_tool_args``,
    which may change shape across LangGraph versions without notice; every
    step here is wrapped so a mismatch degrades to "injection not captured"
    (the pre-existing behavior) rather than breaking tracing, import, or a
    real tool call.
    """
    global _tool_injection_patch_installed  # noqa: PLW0603
    if _tool_injection_patch_installed:
        return
    with _tool_injection_patch_lock:
        if _tool_injection_patch_installed:
            return
        try:
            from langgraph.prebuilt.tool_node import ToolNode
        except Exception:
            logger.debug(
                "agent-trace: LangGraph tool-argument injection capture "
                "patch unavailable (ToolNode._inject_tool_args not found "
                "on this LangGraph version); InjectedState/InjectedStore "
                "resolution will not be captured.",
                exc_info=True,
            )
            _tool_injection_patch_installed = True  # don't retry every call
            return

        original_inject = ToolNode._inject_tool_args

        def _patched_inject_tool_args(
            self: Any, tool_call: Any, tool_runtime: Any, tool: Any = None
        ) -> Any:
            result = original_inject(self, tool_call, tool_runtime, tool)
            try:
                tool_name = tool_call.get("name", "") if tool_call else ""
                injected = (self._injected_args or {}).get(tool_name)
                injected_keys = (
                    sorted(str(k) for k in injected.all_injected_keys)
                    if injected is not None
                    else []
                )
                _record_tool_arg_injection(tool_name, injected_keys)
            except Exception:
                logger.debug(
                    "agent-trace: failed to observe tool-argument "
                    "injection outcome",
                    exc_info=True,
                )
            return result

        ToolNode._inject_tool_args = _patched_inject_tool_args  # type: ignore[method-assign]
        _tool_injection_patch_installed = True


# ---------------------------------------------------------------------------
# on_llm_end helpers — pulled out to keep the callback itself flat/scannable
# ---------------------------------------------------------------------------


def _first_generation_and_message(response: Any) -> tuple[Any, Any]:
    """Return (first_generation, first_generation.message) from a
    langchain-core LLMResult/ChatResult, or (None, None) if absent."""
    generations = getattr(response, "generations", None) or []
    if not generations or not generations[0]:
        return None, None
    first_gen = generations[0][0]
    first_message = getattr(first_gen, "message", None)
    return first_gen, first_message


def _extract_token_usage(response: Any, first_message: Any) -> dict[str, Any]:
    """token_usage from llm_output, falling back to the first message's
    usage_metadata — modern langchain-core often attaches usage there
    independently of llm_output, especially under streaming configurations.
    """
    usage = getattr(response, "llm_output", {}) or {}
    token_usage = usage.get("token_usage") or usage.get("usage") or {}
    if token_usage or first_message is None:
        return token_usage
    msg_usage = getattr(first_message, "usage_metadata", None)
    if not msg_usage:
        return {}
    return {
        "prompt_tokens": msg_usage.get("input_tokens", 0),
        "completion_tokens": msg_usage.get("output_tokens", 0),
        "total_tokens": msg_usage.get("total_tokens", 0),
    }


def _extract_finish_reason(
    response_metadata: dict[str, Any] | None,
    generation_info: dict[str, Any] | None,
) -> str | None:
    """finish_reason (or stop_reason) from response_metadata, falling back to
    generation_info — e.g. Gemini's finish_reason=MALFORMED_FUNCTION_CALL,
    OpenAI's finish_reason/tool_calls presence signal."""
    for source in (response_metadata, generation_info):
        if not source:
            continue
        reason = source.get("finish_reason") or source.get("stop_reason")
        if reason:
            return str(reason)
    return None


def _extract_content(first_message: Any, first_gen: Any) -> Any:
    """Actual generated content — not just usage counts.

    Chat models: response.generations[0][0].message.content
    Legacy completions: response.generations[0][0].text
    """
    if first_message is not None:
        return getattr(first_message, "content", None)
    if first_gen is not None:
        return getattr(first_gen, "text", None)
    return None


def _record_llm_end_data(span: Span, response: Any) -> None:
    """Extract and persist everything on_llm_end previously discarded:
    token usage (with the usage_metadata fallback), response_metadata/
    generation_info (finish_reason, tool-call presence), and the actual
    generated content."""
    first_gen, first_message = _first_generation_and_message(response)

    token_usage = _extract_token_usage(response, first_message)
    if token_usage:
        span.set_attribute(
            "llm.usage.prompt_tokens", int(token_usage.get("prompt_tokens", 0))
        )
        span.set_attribute(
            "llm.usage.completion_tokens",
            int(token_usage.get("completion_tokens", 0)),
        )
        span.set_attribute(
            "llm.usage.total_tokens", int(token_usage.get("total_tokens", 0))
        )

    response_metadata = (
        getattr(first_message, "response_metadata", None)
        if first_message is not None
        else None
    )
    generation_info = getattr(first_gen, "generation_info", None)

    finish_reason = _extract_finish_reason(response_metadata, generation_info)
    if finish_reason:
        span.set_attribute("llm.finish_reason", finish_reason)
    if first_message is not None:
        span.set_attribute(
            "llm.has_tool_calls", bool(getattr(first_message, "tool_calls", None))
        )
    if response_metadata:
        span.set_attribute("llm.response_metadata", _to_attr_string(response_metadata))
    if generation_info:
        span.set_attribute("llm.generation_info", _to_attr_string(generation_info))

    content = _extract_content(first_message, first_gen)
    if content:
        span.set_attribute("llm.content", _stringify(content))


def _extract_tool_call_chunks(chunk: Any) -> Any:
    """Best-effort extraction of a streaming ChatGenerationChunk's
    ``message.tool_call_chunks`` (the partial/incremental tool-call-argument
    fragments LangChain attaches per streamed delta), or None if *chunk* is
    absent or isn't that shape (e.g. a plain GenerationChunk from a legacy,
    non-chat LLM)."""
    message = getattr(chunk, "message", None)
    if message is None:
        return None
    return getattr(message, "tool_call_chunks", None) or None


def _get_declared_node_tags(graph: Any, node_name: str) -> list[str] | None:
    """Best-effort: read a compiled LangGraph graph's node-level *declared*
    tags — the tags a developer attached to the node's own action/runnable
    at graph-construction time (e.g. ``builder.add_node("n",
    my_fn.with_config(tags=["nostream"]))``) — as distinct from the purely
    LangGraph-internal *runtime* tags (e.g. ``"graph:step:2"``) that
    ``on_chain_start``'s own ``tags`` kwarg already carries for every node
    run, which never include a developer-declared tag like ``"nostream"``.

    How this actually works (confirmed by direct inspection of the
    installed LangGraph, not assumed from docs): the current
    ``StateGraph.add_node()`` has **no** ``tags=`` keyword argument at all —
    a node's own declared tags only exist if the developer wrapped the
    node's action in ``.with_config(tags=[...])`` *before* passing it to
    ``add_node()``. LangGraph's own ``PregelNode.tags`` field (whose
    docstring says "Tags to attach to the node for tracing") is never
    actually populated by ``StateGraph`` for a regular node — it stays
    ``None`` regardless of what was declared. The tags survive only on the
    compiled node's own bound Runnable's ``.config`` dict, reachable at
    ``graph.nodes[node_name].bound.config.get("tags")``.

    Wrapped entirely in try/except: this reaches into private-ish LangGraph
    attributes (``.nodes``, ``.bound``, ``.config``) that may not exist, or
    may be shaped differently, on other LangGraph versions — a mismatch
    degrades to "no declared tags captured" (returns None), never an
    exception raised into the caller's ``graph.invoke()``/``.stream()``
    call.
    """
    try:
        nodes = getattr(graph, "nodes", None)
        if nodes is None:
            return None
        node = nodes.get(node_name)
        if node is None:
            return None
        bound = getattr(node, "bound", None)
        config = getattr(bound, "config", None) or {}
        tags = config.get("tags")
        if not tags:
            return None
        return [str(t) for t in tags]
    except Exception:
        logger.debug(
            "agent-trace: failed to read declared node tags for node %r",
            node_name,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Schema-level check: tool parameters shaped like framework state that
# aren't marked injected
# ---------------------------------------------------------------------------
#
# A bound tool's *model-facing* JSON schema (BaseTool.args, which LangChain
# itself already computes with every Annotated[..., InjectedState]/
# InjectedStore/InjectedToolCallId parameter excluded — confirmed by direct
# inspection: a param annotated InjectedState never appears in tool.args)
# tells us exactly which parameters the LLM will have to invent a value for
# at call time. Comparing those names against the compiled graph's own state
# field names (StateGraph.builder.schemas) encodes, as an automatable rule,
# the diagnosis a LangGraph maintainer made purely by inspection in issue
# #3266 ("the LLM is populating the state field and hallucinating the video
# path"): a tool parameter that happens to share a name with a real state
# field, but was never excluded from the model-facing schema via
# InjectedState, is a parameter the model will fill in itself rather than
# receive from real graph state.


def _state_field_names(graph: Any) -> set[str]:
    """Best-effort: every field name declared across a compiled StateGraph's
    state schema(s) (``builder.schemas`` maps each schema class to a dict of
    channel-name -> Channel). Returns an empty set if the shape doesn't
    match (private-ish LangGraph attribute, may change across versions)."""
    try:
        schemas = getattr(getattr(graph, "builder", None), "schemas", None) or {}
        names: set[str] = set()
        for channel_map in schemas.values():
            names.update(str(k) for k in channel_map)
        return names
    except Exception:
        logger.debug(
            "agent-trace: failed to read state schema field names off graph",
            exc_info=True,
        )
        return set()


def _tools_by_name_for_node(node: Any) -> dict[str, Any]:
    """Best-effort: the ``{tool_name: BaseTool}`` mapping a compiled graph
    node runs, if that node is (or wraps) a ``ToolNode`` — read off the
    node's own ``bound`` Runnable's ``_tools_by_name`` attribute. Returns {}
    for any node that isn't a ToolNode (the overwhelmingly common case —
    most nodes aren't) or if the shape doesn't match."""
    try:
        bound = getattr(node, "bound", None)
        tools_by_name = getattr(bound, "_tools_by_name", None)
        if not tools_by_name:
            return {}
        return dict(tools_by_name)
    except Exception:
        return {}


def find_tool_params_shaped_like_state(graph: Any) -> list[dict[str, str]]:
    """Flag tool parameters that are model-facing (not excluded via
    InjectedState/InjectedStore/InjectedToolCallId) but share a name with
    one of the compiled graph's own state fields.

    Returns a list of ``{"node": ..., "tool": ..., "param": ...}`` findings
    (empty if none, or if the graph/state-schema shape can't be introspected
    — e.g. *graph* isn't a compiled LangGraph ``CompiledStateGraph``).
    Read-only: never mutates *graph* or raises into the caller.
    """
    findings: list[dict[str, str]] = []
    try:
        state_fields = _state_field_names(graph)
        if not state_fields:
            return findings
        nodes = getattr(graph, "nodes", None) or {}
        for node_name, node in nodes.items():
            tools_by_name = _tools_by_name_for_node(node)
            for tool_name, tool_obj in tools_by_name.items():
                model_facing_args = getattr(tool_obj, "args", None) or {}
                for param_name in model_facing_args:
                    if param_name in state_fields:
                        findings.append(
                            {
                                "node": str(node_name),
                                "tool": str(tool_name),
                                "param": str(param_name),
                            }
                        )
    except Exception:
        logger.debug(
            "agent-trace: failed to check tool params against state schema",
            exc_info=True,
        )
    return findings


def _require_langchain_core() -> Any:
    """Lazy import guard — raises a clear error if langchain_core is absent."""
    try:
        import langchain_core

        return langchain_core
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _base_callback_handler() -> Any:
    """Return langchain_core.callbacks.BaseCallbackHandler (lazy import)."""
    _require_langchain_core()
    from langchain_core.callbacks import BaseCallbackHandler

    return BaseCallbackHandler


# Module-level singleton: the concrete tracer class is built once (the first
# time _get_tracer_class() is called) so that BaseCallbackHandler is a real
# base at class-definition time rather than being spliced in at instantiation
# via __bases__ mutation (which is unsafe under concurrency and forbidden in
# Python 3.14+).
_LangGraphTracerClass: type | None = None
_tracer_class_lock: threading.Lock = threading.Lock()


def _get_tracer_class() -> type:
    """Return (and lazily build) the concrete LangGraphTracer implementation.

    The class is created exactly once, with ``BaseCallbackHandler`` as a
    genuine base class at definition time.  Subsequent calls return the cached
    class without re-importing or re-defining anything.
    """
    global _LangGraphTracerClass  # noqa: PLW0603
    if _LangGraphTracerClass is not None:
        return _LangGraphTracerClass

    with _tracer_class_lock:
        # Double-checked locking: re-test inside the lock.
        if _LangGraphTracerClass is not None:
            return _LangGraphTracerClass

        base_cls: type = _base_callback_handler()
        # Best-effort — makes the injected Runtime/context object observable
        # to on_chain_start below. No-ops safely if the private LangGraph
        # internals it depends on aren't present.
        _install_runtime_capture_patch()
        # Best-effort — makes Branch (conditional-edge) routing dispatch
        # (which router ran, and on failure why it raised) observable
        # despite LangGraph building that component with trace=False.
        # No-ops safely if the internals it depends on aren't present.
        _install_branch_dispatch_capture_patch()
        # Best-effort — makes ToolNode's InjectedState/InjectedStore
        # argument-injection step observable. No-ops safely if the
        # internals it depends on aren't present.
        _install_tool_arg_injection_capture_patch()

        class _LangGraphTracerImpl(base_cls):  # type: ignore[misc]
            """Concrete implementation — see LangGraphTracer for public docs."""

            def __init__(
                self,
                tracer: Tracer,
                trace: Trace,
                *,
                graph: Any = None,
                long_span_threshold_secs: float | None = None,
            ) -> None:
                super().__init__()
                self._tracer: Tracer = tracer
                self._trace: Trace = trace
                # Optional: the compiled graph this tracer instruments.
                # When supplied, on_chain_start can additionally look up
                # each node's graph-construction-time *declared* tags (see
                # _get_declared_node_tags) — information the runtime `tags`
                # callback kwarg alone never carries. None (the default)
                # keeps the pre-existing behavior: no declared-tags lookup,
                # every other capability unaffected.
                self._graph: Any = graph
                # Optional: flags any span (node/llm/tool) whose measured
                # open duration exceeds this many seconds once it closes —
                # e.g. LangGraph Cloud's ~180s checkpoint-sweep re-dispatch
                # window (issue #7417). None (the default) disables the
                # check entirely.
                self._long_span_threshold_secs: float | None = (
                    long_span_threshold_secs
                )
                # Thread-safe span registry: run_id (UUID str) -> open Span
                self._spans: dict[str, Span] = {}
                # Per-run streaming-token counters (on_llm_new_token), so
                # llm.stream_token_count can keep counting past the
                # per-span SpanEvent cap without holding the events
                # themselves. Cleared alongside the span on close.
                self._stream_token_counts: dict[str, int] = {}
                self._lock: threading.Lock = threading.Lock()
                # Best-effort: makes this instance discoverable to the
                # Branch-dispatch and tool-argument-injection patches (see
                # _record_branch_dispatch / _record_tool_arg_injection),
                # which have no other way to reach a LangGraphTracer/Tracer
                # since the callback manager never invokes them for a
                # trace=False component. Constructed in the same execution
                # context that will go on to call graph.invoke()/.ainvoke(),
                # so the ContextVar value is visible throughout that call.
                _current_langgraph_tracer.set(self)
                # Best-effort, one-time: flag tool parameters shaped like
                # graph state but not marked InjectedState/InjectedStore —
                # see find_tool_params_shaped_like_state(). Only runs when a
                # graph was supplied; writes findings onto the trace's own
                # metadata dict (not a span — this is a static property of
                # the graph's wiring, not a per-invocation event) so it
                # survives even if the graph never routes through the
                # flagged tool during this particular run.
                if graph is not None:
                    self._check_tool_state_shaped_params(graph)

            # ------------------------------------------------------------------
            # Internal helpers
            # ------------------------------------------------------------------

            def _check_tool_state_shaped_params(self, graph: Any) -> None:
                """Run find_tool_params_shaped_like_state() against *graph*
                once, at construction time, and record any findings onto
                the active trace's metadata dict under
                "tool_state_shaped_params" (JSON-serialized). Best-effort:
                a failure here must never break constructing the tracer."""
                try:
                    findings = find_tool_params_shaped_like_state(graph)
                    if not findings:
                        return
                    active = getattr(self._tracer, "active_trace", None)
                    if active is None:
                        return
                    active.metadata["tool_state_shaped_params"] = _to_attr_string(
                        findings
                    )
                except Exception:
                    logger.debug(
                        "agent-trace: failed to run "
                        "find_tool_params_shaped_like_state()",
                        exc_info=True,
                    )

            def _open_span(
                self,
                run_id: uuid.UUID | str,
                name: str,
                parent_run_id: uuid.UUID | str | None = None,
            ) -> Span:
                """Create a span and register it in the local registry."""
                run_key = str(run_id)
                parent_span_id: str | None = None
                if parent_run_id is not None:
                    with self._lock:
                        parent_span = self._spans.get(str(parent_run_id))
                    parent_span_id = (
                        parent_span.span_id if parent_span is not None else None
                    )

                span = self._tracer.start_span(name, parent_id=parent_span_id)
                if name.startswith("llm:") and parent_span_id is not None:
                    with self._lock:
                        parent_span = self._spans.get(str(parent_run_id))
                    if parent_span is not None and parent_span.name.startswith("tool:"):
                        # This LLM call is happening inside a tool's own
                        # execution, not directly under a graph node —
                        # flag it explicitly rather than leaving "this call
                        # happened inside a tool" recoverable only from
                        # tree indentation or a manual parent_id read
                        # (#5665: the exact schema-leak-triggering shape).
                        span.set_attribute("llm.nested_in_tool", True)
                with self._lock:
                    self._spans[run_key] = span
                return span

            def _flag_if_long_running(self, span: Span) -> None:
                """Best-effort: mark *span* with
                ``span.exceeded_long_running_threshold=true`` if its measured
                open duration crosses ``self._long_span_threshold_secs``
                (disabled entirely when that's None — the default).

                Checked at close time rather than via a background timer —
                agent-trace has no in-flight monitoring thread for spans, so
                this surfaces the same information (a span that ran longer
                than a known-risky platform duration, e.g. LangGraph Cloud's
                ~180s checkpoint-sweep re-dispatch window, issue #7417) the
                one place this callback-driven architecture actually can:
                once the span closes and its true duration is known.
                """
                if self._long_span_threshold_secs is None:
                    return
                try:
                    elapsed = get_time() - span.start_time
                    span.set_attribute(
                        "span.duration_secs_at_close", round(elapsed, 3)
                    )
                    if elapsed >= self._long_span_threshold_secs:
                        span.set_attribute(
                            "span.exceeded_long_running_threshold", True
                        )
                        span.set_attribute(
                            "span.long_running_threshold_secs",
                            self._long_span_threshold_secs,
                        )
                except Exception:
                    logger.debug(
                        "agent-trace: failed to evaluate long-running-span "
                        "threshold for span %r",
                        span.name,
                        exc_info=True,
                    )

            def _close_span(
                self,
                run_id: uuid.UUID | str,
                status: SpanStatus = SpanStatus.OK,
            ) -> Span | None:
                """End the span for *run_id* and remove it from the registry."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.pop(run_key, None)
                    self._stream_token_counts.pop(run_key, None)
                if span is not None and span.end_time is None:
                    self._flag_if_long_running(span)
                    span.end(status)
                return span

            def _close_span_with_exception(
                self,
                run_id: uuid.UUID | str,
                error: BaseException,
            ) -> None:
                """Pop the span and close it according to what *error* means.

                Three distinct outcomes, checked in order:
                  1. LangGraph's own internal control-flow signal (a
                     Command/ParentCommand handoff jump or a GraphInterrupt
                     pause) -> span closes OK with an informational
                     attribute, not ERROR — it isn't an application failure.
                  2. asyncio.CancelledError -> span closes CANCELLED, kept
                     distinct from ERROR so a reader can tell "this failed"
                     apart from "this was cut off mid-flight".
                  3. Anything else -> genuine error: record the exception,
                     classify its origin/known pattern, close ERROR.

                Consolidates the three error callbacks into a single lock
                acquisition + record + end sequence.
                """
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.pop(run_key, None)
                    self._stream_token_counts.pop(run_key, None)
                if span is None:
                    return

                if _is_langgraph_control_flow_signal(error):
                    _record_control_flow_signal(span, error)
                    if span.end_time is None:
                        self._flag_if_long_running(span)
                        span.end(SpanStatus.OK)
                    return

                if isinstance(error, asyncio.CancelledError):
                    span.record_exception(error, status=SpanStatus.CANCELLED)
                    if span.end_time is None:
                        self._flag_if_long_running(span)
                        span.end(SpanStatus.CANCELLED)
                    return

                span.record_exception(error)
                _classify_and_tag_exception(span, error)
                if span.end_time is None:
                    self._flag_if_long_running(span)
                    span.end(SpanStatus.ERROR)

            # ------------------------------------------------------------------
            # Chain (graph node) callbacks
            # ------------------------------------------------------------------

            def on_chain_start(
                self,
                serialized: dict[str, Any],
                inputs: dict[str, Any],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                """Start a span when a graph node (chain) begins execution."""
                # LangGraph 1.x passes serialized=None; node name is in kwargs['name'].
                ser = serialized or {}
                node_name: str = (
                    kwargs.get("name")
                    or ser.get("name")
                    or (ser.get("id") or [None])[-1]
                    or "chain"
                )
                span = self._open_span(run_id, f"node:{node_name}", parent_run_id)
                span.set_attribute("langgraph.node", node_name)
                if tags:
                    span.set_attribute("langgraph.tags", ",".join(tags))
                if self._graph is not None:
                    declared_tags = _get_declared_node_tags(self._graph, node_name)
                    if declared_tags:
                        span.set_attribute(
                            "langgraph.declared_tags", ",".join(declared_tags)
                        )
                if metadata:
                    span.set_attribute("chain.metadata", _to_attr_string(metadata))
                if inputs:
                    span.set_attribute("chain.inputs", _to_attr_string(inputs))
                # Best-effort Runtime/context capture — see
                # _install_runtime_capture_patch(). Deliberately kept out of
                # the "langgraph." attribute namespace since the Runtime
                # object (store/writer/context) is not guaranteed to
                # serialize identically across separate invocations of the
                # same graph (e.g. record vs. replay).
                runtime = _current_runtime.get()
                if runtime is not None:
                    span.set_attribute("chain.runtime", _to_attr_string(runtime))

            def on_chain_end(
                self,
                outputs: dict[str, Any],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the span when a graph node completes successfully."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None and outputs:
                    try:
                        span.set_attribute("chain.outputs", _to_attr_string(outputs))
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record chain outputs for run %r",
                            str(run_id),
                            exc_info=True,
                        )
                self._close_span(run_id, SpanStatus.OK)

            def on_chain_error(
                self,
                error: BaseException,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the span with ERROR status when a graph node raises."""
                self._close_span_with_exception(run_id, error)

            # ------------------------------------------------------------------
            # LLM callbacks
            # ------------------------------------------------------------------

            def on_llm_start(
                self,
                serialized: dict[str, Any],
                prompts: list[str],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                """Start a span for a legacy LLM call, recording the model name."""
                ser = serialized or {}
                model_name: str = (
                    kwargs.get("name")
                    or (ser.get("kwargs") or {}).get("model_name")
                    or (ser.get("kwargs") or {}).get("model")
                    or ser.get("name")
                    or "llm"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", model_name)
                span.set_attribute("llm.prompt_count", len(prompts))
                if metadata:
                    span.set_attribute("llm.metadata", _to_attr_string(metadata))

            def on_chat_model_start(
                self,
                serialized: dict[str, Any],
                messages: list[list[Any]],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                """Start a span for a ChatModel call (e.g. ChatOpenAI, ChatAnthropic).

                Modern LangChain chat models fire ``on_chat_model_start`` instead
                of ``on_llm_start``.  Without this handler those spans are silently
                dropped.
                """
                ser = serialized or {}
                model_name: str = (
                    kwargs.get("name")
                    or (ser.get("kwargs") or {}).get("model_name")
                    or (ser.get("kwargs") or {}).get("model")
                    or ser.get("name")
                    or "unknown"
                )
                span = self._open_span(run_id, f"llm:{model_name}", parent_run_id)
                span.set_attribute("llm.model", model_name)
                if messages:
                    span.set_attribute("llm.messages", _to_attr_string(messages))
                if metadata:
                    span.set_attribute("llm.metadata", _to_attr_string(metadata))

            def on_llm_new_token(
                self,
                token: str,
                *,
                chunk: Any = None,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """Record a per-token/per-delta streaming chunk instead of
                discarding it.

                This is the one real streaming hook the current
                ``langchain_core`` ``BaseCallbackHandler`` interface exposes
                (confirmed via direct inspection of the installed
                langchain-core): it fires for both legacy ``LLM.stream()``
                calls and modern chat-model streaming alike — LangChain
                routes both through this single callback, passing a
                ``GenerationChunk`` or ``ChatGenerationChunk`` via *chunk*
                depending on which. There is no separate
                ``on_chat_model_stream`` method on the base handler to
                implement.

                Bounded: after _MAX_STREAM_EVENTS_PER_SPAN tokens, further
                deltas stop generating new SpanEvents (a long stream would
                otherwise grow the span unboundedly) but
                ``llm.stream_token_count`` keeps counting every token that
                arrived.
                """
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                    count = self._stream_token_counts.get(run_key, 0) + 1
                    self._stream_token_counts[run_key] = count
                if span is None:
                    return
                try:
                    span.set_attribute("llm.streamed", True)
                    span.set_attribute("llm.stream_token_count", count)
                    if count <= _MAX_STREAM_EVENTS_PER_SPAN:
                        attrs: dict[str, Any] = {"stream.index": count - 1}
                        if token:
                            attrs["token"] = _stringify(token, max_len=2000)
                        tool_call_chunks = _extract_tool_call_chunks(chunk)
                        if tool_call_chunks:
                            attrs["tool_call_chunks"] = _to_attr_string(
                                tool_call_chunks, max_len=2000
                            )
                        span.add_event("llm_stream_delta", attributes=attrs)
                except Exception:
                    logger.debug(
                        "agent-trace: failed to record streaming token for "
                        "run %r",
                        str(run_id),
                        exc_info=True,
                    )

            def on_llm_end(
                self,
                response: Any,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the LLM span, attaching token usage and response content
                when available."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None:
                    try:
                        _record_llm_end_data(span, response)
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record LLM response data "
                            "for run %r",
                            str(run_id),
                            exc_info=True,
                        )
                self._close_span(run_id, SpanStatus.OK)

            def on_llm_error(
                self,
                error: BaseException,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the LLM span with ERROR status."""
                self._close_span_with_exception(run_id, error)

            # ------------------------------------------------------------------
            # Tool callbacks
            # ------------------------------------------------------------------

            def on_tool_start(
                self,
                serialized: dict[str, Any],
                input_str: str,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                """Start a span when a tool begins execution."""
                ser = serialized or {}
                tool_name: str = kwargs.get("name") or ser.get("name") or "tool"
                span = self._open_span(run_id, f"tool:{tool_name}", parent_run_id)
                span.set_attribute("tool.name", tool_name)
                if input_str:
                    span.set_attribute("tool.input", _stringify(input_str))
                if metadata:
                    span.set_attribute("tool.metadata", _to_attr_string(metadata))
                # Threading/event-loop context — the exact structured data
                # needed to diagnose a sync tool dispatched into a
                # ThreadPoolExecutor with no event loop (RuntimeError: "There
                # is no current event loop in thread ...").
                span.set_attribute("tool.thread_name", threading.current_thread().name)
                try:
                    asyncio.get_running_loop()
                    span.set_attribute("tool.has_event_loop", True)
                except RuntimeError:
                    span.set_attribute("tool.has_event_loop", False)

            def on_tool_end(
                self,
                output: str,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the tool span with OK status, recording output text."""
                run_key = str(run_id)
                with self._lock:
                    span = self._spans.get(run_key)
                if span is not None and output is not None:
                    try:
                        span.set_attribute("tool.output", _stringify(output))
                    except Exception:
                        logger.debug(
                            "agent-trace: failed to record tool output for run %r",
                            str(run_id),
                            exc_info=True,
                        )
                self._close_span(run_id, SpanStatus.OK)

            def on_tool_error(
                self,
                error: BaseException,
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                **kwargs: Any,
            ) -> None:
                """End the tool span with ERROR status."""
                self._close_span_with_exception(run_id, error)

        _LangGraphTracerClass = _LangGraphTracerImpl
        return _LangGraphTracerClass


# ---------------------------------------------------------------------------
# traced_stream / traced_astream — stream-yield timestamp + content capture
# ---------------------------------------------------------------------------
#
# LangGraphTracer only implements the standard LangChain callback pairs
# (on_chain_start/end, on_llm_start/end, on_tool_start/end, ...), none of
# which fire on "a value was actually yielded from the graph's own
# .stream()/.astream() iterator to the caller's code". That boundary matters:
# graph.invoke(state, stream_mode=...) fully drains the generator internally
# before returning (Pregel.invoke() loops `for chunk in self.stream(...)`
# and only returns once exhausted) while graph.stream(...) yields
# progressively — two very different externally-observed delivery timings
# that look identical in a trace with only callback-derived spans, since
# every chain/llm/tool span closes at the same internal moment either way.
#
# traced_stream()/traced_astream() wrap *any* iterable/async-iterable
# (typically the return value of graph.stream(...)/graph.astream(...), with
# whatever stream_mode was requested) in a dedicated span, recording a
# SpanEvent — timestamped on the same clock as every other span
# (core.clock.get_time(), via Span.add_event) — at the exact moment each
# item is yielded back to the calling code, plus a bounded, serialized copy
# of the chunk's own content. This also directly captures stream_mode's
# actual per-chunk output (messages/updates/values, whichever mode was
# requested) onto the trace, not just its timing.


def _record_stream_yield(span: Span, index: int, item: Any) -> None:
    """Append one bounded stream_yield SpanEvent, capped at
    _MAX_STREAM_EVENTS_PER_SPAN so an unbounded stream can't grow a single
    span's event list without limit."""
    if index >= _MAX_STREAM_EVENTS_PER_SPAN:
        return
    span.add_event(
        "stream_yield",
        attributes={
            "stream.index": index,
            "stream.chunk": _stringify(item, max_len=2000),
        },
    )


def traced_stream(
    tracer: Tracer,
    stream: Iterable[Any],
    *,
    span_name: str = "graph:stream",
) -> Generator[Any, None, None]:
    """Wrap a LangGraph graph's ``.stream()`` iterator (or any other
    iterable) so each item's yield moment — and a bounded copy of its
    content — lands on the trace timeline.

    Usage::

        for chunk in traced_stream(tracer, graph.stream(state,
                                                          stream_mode="messages")):
            ...

    Opens one ``graph:stream`` span (customizable via *span_name*) that
    stays open for the lifetime of the iteration, closing OK once the
    source stream is exhausted, ERROR if the source stream itself raises
    (the exception is recorded onto the span then re-raised unchanged), or
    CANCELLED if the caller stops iterating early (e.g. a ``break``) and
    this generator is garbage-collected/closed before exhaustion.
    """
    span = tracer.start_span(span_name)
    index = 0
    status = SpanStatus.OK
    try:
        for item in stream:
            _record_stream_yield(span, index, item)
            index += 1
            yield item
    except GeneratorExit:
        status = SpanStatus.CANCELLED
        raise
    except Exception as exc:
        status = SpanStatus.ERROR
        span.record_exception(exc)
        raise
    finally:
        span.set_attribute("stream.chunk_count", index)
        if span.end_time is None:
            span.end(status)


async def traced_astream(
    tracer: Tracer,
    stream: AsyncIterable[Any],
    *,
    span_name: str = "graph:astream",
) -> AsyncGenerator[Any, None]:
    """Async equivalent of :func:`traced_stream` — wraps
    ``graph.astream(...)`` (or any other async iterable) the same way."""
    span = tracer.start_span(span_name)
    index = 0
    status = SpanStatus.OK
    try:
        async for item in stream:
            _record_stream_yield(span, index, item)
            index += 1
            yield item
    except GeneratorExit:
        status = SpanStatus.CANCELLED
        raise
    except Exception as exc:
        status = SpanStatus.ERROR
        span.record_exception(exc)
        raise
    finally:
        span.set_attribute("stream.chunk_count", index)
        if span.end_time is None:
            span.end(status)


class LangGraphTracer:
    """Langchain/LangGraph callback handler that emits agent-trace spans.

    Implements the ``BaseCallbackHandler`` interface from ``langchain_core``
    so it can be passed directly in the ``config["callbacks"]`` list of any
    LangGraph graph invocation.

    ``langchain_core`` is imported lazily — importing this module succeeds even
    when ``langchain_core`` is not installed.  The import (and the class
    definition that inherits from ``BaseCallbackHandler``) happens once, the
    first time a ``LangGraphTracer`` instance is created.

    Parameters
    ----------
    tracer:
        The active :class:`~agent_trace.Tracer` instance.
    trace:
        The :class:`~agent_trace.Trace` that spans will be registered on.
    graph:
        Optional: the compiled LangGraph graph this tracer instruments.
        When supplied, node spans additionally carry a
        ``langgraph.declared_tags`` attribute — each node's
        graph-construction-time declared tags (e.g. from
        ``.with_config(tags=["nostream"])``), which the runtime callback
        ``tags`` kwarg alone never exposes. Omit (the default, None) to keep
        the pre-existing behavior with no declared-tags lookup.
    long_span_threshold_secs:
        Optional: once a span (node/llm/tool) closes having been open for at
        least this many seconds, it is flagged with
        ``span.exceeded_long_running_threshold=true`` and
        ``span.duration_secs_at_close``. Useful for platform-level behaviors
        keyed off wall-clock duration — e.g. LangGraph Cloud's ~180s
        checkpoint-sweep re-dispatch window (issue #7417). Omit (the
        default, None) to disable the check entirely.
    """

    def __new__(
        cls,
        tracer: Tracer,
        trace: Trace,
        *,
        graph: Any = None,
        long_span_threshold_secs: float | None = None,
    ) -> LangGraphTracer:
        # Construct the concrete impl directly so Python's normal type.__call__
        # runs _LangGraphTracerImpl.__init__ automatically.  We cannot use
        # impl_cls.__new__(impl_cls) + manual __init__ because the returned
        # object would not be an instance of LangGraphTracer, causing Python
        # to skip __init__ entirely — leaving _tracer/_trace/_spans unset.
        impl_cls = _get_tracer_class()
        return impl_cls(  # type: ignore[no-any-return]
            tracer,
            trace,
            graph=graph,
            long_span_threshold_secs=long_span_threshold_secs,
        )

    def __init__(
        self,
        tracer: Tracer,
        trace: Trace,
        *,
        graph: Any = None,
        long_span_threshold_secs: float | None = None,
    ) -> None:
        # __init__ is called on the instance whose __class__ is already the
        # concrete impl class (set by __new__).  Delegate to its __init__.
        # This path is only reached if someone subclasses LangGraphTracer
        # directly; normal construction goes through the impl class __init__.
        pass  # pragma: no cover
