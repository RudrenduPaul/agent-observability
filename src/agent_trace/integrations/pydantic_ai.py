"""
pydantic-ai integration — graph-aware span enrichment for Agent runs.

pydantic-ai does not expose a callback/hook-registration surface the way
LangChain's ``BaseCallbackHandler`` or the openai-agents SDK's ``RunHooks``
do.  Its own documented instrumentation point is ``Agent.iter()``, an async
context manager that yields the underlying pydantic-graph nodes
(``UserPromptNode`` -> ``ModelRequestNode`` -> ``CallToolsNode`` -> ...  ->
``End``) one at a time as the run advances — the same object graph
``Agent.run()``/``Agent.run_sync()`` build and drive internally.

This module wraps that iterator: it walks the node sequence, opening an
``llm:<model>`` span for each model request/response round trip and a
``tool:<name>`` span for each tool call, and tags a request as a retry
attempt when pydantic-ai's own ``ModelRetry``/output-validator retry
mechanism appends a ``RetryPromptPart`` to the next request — the exact
"was this exchange a retry, and of what" attribution the generic httpx
interceptor cannot supply because it sees only anonymous raw HTTP bodies.

Usage (drain to completion)::

    from agent_trace import Tracer
    from agent_trace.integrations.pydantic_ai import run_traced

    t = Tracer()
    with t.start_trace("my_agent_run", record=True) as trace:
        result = await run_traced(agent, "hello", tracer=t, trace=trace)

Usage (step through nodes yourself, e.g. to inspect intermediate state)::

    from agent_trace.integrations.pydantic_ai import instrument_agent_run

    with t.start_trace("my_agent_run", record=True) as trace:
        async with instrument_agent_run(agent, "hello", tracer=t, trace=trace) as run:
            async for node in run:
                ...
            result = run.result

Known limitation — ``pydantic_evals.evaluators.llm_as_a_judge.judge_output()``
(and its siblings ``judge_output_expected``/``judge_input_output``/
``judge_g_eval``) construct and run their own internal ``Agent`` instance
that is not exposed for a caller to wrap.  Attributing a captured HTTP
exchange to "this was a judge call" therefore requires either running your
own judge through a manually-constructed ``Agent`` wrapped with
``instrument_agent_run``, or monkeypatching pydantic_evals internals, which
this module deliberately does not do.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    from agent_trace import Trace, Tracer

__all__ = [
    "TracedAgentRun",
    "instrument_agent_run",
    "run_traced",
]

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The pydantic-ai integration requires the pydantic-ai package.\n"
    "Install it with:\n\n"
    "    pip install pydantic-ai\n"
)


def _require_pydantic_ai() -> Any:
    """Lazy import guard — raises a clear error if pydantic-ai is absent."""
    try:
        import pydantic_ai

        return pydantic_ai
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _model_display_name(model: Any) -> str:
    """Best-effort model identifier for a pydantic-ai ``Model`` instance/string.

    Used only as the *initial* span name/attribute before the real request
    completes — ``ModelResponse.model_name`` (captured in
    ``TracedAgentRun._close_llm_span``) is always the authoritative value
    once a response has actually arrived.
    """
    if model is None:
        return "unknown"
    if isinstance(model, str):
        return model
    name = getattr(model, "model_name", None)
    if name:
        return str(name)
    return type(model).__name__


# ---------------------------------------------------------------------------
# TracedAgentRun
# ---------------------------------------------------------------------------


class TracedAgentRun:
    """Wraps a pydantic-ai ``AgentRun`` and emits agent-trace spans as the
    agent graph advances node-by-node.

    Not instantiated directly — obtained from :func:`instrument_agent_run`.

    Span tree produced for one ``agent.iter()`` run::

        agent:<name>
          llm:<model>              (one per ModelRequestNode -> next-node transition)
          tool:<tool_name>         (one per ToolCallPart on a model response)
          llm:<model>              (retry / next turn, if any)
          ...

    A model-request span tagged ``llm.is_retry=True`` means pydantic-ai
    appended a ``RetryPromptPart`` to that request — i.e. this call is a
    retry produced by a raised ``ModelRetry`` (from a tool implementation or
    an ``@agent.output_validator``), not a fresh turn.
    """

    def __init__(
        self,
        run: Any,
        *,
        tracer: Tracer,
        trace: Trace,
        root_span: Span,
        agent_name: str,
        default_model_name: str,
    ) -> None:
        self._run = run
        self._tracer: Tracer = tracer
        self._trace: Trace = trace
        self._root_span = root_span
        self._agent_name = agent_name
        self._default_model_name = default_model_name
        self._llm_span: Span | None = None
        self._tool_spans: dict[str, Span] = {}
        self._retry_count = 0

    # ------------------------------------------------------------------
    # Public accessors — mirror the underlying AgentRun
    # ------------------------------------------------------------------

    @property
    def result(self) -> Any:
        """The pydantic-ai ``AgentRunResult`` once the run has completed."""
        return self._run.result

    @property
    def run_usage(self) -> Any:
        """The pydantic-ai ``RunUsage`` accumulated by the run so far."""
        return self._run.usage

    # ------------------------------------------------------------------
    # Async iteration — transparent pass-through of the wrapped AgentRun
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        pydantic_ai_mod = _require_pydantic_ai()
        agent_cls = pydantic_ai_mod.Agent

        async for node in self._run:
            self._on_node(agent_cls, node)
            yield node

        self._finalize()

    # ------------------------------------------------------------------
    # Node-transition handling
    # ------------------------------------------------------------------

    def _on_node(self, agent_cls: Any, node: Any) -> None:
        try:
            if agent_cls.is_model_request_node(node):
                request = node.request
                # Resolve any tool spans opened for the previous CallToolsNode
                # using this request's ToolReturnPart/RetryPromptPart entries
                # before opening the next llm span.
                self._close_tool_spans(request)
                self._open_llm_span(request)
            elif agent_cls.is_call_tools_node(node):
                model_response = node.model_response
                self._close_llm_span(model_response)
                self._open_tool_spans(model_response)
        except Exception:
            logger.debug(
                "agent-trace: failed to process pydantic-ai node %r",
                type(node).__name__,
                exc_info=True,
            )

    def _open_llm_span(self, request: Any) -> None:
        span = self._tracer.start_span(
            f"llm:{self._default_model_name}", parent_id=self._root_span.span_id
        )
        span.set_attribute("llm.model", self._default_model_name)
        span.set_attribute("agent.name", self._agent_name)

        parts = list(getattr(request, "parts", []) or [])
        span.set_attribute("llm.request_message_count", len(parts))

        # Which ModelRequestPart types were actually sent — specifically
        # whether a SystemPromptPart was included vs. silently dropped
        # (#3277: a developer had no way to tell, from a captured run,
        # whether their system prompt/instructions actually made it into a
        # given request without manually diffing two runs by hand).
        part_kinds = sorted({type(p).__name__ for p in parts})
        if part_kinds:
            span.set_attribute("llm.request_part_kinds", ",".join(part_kinds))
        system_prompt_parts = [
            p for p in parts if type(p).__name__ == "SystemPromptPart"
        ]
        span.set_attribute("llm.has_system_prompt_part", bool(system_prompt_parts))
        if system_prompt_parts:
            resolved = getattr(system_prompt_parts[0], "content", None)
            if isinstance(resolved, str) and resolved:
                truncated = "...<truncated>" if len(resolved) > 2000 else ""
                span.set_attribute(
                    "llm.system_prompt_content", resolved[:2000] + truncated
                )

        retry_parts = [p for p in parts if type(p).__name__ == "RetryPromptPart"]
        if retry_parts:
            self._retry_count += 1
            span.set_attribute("llm.is_retry", True)
            span.set_attribute("llm.retry_index", self._retry_count)
            tool_names: list[str] = sorted(
                {
                    str(getattr(p, "tool_name", None))
                    for p in retry_parts
                    if getattr(p, "tool_name", None)
                }
            )
            if tool_names:
                span.set_attribute("llm.retry_tool_name", ",".join(tool_names))
            else:
                span.set_attribute("llm.retry_reason", "output_validator")

        self._llm_span = span

    def _close_llm_span(self, model_response: Any) -> None:
        span = self._llm_span
        self._llm_span = None
        if span is None:
            return

        model_name = getattr(model_response, "model_name", None)
        if model_name:
            # The concrete model wasn't knowable until the response arrived
            # (e.g. fallback models, per-call overrides) — correct both the
            # span name and attribute now that we have ground truth.
            span.name = f"llm:{model_name}"
            span.set_attribute("llm.model", str(model_name))

        provider_name = getattr(model_response, "provider_name", None)
        if provider_name:
            span.set_attribute("llm.provider", str(provider_name))

        finish_reason = getattr(model_response, "finish_reason", None)
        if finish_reason:
            span.set_attribute("llm.finish_reason", str(finish_reason))

        usage = getattr(model_response, "usage", None)
        if usage is not None:
            for attr_name, field_name in (
                ("llm.usage.prompt_tokens", "input_tokens"),
                ("llm.usage.completion_tokens", "output_tokens"),
                ("llm.usage.cache_read_tokens", "cache_read_tokens"),
                ("llm.usage.cache_write_tokens", "cache_write_tokens"),
            ):
                value = getattr(usage, field_name, None)
                if value:
                    span.set_attribute(attr_name, int(value))

        tool_call_count = sum(
            1
            for part in getattr(model_response, "parts", []) or []
            if type(part).__name__ == "ToolCallPart"
        )
        span.set_attribute("llm.tool_call_count", tool_call_count)

        span.end(SpanStatus.OK)

    def _open_tool_spans(self, model_response: Any) -> None:
        for part in getattr(model_response, "parts", []) or []:
            if type(part).__name__ != "ToolCallPart":
                continue
            tool_name = getattr(part, "tool_name", None) or "tool"
            call_id = (
                getattr(part, "tool_call_id", None)
                or f"{tool_name}:{len(self._tool_spans)}"
            )
            span = self._tracer.start_span(
                f"tool:{tool_name}", parent_id=self._root_span.span_id
            )
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.call_id", str(call_id))
            args = getattr(part, "args", None)
            if isinstance(args, dict):
                span.set_attribute("tool.arg_count", len(args))
            self._tool_spans[str(call_id)] = span

    def _close_tool_spans(self, request: Any) -> None:
        for part in getattr(request, "parts", []) or []:
            call_id = getattr(part, "tool_call_id", None)
            if call_id is None:
                continue
            span = self._tool_spans.pop(str(call_id), None)
            if span is None:
                continue

            part_type = type(part).__name__
            if part_type == "RetryPromptPart":
                # The tool implementation raised ModelRetry — pydantic-ai's
                # own soft-retry control-flow signal, not an application
                # failure. Mirrors agent-trace's LangGraph handling of
                # Command/GraphInterrupt: close OK, tag as retried rather
                # than marking the span ERROR.
                span.set_attribute("tool.retried", True)
                span.end(SpanStatus.OK)
            else:
                content = getattr(part, "content", None)
                span.set_attribute(
                    "tool.output_length",
                    len(str(content)) if content is not None else 0,
                )
                span.end(SpanStatus.OK)

    def _finalize(self) -> None:
        """Defensive cleanup after a normal (non-exception) run completion.

        In the ordinary case both ``_llm_span`` and ``_tool_spans`` are
        already empty by the time the graph reaches its ``End`` node — this
        only fires if a node shape agent-trace doesn't recognise leaves
        something dangling.
        """
        if self._llm_span is not None and self._llm_span.end_time is None:
            self._llm_span.end(SpanStatus.OK)
        self._llm_span = None
        for span in self._tool_spans.values():
            if span.end_time is None:
                span.end(SpanStatus.OK)
        self._tool_spans.clear()

        self._record_final_usage()

    def _record_final_usage(self) -> None:
        usage = self._run.usage
        if usage is None:
            return
        requests = getattr(usage, "requests", None)
        if requests is not None:
            self._root_span.set_attribute("agent.usage.requests", int(requests))
        for attr_name, field_name in (
            ("agent.usage.input_tokens", "input_tokens"),
            ("agent.usage.output_tokens", "output_tokens"),
        ):
            value = getattr(usage, field_name, None)
            if value:
                self._root_span.set_attribute(attr_name, int(value))
        if self._retry_count:
            self._root_span.set_attribute("agent.retry_count", self._retry_count)

    def _close_dangling_spans(self, exc: BaseException) -> None:
        """Close any still-open child spans as ERROR when the run raises."""
        if self._llm_span is not None and self._llm_span.end_time is None:
            self._llm_span.record_exception(exc)
            self._llm_span.end(SpanStatus.ERROR)
        self._llm_span = None
        for span in self._tool_spans.values():
            if span.end_time is None:
                span.record_exception(exc)
                span.end(SpanStatus.ERROR)
        self._tool_spans.clear()


# ---------------------------------------------------------------------------
# instrument_agent_run / run_traced — public entry points
# ---------------------------------------------------------------------------


@asynccontextmanager
async def instrument_agent_run(
    agent: Any,
    user_prompt: Any = None,
    *,
    tracer: Tracer,
    trace: Trace,
    **kwargs: Any,
) -> AsyncIterator[TracedAgentRun]:
    """Run a pydantic-ai ``Agent`` through ``Agent.iter()`` with span instrumentation.

    This is the graph-aware equivalent of ``async with agent.iter(...) as run``
    — every model request/response round trip and tool call along the way is
    captured as an agent-trace span, parented under a root ``agent:<name>``
    span, instead of relying solely on the generic httpx monkeypatch to
    capture anonymous raw HTTP rows with no pydantic-ai-level context.

    Parameters
    ----------
    agent:
        A pydantic-ai ``Agent`` instance.
    user_prompt:
        Forwarded to ``Agent.iter()`` — the user prompt (or ``None`` when
        driving the run entirely from ``message_history``).
    tracer:
        The active :class:`~agent_trace.Tracer`.
    trace:
        The current :class:`~agent_trace.Trace` that spans will be registered
        on.
    **kwargs:
        Additional keyword arguments forwarded to ``Agent.iter()`` (e.g.
        ``deps``, ``model``, ``message_history``, ``usage_limits``).

    Yields
    ------
    TracedAgentRun
        An async-iterable wrapper around the underlying ``AgentRun`` — iterate
        it exactly as you would ``agent.iter()``'s own result, then read
        ``.result`` for the final ``AgentRunResult``.
    """
    _require_pydantic_ai()
    agent_name: str = getattr(agent, "name", None) or "agent"
    default_model_name = _model_display_name(
        kwargs.get("model") or getattr(agent, "model", None)
    )

    root_span = tracer.start_span(f"agent:{agent_name}")
    root_span.set_attribute("agent.name", agent_name)
    root_span.set_attribute("agent.model", default_model_name)

    traced: TracedAgentRun | None = None
    try:
        async with agent.iter(user_prompt, **kwargs) as run:
            traced = TracedAgentRun(
                run,
                tracer=tracer,
                trace=trace,
                root_span=root_span,
                agent_name=agent_name,
                default_model_name=default_model_name,
            )
            yield traced
        root_span.end(SpanStatus.OK)
    except BaseException as exc:
        if traced is not None:
            traced._close_dangling_spans(exc)
        root_span.record_exception(exc)
        if root_span.end_time is None:
            root_span.end(SpanStatus.ERROR)
        raise


async def run_traced(
    agent: Any,
    user_prompt: Any = None,
    *,
    tracer: Tracer,
    trace: Trace,
    **kwargs: Any,
) -> Any:
    """Run a pydantic-ai ``Agent`` to completion with span instrumentation.

    Convenience wrapper around :func:`instrument_agent_run` that drains the
    node iterator and returns the underlying ``AgentRunResult`` — equivalent
    to ``await agent.run(user_prompt, **kwargs)`` but with every model
    request and tool call captured as a span.

    Parameters
    ----------
    agent:
        A pydantic-ai ``Agent`` instance.
    user_prompt:
        Forwarded to ``Agent.iter()``.
    tracer:
        The active :class:`~agent_trace.Tracer`.
    trace:
        The current :class:`~agent_trace.Trace`.
    **kwargs:
        Additional keyword arguments forwarded to ``Agent.iter()``.

    Returns
    -------
    Any
        The ``AgentRunResult`` from the completed run.
    """
    async with instrument_agent_run(
        agent, user_prompt, tracer=tracer, trace=trace, **kwargs
    ) as traced_run:
        async for _node in traced_run:
            pass
        return traced_run.result
