"""
AutoGen integration for the modern autogen-agentchat / autogen-ext
architecture (v0.4+/v0.7.x).

AutoGen's agentchat layer has no LangChain-style pluggable callback list, so
this integration works by wrapping the async-generator methods that already
exist on agents, teams, and code executors (``on_messages_stream``,
``run_stream``, ``execute_code_blocks``) at the *instance* level.  Because
``on_messages``/``run`` internally call ``self.on_messages_stream``/
``self.run_stream`` (attribute lookup resolves to the instance override
first), wrapping the stream method transparently covers both the streaming
and non-streaming call paths with a single patch point.

Provides
--------
``instrument_agent(agent, tracer, trace)``
    Wraps a single ``BaseChatAgent`` (e.g. ``AssistantAgent``,
    ``CodeExecutorAgent``) so every turn gets an ``agent:<name>`` span
    tagged with the agent's name -- so a captured LLM/tool exchange can be
    attributed to which agent in a multi-agent team produced it -- plus
    tool-call, handoff, thought, and token-usage span events.  Any
    exception raised anywhere during the turn (including before any HTTP
    request is ever built, e.g. a tool-schema conversion ``TypeError``) is
    recorded on the span and the span is closed ``ERROR`` before the
    exception propagates.

``instrument_team(team, tracer, trace)``
    Wraps a ``BaseGroupChat`` (``SelectorGroupChat``,
    ``RoundRobinGroupChat``, ...): recursively instruments every
    participant (agent or nested team) via ``instrument_agent``/
    ``instrument_team``, and wraps the team's own ``run_stream`` for a root
    span that records routing/speaker-selection events as they occur.

``instrument_code_executor(executor, tracer, trace)``
    Wraps ``CodeExecutor.execute_code_blocks`` (e.g.
    ``LocalCommandLineCodeExecutor``) to record the executed code, working
    directory, combined stdout+stderr output (AutoGen's own
    ``CodeResult``/``CommandLineCodeResult`` merges stdout and stderr into
    a single ``output`` field -- there is no separate stdout/stderr split
    anywhere in the executor's return type), and exit code as a span
    event.  This is independent of, and in addition to, any LLM HTTP
    capture: code execution never touches HTTP, so the httpx/requests
    interceptor has zero visibility into it regardless of how well it is
    wired up.

``recording_http_client(fixture, is_async=True, inner=None)``
    Builds an httpx client pre-wired with agent-trace's
    ``RecordingTransport``/``AsyncRecordingTransport``, for passing as the
    ``http_client=`` kwarg into autogen-ext's ``OpenAIChatCompletionClient``/
    ``AzureOpenAIChatCompletionClient`` (confirmed: both forward any kwarg
    matching ``AsyncOpenAI.__init__``'s keyword-only args -- including
    ``http_client`` -- straight through, via
    ``autogen_ext/models/openai/_openai_client.py``'s
    ``openai_init_kwargs = set(inspect.getfullargspec(``
    ``AsyncOpenAI.__init__).kwonlyargs)``)
    or into legacy AutoGen 0.2's ``config_list`` dicts (same passthrough
    mechanism, via ``OpenAIWrapper.openai_kwargs`` in ``autogen/oai/client.py``).

    Note this is usually *not required* for the modern v0.4+ path: any
    ``httpx.AsyncClient`` constructed with no explicit ``transport=`` kwarg
    *after* entering ``tracer.start_trace(..., record=True)`` is already
    captured automatically by agent-trace's global ``httpx.AsyncClient``
    patch (confirmed: OpenAI's own ``AsyncHttpxClientWrapper.__init__`` only
    ever calls ``kwargs.setdefault(...)`` for timeout/limits/redirects, never
    setting ``transport`` itself, so agent-trace's own ``setdefault`` fires).
    Use ``recording_http_client`` when you want an explicit, documented,
    zero-surprise wiring path instead of relying on that global patch -- most
    importantly for legacy AutoGen 0.2's ``config_list``, which requires an
    actual client object as a dict value rather than relying on any global
    state.

    Documented caveat (see module docstring "Known limitations" below):
    this pattern does **not** work for legacy AutoGen 0.2's
    ``GPTAssistantAgent`` specifically, because
    ``_process_assistant_config()`` (``gpt_assistant_agent.py:490``) does
    ``copy.deepcopy(llm_config)`` before constructing the client, and a live
    ``httpx.Client``/``httpx.AsyncClient`` instance is not deep-copyable
    (``TypeError: cannot pickle '_thread.RLock' object``).  For
    ``GPTAssistantAgent``, use agent-trace's own
    ``Tracer(...).start_trace(record=True)`` context manager instead --
    ``GPTAssistantAgent``'s underlying ``openai.OpenAI()`` client never
    passes an explicit ``transport=`` kwarg, so the global patch captures it
    with zero AutoGen-specific wiring at all.

Usage
-----
Agent/message-routing spans::

    from agent_trace import tracer
    from agent_trace.integrations.autogen import instrument_agent

    with tracer.start_trace("my_run", record=True) as trace:
        agent = AssistantAgent("writer", model_client=model_client)
        instrument_agent(agent, tracer=tracer, trace=trace)
        result = await agent.run(task="write a haiku")

Multi-agent team::

    from agent_trace.integrations.autogen import instrument_team

    with tracer.start_trace("my_team_run", record=True) as trace:
        team = SelectorGroupChat([planner, writer], model_client=model_client)
        instrument_team(team, tracer=tracer, trace=trace)
        result = await team.run(task="write a plan and a haiku")

Code-execution capture::

    from agent_trace.integrations.autogen import instrument_code_executor

    executor = LocalCommandLineCodeExecutor(work_dir="./coding")
    instrument_code_executor(executor, tracer=tracer, trace=trace)
    agent = CodeExecutorAgent("executor", code_executor=executor)

Explicit http_client wiring (modern v0.4+/v0.7.x)::

    from agent_trace import Fixture
    from agent_trace.integrations.autogen import recording_http_client

    fixture = Fixture(run_dir / "fixture.db")
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        http_client=recording_http_client(fixture),
    )

Explicit http_client wiring (legacy AutoGen 0.2 config_list)::

    config_list = [{
        "model": "gpt-4o-mini",
        "api_key": "...",
        "http_client": recording_http_client(fixture, is_async=False),
    }]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_trace.core.span import Span, SpanStatus

if TYPE_CHECKING:
    import httpx

    from agent_trace import Trace, Tracer
    from agent_trace._replay.fixture import Fixture

__all__ = [
    "instrument_agent",
    "instrument_code_executor",
    "instrument_team",
    "recording_http_client",
]

logger = logging.getLogger(__name__)

_TRUNCATE = 2000  # max chars persisted per captured code/output field


def _truncate(value: str, limit: int = _TRUNCATE) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated, {len(value)} chars total]"


# ---------------------------------------------------------------------------
# recording_http_client — http_client wiring helper (backlog: "Add a
# documented http_client wiring helper ... so config_list-based frameworks
# route through RecordingTransport")
# ---------------------------------------------------------------------------


def recording_http_client(
    fixture: Fixture,
    *,
    is_async: bool = True,
    inner: Any | None = None,
) -> httpx.Client | httpx.AsyncClient:
    """Build an httpx client wired with agent-trace's RecordingTransport.

    Pass the result as the ``http_client=`` kwarg into
    ``autogen_ext.models.openai.OpenAIChatCompletionClient``/
    ``AzureOpenAIChatCompletionClient`` (modern v0.4+/v0.7.x), or as a
    ``config_list`` dict value's ``"http_client"`` key (legacy AutoGen 0.2).

    Parameters
    ----------
    fixture:
        Open :class:`~agent_trace.Fixture` to record exchanges into.
    is_async:
        Build an ``httpx.AsyncClient`` (the default -- required by
        ``OpenAIChatCompletionClient``/``AsyncOpenAI``) when True, or a
        synchronous ``httpx.Client`` (required by legacy AutoGen 0.2's
        synchronous ``OpenAI`` client) when False.
    inner:
        Optional real transport to forward requests through.  Defaults to
        ``httpx.AsyncHTTPTransport()``/``httpx.HTTPTransport()``.

    Returns
    -------
    httpx.Client | httpx.AsyncClient
        Ready to pass directly as ``http_client=``.

    Not usable for legacy AutoGen 0.2's ``GPTAssistantAgent`` -- see the
    module docstring's "Known limitations" section for why and what to use
    instead.
    """
    import httpx

    from agent_trace.interceptor.httpx_hook import (
        AsyncRecordingTransport,
        RecordingTransport,
    )

    if is_async:
        return httpx.AsyncClient(
            transport=AsyncRecordingTransport(fixture, inner=inner)
        )
    return httpx.Client(transport=RecordingTransport(fixture, inner=inner))


# ---------------------------------------------------------------------------
# Shared span-enrichment helpers
# ---------------------------------------------------------------------------


def _record_usage(span: Span, message: Any) -> None:
    """If *message* carries a non-None ``models_usage`` (RequestUsage),
    accumulate prompt/completion/total token counts onto *span*.

    Every AutoGen agentchat message/event type (``BaseChatMessage``,
    ``BaseAgentEvent``, and ``Response.chat_message``) carries an optional
    ``models_usage: RequestUsage | None`` field -- this is the one place
    token usage surfaces regardless of which concrete message type produced
    it, so a single check here covers every LLM call in the turn without
    needing to hook ``_call_llm`` directly.
    """
    usage = getattr(message, "models_usage", None)
    if usage is None:
        return
    try:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        return
    prior_prompt = int(span.attributes.get("llm.usage.prompt_tokens", 0) or 0)
    prior_completion = int(span.attributes.get("llm.usage.completion_tokens", 0) or 0)
    span.set_attribute("llm.usage.prompt_tokens", prior_prompt + prompt)
    span.set_attribute("llm.usage.completion_tokens", prior_completion + completion)
    span.set_attribute(
        "llm.usage.total_tokens",
        prior_prompt + prompt + prior_completion + completion,
    )


def _record_event(span: Span, event: Any) -> None:
    """Enrich *span* with routing/tool/handoff context from one yielded
    item of ``on_messages_stream``/``run_stream``.

    This is what gives a captured trace agent/message-routing context: a
    tool call, handoff, or code-execution result is tagged with the
    concrete AutoGen event type and the agent (``source``) that produced
    it, rather than being visible only as an undifferentiated raw HTTP body.
    """
    type_name = type(event).__name__
    source = getattr(event, "source", None)

    _record_usage(span, event)
    if type_name == "Response":
        # Response itself never carries models_usage -- only its nested
        # chat_message does (confirmed: Response(chat_message=TextMessage(
        # ..., models_usage=RequestUsage(...)), ...)).
        chat_message = getattr(event, "chat_message", None)
        if chat_message is not None:
            _record_usage(span, chat_message)

    if type_name == "ToolCallRequestEvent":
        calls = getattr(event, "content", None) or []
        names = [getattr(c, "name", "?") for c in calls]
        span.add_event(
            "tool_call_request",
            attributes={"tool.names": ",".join(names), "source": source or ""},
        )
    elif type_name == "ToolCallExecutionEvent":
        results = getattr(event, "content", None) or []
        names = [getattr(r, "name", "?") for r in results]
        any_error = any(bool(getattr(r, "is_error", False)) for r in results)
        span.add_event(
            "tool_call_execution",
            attributes={
                "tool.names": ",".join(names),
                "tool.any_error": any_error,
                "source": source or "",
            },
        )
    elif type_name == "HandoffMessage":
        span.add_event(
            "handoff",
            attributes={
                "handoff.from_agent": source or "",
                "handoff.target": getattr(event, "target", "") or "",
            },
        )
    elif type_name == "ThoughtEvent":
        content = getattr(event, "content", "") or ""
        span.add_event(
            "thought",
            attributes={
                "thought.content": _truncate(str(content)),
                "source": source or "",
            },
        )
    elif type_name in ("CodeGenerationEvent", "CodeExecutionEvent"):
        # Belt-and-suspenders: some agent configurations surface these
        # events directly in the message stream.  The primary, reliable
        # capture path for code-execution results is
        # instrument_code_executor() below (see its docstring for why).
        attrs: dict[str, Any] = {"source": source or ""}
        if type_name == "CodeGenerationEvent":
            blocks = getattr(event, "code_blocks", None) or []
            attrs["code"] = _truncate(
                "\n---\n".join(getattr(b, "code", "") for b in blocks)
            )
        else:
            result = getattr(event, "result", None)
            attrs["exit_code"] = int(getattr(result, "exit_code", -1))
            attrs["output"] = _truncate(str(getattr(result, "output", "")))
        span.add_event(type_name, attributes=attrs)
    elif type_name in ("SelectorEvent", "SelectSpeakerEvent"):
        span.add_event(
            "speaker_selection",
            attributes={
                "content": _truncate(str(getattr(event, "content", ""))),
                "source": source or "",
            },
        )


# ---------------------------------------------------------------------------
# instrument_agent
# ---------------------------------------------------------------------------


def instrument_agent(agent: Any, *, tracer: Tracer, trace: Trace) -> Any:
    """Wrap *agent* so every turn emits an agent-attributed span.

    *agent* must implement the ``BaseChatAgent`` protocol's
    ``on_messages_stream(messages, cancellation_token)`` async-generator
    method (true for ``AssistantAgent``, ``CodeExecutorAgent``,
    ``UserProxyAgent``, ``SocietyOfMindAgent``, and any custom
    ``BaseChatAgent`` subclass).  The wrap is applied at the *instance*
    level (``agent.on_messages_stream = wrapped``) so ``agent.run()``/
    ``agent.run_stream()``/``agent.on_messages()`` -- all of which resolve
    ``self.on_messages_stream`` via normal attribute lookup -- are covered
    by this single patch point without needing to hook each one separately.

    Idempotent: instrumenting the same agent instance twice is a no-op on
    the second call.

    Returns *agent* for chaining.
    """
    if getattr(agent, "_agent_trace_instrumented", False):
        return agent

    original_stream = agent.on_messages_stream
    agent_name: str = getattr(agent, "name", None) or "agent"

    async def _wrapped_stream(messages: Any, cancellation_token: Any) -> Any:
        span = tracer.start_span(f"agent:{agent_name}")
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.type", type(agent).__name__)
        try:
            async for event in original_stream(messages, cancellation_token):
                try:
                    _record_event(span, event)
                except Exception:
                    logger.debug(
                        "agent-trace: failed to enrich span for autogen event %r",
                        type(event).__name__,
                        exc_info=True,
                    )
                if type(event).__name__ == "Response" and span.end_time is None:
                    # Response is always the last item on_messages_stream
                    # yields.  Close the span synchronously *before*
                    # yielding it rather than waiting for this generator to
                    # naturally finish, because BaseChatAgent.on_messages()
                    # returns as soon as it sees a Response and never calls
                    # anext() again -- this generator's eventual
                    # GeneratorExit-based cleanup is then scheduled
                    # asynchronously by the event loop's asyncgen finalizer
                    # hook and is not guaranteed to run before the caller's
                    # code continues, which would otherwise leave the span
                    # open (status UNSET) indefinitely.
                    span.end(SpanStatus.OK)
                yield event
        except GeneratorExit:
            # The caller stopped iterating without draining the generator --
            # e.g. BaseChatAgent.on_messages() returns as soon as it sees the
            # final Response and never calls anext() again, so this
            # generator is torn down (via GeneratorExit, or in practice
            # asyncio.run()'s implicit shutdown_asyncgens() cleanup, which
            # has been observed to surface as asyncio.CancelledError instead
            # of GeneratorExit here -- see the broader except clause below)
            # rather than ever reaching normal completion.  In the common
            # case the Response-triggered proactive close above has already
            # set end_time, so this is a no-op; it only fires if the stream
            # was abandoned before ever yielding a Response, which is still
            # expected/successful termination, not a failure.
            if span.end_time is None:
                span.end(SpanStatus.OK)
            raise
        except BaseException as exc:
            # Guard on end_time rather than unconditionally recording: if
            # the span was already closed above (Response already seen and
            # yielded), a CancelledError/GeneratorExit arriving afterwards
            # is just interpreter/event-loop teardown of an abandoned
            # generator, not a real turn failure -- record_exception() would
            # otherwise unconditionally flip an already-OK span to ERROR.
            if span.end_time is None:
                span.record_exception(exc)
                span.end(SpanStatus.ERROR)
            raise
        else:
            if span.end_time is None:
                span.end(SpanStatus.OK)

    agent.on_messages_stream = _wrapped_stream
    agent._agent_trace_instrumented = True
    return agent


# ---------------------------------------------------------------------------
# instrument_team
# ---------------------------------------------------------------------------


def instrument_team(team: Any, *, tracer: Tracer, trace: Trace) -> Any:
    """Recursively instrument every participant of *team*, plus a root span
    for the team's own ``run_stream``.

    *team* must implement the ``TaskRunner`` protocol's
    ``run_stream(task, cancellation_token)`` async-generator method (true
    for ``SelectorGroupChat``, ``RoundRobinGroupChat``, ``Swarm``, and any
    other ``BaseGroupChat`` subclass).

    Participants are discovered via the ``_participants`` attribute set by
    ``BaseGroupChat.__init__`` -- AutoGen does not expose a public accessor
    for the participant list, so this reaches into the same underscore-
    prefixed attribute the backlog investigation for this integration
    confirmed by reading ``_base_group_chat.py`` directly.  Each participant
    is instrumented via :func:`instrument_agent` (if it looks like a
    ``BaseChatAgent``) or recursively via :func:`instrument_team` (if it
    looks like a nested ``Team``, e.g. inside a ``SocietyOfMindAgent``).

    Idempotent: instrumenting the same team instance twice is a no-op on
    the second call for the team's own root span (participants already
    instrumented are also left untouched, since :func:`instrument_agent`/
    :func:`instrument_team` are themselves idempotent).

    Returns *team* for chaining.
    """
    for participant in getattr(team, "_participants", None) or []:
        if hasattr(participant, "on_messages_stream"):
            instrument_agent(participant, tracer=tracer, trace=trace)
        elif hasattr(participant, "run_stream"):
            instrument_team(participant, tracer=tracer, trace=trace)

    if getattr(team, "_agent_trace_instrumented", False):
        return team

    original_run_stream = team.run_stream
    team_name: str = getattr(team, "name", None) or type(team).__name__

    async def _wrapped_run_stream(*args: Any, **kwargs: Any) -> Any:
        span = tracer.start_span(f"team:{team_name}")
        span.set_attribute("team.name", team_name)
        try:
            async for event in original_run_stream(*args, **kwargs):
                try:
                    _record_event(span, event)
                except Exception:
                    logger.debug(
                        "agent-trace: failed to enrich span for autogen team event %r",
                        type(event).__name__,
                        exc_info=True,
                    )
                if type(event).__name__ == "TaskResult" and span.end_time is None:
                    # Same proactive-close reasoning as instrument_agent()'s
                    # Response check: some callers may iterate run_stream()
                    # directly and stop as soon as they see the TaskResult,
                    # never resuming this generator.
                    span.end(SpanStatus.OK)
                yield event
        except GeneratorExit:
            # Same early-close case as instrument_agent()'s _wrapped_stream.
            if span.end_time is None:
                span.end(SpanStatus.OK)
            raise
        except BaseException as exc:
            # See instrument_agent()'s matching guard: don't let a
            # CancelledError/GeneratorExit from abandoned-generator teardown
            # retroactively flip an already-closed OK span to ERROR.
            if span.end_time is None:
                span.record_exception(exc)
                span.end(SpanStatus.ERROR)
            raise
        else:
            if span.end_time is None:
                span.end(SpanStatus.OK)

    team.run_stream = _wrapped_run_stream
    team._agent_trace_instrumented = True
    return team


# ---------------------------------------------------------------------------
# instrument_code_executor
# ---------------------------------------------------------------------------


def instrument_code_executor(executor: Any, *, tracer: Tracer, trace: Trace) -> Any:
    """Wrap *executor* so every ``execute_code_blocks`` call is recorded as
    a span, independent of and in addition to any LLM HTTP capture.

    *executor* must implement the ``CodeExecutor`` protocol's
    ``execute_code_blocks(code_blocks, cancellation_token) -> CodeResult``
    coroutine method (true for ``LocalCommandLineCodeExecutor``,
    ``DockerCommandLineCodeExecutor``, ``JupyterCodeExecutor``, and any
    other ``CodeExecutor`` implementation).

    Records the executed code (each ``CodeBlock.code``/``.language``), the
    executor's working directory (``executor.work_dir`` when present), and
    the returned ``CodeResult``'s ``exit_code`` and ``output``.  AutoGen's
    ``CodeResult``/``CommandLineCodeResult`` merges stdout and stderr into a
    single ``output`` string with no separate split anywhere in the
    executor's own return type -- confirmed by reading both
    ``autogen_core.code_executor.CodeResult`` and
    ``autogen_ext.code_executors.local.CommandLineCodeResult`` -- so
    ``output`` here is that combined stream, not stdout alone.

    Idempotent: instrumenting the same executor instance twice is a no-op
    on the second call.

    Returns *executor* for chaining.
    """
    if getattr(executor, "_agent_trace_instrumented", False):
        return executor

    original_execute = executor.execute_code_blocks

    async def _wrapped_execute(code_blocks: Any, cancellation_token: Any) -> Any:
        span = tracer.start_span("code_execution")
        work_dir = getattr(executor, "work_dir", None)
        if work_dir is not None:
            span.set_attribute("code_execution.work_dir", str(work_dir))
        span.set_attribute(
            "code_execution.code",
            _truncate("\n---\n".join(getattr(b, "code", "") for b in code_blocks)),
        )
        span.set_attribute(
            "code_execution.languages",
            ",".join(getattr(b, "language", "") for b in code_blocks),
        )
        try:
            result = await original_execute(code_blocks, cancellation_token)
        except BaseException as exc:
            span.record_exception(exc)
            if span.end_time is None:
                span.end(SpanStatus.ERROR)
            raise

        exit_code = int(getattr(result, "exit_code", -1))
        span.set_attribute("code_execution.exit_code", exit_code)
        span.set_attribute(
            "code_execution.output",
            _truncate(str(getattr(result, "output", ""))),
        )
        span.end(SpanStatus.OK if exit_code == 0 else SpanStatus.ERROR)
        return result

    executor.execute_code_blocks = _wrapped_execute
    executor._agent_trace_instrumented = True
    return executor
