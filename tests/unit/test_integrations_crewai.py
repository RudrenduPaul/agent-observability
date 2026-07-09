"""
Unit tests for agent_trace.integrations.crewai.CrewAITracer.

Uses the REAL installed ``crewai`` package (event classes, ``Agent``/``Task``/
``Crew`` model construction) rather than mocks — construction of these objects
does not make any network/LLM calls, so tests stay hermetic. Skips cleanly
when ``crewai`` is not installed.

Most tests call the tracer's handler methods directly with real crewAI event
objects (bypassing the event bus's threaded dispatch) for determinism — the
same style ``test_integrations_langgraph.py`` uses for LangGraph callbacks.
A smaller set of "wired" tests goes through the real
``crewai_event_bus.emit()`` path, scoped with ``crewai_event_bus.scoped_handlers()``
so registered handlers never leak into other tests.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("crewai", reason="crewai not installed")

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.crewai import CrewAITracer

# crewAI's Agent construction validates an LLM string against known
# providers, but does not make a network call at construction time — a
# placeholder key is enough. Set this once, defensively, in case no real
# key is present in the test environment.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-not-a-real-key")

from crewai import Agent, Task
from crewai.events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)
from crewai.events.types.llm_events import LLMCallType
from crewai.tasks.task_output import TaskOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tracer_and_hook(tmp_path: Path):
    """Yield (tracer, trace, hook) with the trace context kept alive for the
    whole test body.

    ``Tracer.start_trace`` is a ``@contextmanager``-wrapped generator. Calling
    ``.__enter__()`` on it directly and discarding the returned context
    manager object (e.g. inside a plain helper *function* that returns before
    the caller is done) lets CPython's refcounting garbage-collect that
    object as soon as the function returns — which closes the underlying
    generator and runs its ``finally: self._active_trace_var.reset(token)``
    immediately, silently clearing the active trace before the test body
    ever runs. Using a pytest fixture with ``yield`` (so the ``with`` block
    spans the entire test) avoids that trap.
    """
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("crewai-unit-test") as trace:
        hook = CrewAITracer(tracer=t, trace=trace)
        try:
            yield t, trace, hook
        finally:
            hook.close()


def _fake_agent(role: str = "researcher") -> Agent:
    return Agent(
        role=role,
        goal=f"{role} goal",
        backstory=f"{role} backstory",
        llm="gpt-4o-mini",
    )


def _fake_task(agent: Agent, description: str = "do the thing") -> Task:
    return Task(description=description, expected_output="a result", agent=agent)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestCrewAITracerInit:
    def test_spans_dict_empty_at_start(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        assert hook._spans == {}

    def test_tracer_and_trace_stored(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        assert hook._tracer is t
        assert hook._trace is trace

    def test_handlers_registered_on_bus(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        assert len(hook._registered) == 15

    def test_close_unregisters_all_handlers(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        registered = list(hook._registered)
        hook.close()
        for event_type, handler in registered:
            assert handler not in crewai_event_bus._sync_handlers.get(
                event_type, frozenset()
            )
        assert hook._registered == []

    def test_close_is_idempotent(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        hook.close()
        hook.close()  # must not raise

    def test_context_manager_closes_on_exit(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("ctx-mgr-test") as trace:
            with CrewAITracer(tracer=t, trace=trace) as hook:
                assert hook._closed is False
            assert hook._closed is True


# ---------------------------------------------------------------------------
# Crew kickoff span lifecycle
# ---------------------------------------------------------------------------


class TestCrewKickoffLifecycle:
    def test_crew_started_opens_span(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        event = CrewKickoffStartedEvent(crew_name="research-crew", inputs=None)
        hook._on_crew_started(None, event)
        assert event.event_id in hook._spans
        span = hook._spans[event.event_id]
        assert span.name == "crew:research-crew"
        assert span.attributes["crew.name"] == "research-crew"

    def test_crew_completed_closes_span_ok(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        started = CrewKickoffStartedEvent(crew_name="c", inputs=None)
        hook._on_crew_started(None, started)

        completed = CrewKickoffCompletedEvent(
            crew_name="c",
            output="final answer",
            total_tokens=42,
            started_event_id=started.event_id,
        )
        hook._on_crew_completed(None, completed)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.OK
        assert trace.spans[0].attributes["crew.total_tokens"] == 42

    def test_crew_failed_closes_span_error(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        started = CrewKickoffStartedEvent(crew_name="c", inputs=None)
        hook._on_crew_started(None, started)

        failed = CrewKickoffFailedEvent(
            error="boom", crew_name="c", started_event_id=started.event_id
        )
        hook._on_crew_failed(None, failed)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.ERROR
        exc_events = [e for e in trace.spans[0].events if e.name == "exception"]
        assert exc_events
        assert exc_events[0].attributes["exception.message"] == "boom"


# ---------------------------------------------------------------------------
# Agent execution span lifecycle + parenting under crew span
# ---------------------------------------------------------------------------


class TestAgentExecutionLifecycle:
    def test_agent_started_opens_span(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent("researcher")
        task = _fake_task(agent)
        event = AgentExecutionStartedEvent(
            agent=agent, task=task, tools=[], task_prompt="go"
        )
        hook._on_agent_started(agent, event)
        assert event.event_id in hook._spans
        span = hook._spans[event.event_id]
        assert span.name == "agent:researcher"
        assert span.attributes["agent.role"] == "researcher"

    def test_agent_span_parented_under_crew_span(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent("researcher")
        task = _fake_task(agent)

        crew_started = CrewKickoffStartedEvent(crew_name="c", inputs=None)
        hook._on_crew_started(None, crew_started)
        crew_span = hook._spans[crew_started.event_id]

        agent_started = AgentExecutionStartedEvent(
            agent=agent,
            task=task,
            tools=[],
            task_prompt="go",
            parent_event_id=crew_started.event_id,
        )
        hook._on_agent_started(agent, agent_started)
        agent_span = hook._spans[agent_started.event_id]

        assert agent_span.parent_id == crew_span.span_id

    def test_agent_completed_closes_span_ok(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        started = AgentExecutionStartedEvent(
            agent=agent, task=task, tools=[], task_prompt="go"
        )
        hook._on_agent_started(agent, started)

        completed = AgentExecutionCompletedEvent(
            agent=agent,
            task=task,
            output="done",
            started_event_id=started.event_id,
        )
        hook._on_agent_completed(agent, completed)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.OK

    def test_agent_error_closes_span_error(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        started = AgentExecutionStartedEvent(
            agent=agent, task=task, tools=[], task_prompt="go"
        )
        hook._on_agent_started(agent, started)

        error_event = AgentExecutionErrorEvent(
            agent=agent,
            task=task,
            error="agent blew up",
            started_event_id=started.event_id,
        )
        hook._on_agent_error(agent, error_event)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.ERROR


# ---------------------------------------------------------------------------
# Task span lifecycle
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    def test_task_started_opens_span_with_task_id(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent, description="write the report")
        event = TaskStartedEvent(context=None, task=task)
        hook._on_task_started(task, event)

        assert event.event_id in hook._spans
        span = hook._spans[event.event_id]
        assert span.name.startswith("task:")
        assert span.attributes["task.id"] == str(task.id)

    def test_task_completed_closes_span_ok(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        started = TaskStartedEvent(context=None, task=task)
        hook._on_task_started(task, started)

        output = TaskOutput(
            description=task.description, agent=agent.role, raw="the result"
        )
        completed = TaskCompletedEvent(
            output=output, task=task, started_event_id=started.event_id
        )
        hook._on_task_completed(task, completed)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.OK

    def test_task_failed_closes_span_error(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        started = TaskStartedEvent(context=None, task=task)
        hook._on_task_started(task, started)

        failed = TaskFailedEvent(
            error="task failed", task=task, started_event_id=started.event_id
        )
        hook._on_task_failed(task, failed)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.ERROR


# ---------------------------------------------------------------------------
# LLM call span lifecycle — token usage (mirrors LangGraph/OpenAI Agents tests)
# ---------------------------------------------------------------------------


class TestLLMCallLifecycle:
    def test_llm_started_opens_span(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        event = LLMCallStartedEvent(call_id="call-1", model="gpt-4o-mini")
        hook._on_llm_started(None, event)
        assert event.event_id in hook._spans
        span = hook._spans[event.event_id]
        assert span.name == "llm:gpt-4o-mini"

    def test_llm_completed_records_token_usage(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        started = LLMCallStartedEvent(call_id="call-2", model="gpt-4o-mini")
        hook._on_llm_started(None, started)

        completed = LLMCallCompletedEvent(
            call_id="call-2",
            model="gpt-4o-mini",
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            finish_reason="stop",
            started_event_id=started.event_id,
        )
        hook._on_llm_completed(None, completed)

        assert started.event_id not in hook._spans
        span = trace.spans[0]
        assert span.attributes["llm.usage.prompt_tokens"] == 10
        assert span.attributes["llm.usage.completion_tokens"] == 5
        assert span.attributes["llm.usage.total_tokens"] == 15
        assert span.attributes["llm.finish_reason"] == "stop"

    def test_llm_completed_tolerates_missing_usage(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        started = LLMCallStartedEvent(call_id="call-3", model="gpt-4o-mini")
        hook._on_llm_started(None, started)

        completed = LLMCallCompletedEvent(
            call_id="call-3",
            model="gpt-4o-mini",
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            usage=None,
            started_event_id=started.event_id,
        )
        hook._on_llm_completed(None, completed)  # must not raise

        assert trace.spans[0].end_time is not None

    def test_llm_failed_closes_span_error(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        started = LLMCallStartedEvent(call_id="call-4", model="gpt-4o-mini")
        hook._on_llm_started(None, started)

        failed = LLMCallFailedEvent(
            call_id="call-4",
            model="gpt-4o-mini",
            error="rate limited",
            started_event_id=started.event_id,
        )
        hook._on_llm_failed(None, failed)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.ERROR

    def test_two_concurrent_llm_calls_keyed_independently(
        self, tracer_and_hook
    ) -> None:
        """Two overlapping LLM calls (different call_id) must not collide."""
        t, trace, hook = tracer_and_hook
        started_a = LLMCallStartedEvent(call_id="call-a", model="gpt-4o-mini")
        started_b = LLMCallStartedEvent(call_id="call-b", model="gpt-4o-mini")
        hook._on_llm_started(None, started_a)
        hook._on_llm_started(None, started_b)

        assert len(hook._spans) == 2

        completed_a = LLMCallCompletedEvent(
            call_id="call-a",
            model="gpt-4o-mini",
            response="a",
            call_type=LLMCallType.LLM_CALL,
            started_event_id=started_a.event_id,
        )
        hook._on_llm_completed(None, completed_a)

        assert len(hook._spans) == 1
        assert started_b.event_id in hook._spans


# ---------------------------------------------------------------------------
# Tool usage span lifecycle
# ---------------------------------------------------------------------------


class TestToolUsageLifecycle:
    def test_tool_started_opens_span(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        event = ToolUsageStartedEvent(
            tool_name="web_search",
            tool_args={"query": "crewai"},
            from_agent=agent,
            from_task=task,
        )
        hook._on_tool_started(agent, event)
        assert event.event_id in hook._spans
        span = hook._spans[event.event_id]
        assert span.name == "tool:web_search"
        assert span.attributes["tool.name"] == "web_search"

    def test_tool_finished_closes_span_ok_with_output_length(
        self, tracer_and_hook
    ) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        started = ToolUsageStartedEvent(
            tool_name="calc", tool_args={}, from_agent=agent, from_task=task
        )
        hook._on_tool_started(agent, started)

        finished = ToolUsageFinishedEvent(
            tool_name="calc",
            tool_args={},
            from_agent=agent,
            from_task=task,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            output="42",
            started_event_id=started.event_id,
        )
        hook._on_tool_finished(agent, finished)

        assert started.event_id not in hook._spans
        span = trace.spans[0]
        assert span.status == SpanStatus.OK
        assert span.attributes["tool.output_length"] == 2

    def test_tool_error_closes_span_error(self, tracer_and_hook) -> None:
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        started = ToolUsageStartedEvent(
            tool_name="broken_tool",
            tool_args={},
            from_agent=agent,
            from_task=task,
        )
        hook._on_tool_started(agent, started)

        error_event = ToolUsageErrorEvent(
            tool_name="broken_tool",
            tool_args={},
            from_agent=agent,
            from_task=task,
            error="tool crashed",
            started_event_id=started.event_id,
        )
        hook._on_tool_error(agent, error_event)

        assert started.event_id not in hook._spans
        assert trace.spans[0].status == SpanStatus.ERROR

    def test_unknown_started_event_id_in_end_is_noop(self, tracer_and_hook) -> None:
        """Closing a span keyed to an event_id that was never opened must not raise."""
        t, trace, hook = tracer_and_hook
        agent = _fake_agent()
        task = _fake_task(agent)
        finished = ToolUsageFinishedEvent(
            tool_name="ghost",
            tool_args={},
            from_agent=agent,
            from_task=task,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            output="x",
            started_event_id="nonexistent-event-id",
        )
        hook._on_tool_finished(agent, finished)  # must not raise
        assert trace.spans == []


# ---------------------------------------------------------------------------
# Wired end-to-end tests — real crewai_event_bus.emit() dispatch
# ---------------------------------------------------------------------------


class TestWiredEventBus:
    """These go through the real, threaded crewai_event_bus.emit() path
    rather than calling handler methods directly, proving the handlers are
    actually registered under the exact event classes crewAI emits.
    """

    def test_emit_crew_kickoff_started_and_completed_pair(
        self, tmp_path: Path
    ) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("wired-test") as trace:
            with crewai_event_bus.scoped_handlers():
                with CrewAITracer(tracer=t, trace=trace):
                    started = CrewKickoffStartedEvent(
                        crew_name="wired-crew", inputs=None
                    )
                    f1 = crewai_event_bus.emit(None, started)
                    if f1 is not None:
                        f1.result(timeout=5)

                    assert len(trace.spans) == 1
                    assert trace.spans[0].name == "crew:wired-crew"

                    # Emit the matching completed event too, so this test
                    # pops the same event-scope stack entry it pushed rather
                    # than leaking an unclosed scope into later tests that
                    # share the same contextvar-backed stack.
                    completed = CrewKickoffCompletedEvent(
                        crew_name="wired-crew",
                        output="done",
                        total_tokens=0,
                    )
                    f2 = crewai_event_bus.emit(None, completed)
                    if f2 is not None:
                        f2.result(timeout=5)

                    assert trace.spans[0].status == SpanStatus.OK

    def test_emit_llm_call_started_and_completed_pair(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("wired-llm-test") as trace:
            with crewai_event_bus.scoped_handlers():
                with CrewAITracer(tracer=t, trace=trace):
                    started = LLMCallStartedEvent(call_id="wired-1", model="gpt-4o")
                    f1 = crewai_event_bus.emit(None, started)
                    if f1 is not None:
                        f1.result(timeout=5)

                    completed = LLMCallCompletedEvent(
                        call_id="wired-1",
                        model="gpt-4o",
                        response="hi",
                        call_type=LLMCallType.LLM_CALL,
                        started_event_id=started.event_id,
                    )
                    f2 = crewai_event_bus.emit(None, completed)
                    if f2 is not None:
                        f2.result(timeout=5)

                    assert len(trace.spans) == 1
                    assert trace.spans[0].status == SpanStatus.OK
                    assert trace.spans[0].name == "llm:gpt-4o"
