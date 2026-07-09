"""
Integration tests for the AutoGen (autogen-agentchat/autogen-ext v0.4+/v0.7.x)
integration.

These tests require real autogen-agentchat/autogen-ext/autogen-core
installations but do NOT require live LLM API calls:

- Agent/team/token-usage/tool-call/handoff tests use
  ``autogen_ext.models.replay.ReplayChatCompletionClient``, AutoGen's own
  canned-response test double -- real ``AssistantAgent``/``BaseGroupChat``
  code runs end-to-end with zero network I/O.
- The code-execution test uses a real ``LocalCommandLineCodeExecutor``
  running a local Python subprocess (no network).
- The http_client wiring test uses ``respx`` to mock the OpenAI chat
  completions endpoint, verifying agent-trace's RecordingTransport captures
  real HTTP traffic issued by autogen-ext's own OpenAIChatCompletionClient.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("autogen_agentchat", reason="autogen-agentchat not installed")
pytest.importorskip("autogen_ext", reason="autogen-ext not installed")
respx = pytest.importorskip("respx", reason="respx not installed")

from agent_trace import SpanStatus, Tracer
from agent_trace.integrations.autogen import (
    instrument_agent,
    instrument_code_executor,
    instrument_team,
    recording_http_client,
)


def _model_info() -> object:
    from autogen_core.models import ModelInfo

    return ModelInfo(
        vision=False,
        function_calling=True,
        json_output=False,
        family="unknown",
        structured_output=False,
    )


@pytest.mark.integration
class TestInstrumentAgentIntegration:
    async def test_agent_span_created_for_real_assistant_agent(
        self, tmp_path: Path
    ) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        from autogen_ext.models.replay import ReplayChatCompletionClient

        model_client = ReplayChatCompletionClient(
            ["Hello there!"], model_info=_model_info()
        )
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-agent-real") as trace:
            agent = AssistantAgent("writer", model_client=model_client)
            instrument_agent(agent, tracer=t, trace=trace)

            response = await agent.on_messages(
                [TextMessage(content="hi", source="user")], CancellationToken()
            )
            assert response.chat_message.content == "Hello there!"

        agent_spans = [s for s in trace.spans if s.name == "agent:writer"]
        assert len(agent_spans) == 1
        assert agent_spans[0].status == SpanStatus.OK
        assert agent_spans[0].attributes["agent.name"] == "writer"
        assert agent_spans[0].attributes["agent.type"] == "AssistantAgent"

    async def test_agent_run_also_covered_by_instance_patch(
        self, tmp_path: Path
    ) -> None:
        """agent.run() (TaskRunner interface) calls self.run_stream(), which
        calls self.on_messages_stream() -- verify the instance-level patch
        applied by instrument_agent() is not bypassed by that indirection."""
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.replay import ReplayChatCompletionClient

        model_client = ReplayChatCompletionClient(
            ["Done via .run()"], model_info=_model_info()
        )
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-agent-run") as trace:
            agent = AssistantAgent("worker", model_client=model_client)
            instrument_agent(agent, tracer=t, trace=trace)
            result = await agent.run(task="do something")

        assert result.messages[-1].content == "Done via .run()"
        agent_spans = [s for s in trace.spans if s.name == "agent:worker"]
        assert len(agent_spans) == 1
        assert agent_spans[0].status == SpanStatus.OK

    async def test_token_usage_recorded_from_real_replay_client(
        self, tmp_path: Path
    ) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        from autogen_ext.models.replay import ReplayChatCompletionClient

        model_client = ReplayChatCompletionClient(["A reply"], model_info=_model_info())
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-usage-real") as trace:
            agent = AssistantAgent("worker", model_client=model_client)
            instrument_agent(agent, tracer=t, trace=trace)
            await agent.on_messages(
                [TextMessage(content="hi", source="user")], CancellationToken()
            )

        span = next(s for s in trace.spans if s.name == "agent:worker")
        assert span.attributes["llm.usage.prompt_tokens"] > 0
        assert "llm.usage.completion_tokens" in span.attributes

    async def test_tool_call_events_recorded_from_real_tool_loop(
        self, tmp_path: Path
    ) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken, FunctionCall
        from autogen_core.models import CreateResult, RequestUsage
        from autogen_ext.models.replay import ReplayChatCompletionClient

        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        responses = [
            CreateResult(
                finish_reason="function_calls",
                content=[
                    FunctionCall(id="call_1", arguments='{"x": 3}', name="double")
                ],
                usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
                cached=False,
            ),
            CreateResult(
                finish_reason="stop",
                content="The result is 6.",
                usage=RequestUsage(prompt_tokens=15, completion_tokens=6),
                cached=False,
            ),
        ]
        model_client = ReplayChatCompletionClient(responses, model_info=_model_info())
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-tools-real") as trace:
            agent = AssistantAgent(
                "worker",
                model_client=model_client,
                tools=[double],
                reflect_on_tool_use=True,
            )
            instrument_agent(agent, tracer=t, trace=trace)
            await agent.on_messages(
                [TextMessage(content="double 3", source="user")],
                CancellationToken(),
            )

        span = next(s for s in trace.spans if s.name == "agent:worker")
        event_names = [e.name for e in span.events]
        assert "tool_call_request" in event_names
        assert "tool_call_execution" in event_names
        req = next(e for e in span.events if e.name == "tool_call_request")
        assert req.attributes["tool.names"] == "double"

    async def test_exception_before_http_recorded_and_reraised(
        self, tmp_path: Path
    ) -> None:
        """A failure raised inside _call_llm (e.g. tool-schema conversion, per
        issue #6912) must be recorded on the agent span and re-raised --
        this is the exception-to-span attribution the modern-architecture
        integration exists for, independent of any HTTP-layer capture."""
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        from autogen_ext.models.replay import ReplayChatCompletionClient

        model_client = ReplayChatCompletionClient(["unused"], model_info=_model_info())

        async def _boom(*args: object, **kwargs: object) -> None:
            raise TypeError("Cannot instantiate typing.Union")

        model_client.create = _boom  # type: ignore[method-assign]

        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(TypeError, match="Cannot instantiate"):
            with t.start_trace("autogen-error-real") as trace:
                agent = AssistantAgent("worker", model_client=model_client)
                instrument_agent(agent, tracer=t, trace=trace)
                await agent.on_messages(
                    [TextMessage(content="hi", source="user")], CancellationToken()
                )

        span = next(s for s in trace.spans if s.name == "agent:worker")
        assert span.status == SpanStatus.ERROR
        exc_events = [e for e in span.events if e.name == "exception"]
        assert exc_events
        assert exc_events[0].attributes["exception.type"] == "TypeError"


@pytest.mark.integration
class TestInstrumentTeamIntegration:
    async def test_team_and_participant_spans_created(self, tmp_path: Path) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.replay import ReplayChatCompletionClient

        planner_client = ReplayChatCompletionClient(
            ["Plan: write a haiku."], model_info=_model_info()
        )
        writer_client = ReplayChatCompletionClient(
            ["An old silent pond..."], model_info=_model_info()
        )
        planner = AssistantAgent("planner", model_client=planner_client)
        writer = AssistantAgent("writer", model_client=writer_client)

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("autogen-team-real") as trace:
            # 3 = task message + planner's turn + writer's turn, matching
            # AutoGen's own RoundRobinGroupChat docs example for a 2-agent team.
            team = RoundRobinGroupChat(
                [planner, writer], termination_condition=MaxMessageTermination(3)
            )
            instrument_team(team, tracer=t, trace=trace)
            await team.run(task="write something")

        span_names = {s.name for s in trace.spans}
        assert "team:RoundRobinGroupChat" in span_names
        assert "agent:planner" in span_names
        assert "agent:writer" in span_names
        team_span = next(s for s in trace.spans if s.name == "team:RoundRobinGroupChat")
        assert team_span.status == SpanStatus.OK


@pytest.mark.integration
class TestInstrumentCodeExecutorIntegration:
    async def test_local_command_line_executor_captures_exit_code_and_output(
        self, tmp_path: Path
    ) -> None:
        from autogen_core import CancellationToken
        from autogen_core.code_executor import CodeBlock
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

        t = Tracer(trace_dir=tmp_path)
        with tempfile.TemporaryDirectory() as work_dir:
            executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
            with t.start_trace("autogen-code-real") as trace:
                instrument_code_executor(executor, tracer=t, trace=trace)
                result = await executor.execute_code_blocks(
                    [
                        CodeBlock(
                            code="print('hello from real subprocess')",
                            language="python",
                        )
                    ],
                    CancellationToken(),
                )

            assert result.exit_code == 0
            assert "hello from real subprocess" in result.output

            span = next(s for s in trace.spans if s.name == "code_execution")
            assert span.status == SpanStatus.OK
            assert span.attributes["code_execution.exit_code"] == 0
            assert (
                "hello from real subprocess" in span.attributes["code_execution.output"]
            )
            assert span.attributes["code_execution.work_dir"] == str(executor.work_dir)

    async def test_nonzero_exit_code_from_real_subprocess_marks_error(
        self, tmp_path: Path
    ) -> None:
        from autogen_core import CancellationToken
        from autogen_core.code_executor import CodeBlock
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

        t = Tracer(trace_dir=tmp_path)
        with tempfile.TemporaryDirectory() as work_dir:
            executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
            with t.start_trace("autogen-code-fail-real") as trace:
                instrument_code_executor(executor, tracer=t, trace=trace)
                result = await executor.execute_code_blocks(
                    [CodeBlock(code="import sys; sys.exit(1)", language="python")],
                    CancellationToken(),
                )

            assert result.exit_code == 1
            span = next(s for s in trace.spans if s.name == "code_execution")
            assert span.status == SpanStatus.ERROR
            assert span.attributes["code_execution.exit_code"] == 1


@pytest.mark.integration
class TestRecordingHttpClientIntegration:
    async def test_recording_http_client_captures_real_openai_ext_client_traffic(
        self, tmp_path: Path
    ) -> None:
        """Backlog verification: OpenAIChatCompletionClient(http_client=...)
        actually routes through RecordingTransport with zero AutoGen-specific
        code changes to agent-trace itself -- confirmed live against the
        installed autogen-ext package, not just by reading source."""
        import httpx
        from autogen_core.models import UserMessage
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        from agent_trace._replay.fixture import Fixture

        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from mock!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        fixture = Fixture(tmp_path / "fixture.db")
        try:
            with respx.mock(assert_all_called=True) as mock:
                mock.post("https://api.openai.com/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json=mock_response)
                )
                client = OpenAIChatCompletionClient(
                    model="gpt-4o-mini",
                    api_key="test-key",
                    http_client=recording_http_client(fixture, is_async=True),
                )
                result = await client.create([UserMessage(content="hi", source="user")])
                assert result.content == "Hello from mock!"

            assert fixture.exchange_count() == 1
        finally:
            fixture.close()

    async def test_global_start_trace_record_true_also_captures_autogen_ext_client(
        self, tmp_path: Path
    ) -> None:
        """Documents that recording_http_client() is not the *only* path:
        a client constructed with no explicit http_client, after entering
        tracer.start_trace(record=True), is already captured by agent-trace's
        existing global httpx.AsyncClient patch -- confirmed live here."""
        import httpx
        from autogen_core.models import UserMessage
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        mock_response = {
            "id": "chatcmpl-test2",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Auto-captured!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        }

        t = Tracer(trace_dir=tmp_path)
        with respx.mock(assert_all_called=True) as mock:
            mock.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )
            with t.start_trace("autogen-global-patch", record=True) as trace:
                # No explicit http_client kwarg at all.
                client = OpenAIChatCompletionClient(
                    model="gpt-4o-mini", api_key="test-key"
                )
                result = await client.create([UserMessage(content="hi", source="user")])
                assert result.content == "Auto-captured!"

        from agent_trace._replay.fixture import Fixture

        run_dir = tmp_path / trace.run_id
        with Fixture(run_dir / "fixture.db") as fixture:
            assert fixture.exchange_count() == 1
