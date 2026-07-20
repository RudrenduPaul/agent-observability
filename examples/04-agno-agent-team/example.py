"""
Agno Agent/Team example.

Demonstrates the Agno integration (agent_trace.integrations.agno) capturing:
  - A solo Agent run (agent span + llm span + tool span)
  - A Team run that delegates to a member Agent (per-team-member attribution:
    the member's own run span is a child of the team's run span)
  - An in-process exception raised entirely inside a model's own code, never
    reaching the HTTP layer at all — the exact failure class the
    framework-agnostic HTTP interceptor alone cannot see (upstream issue #5298)

Runs with NO API key by default (a tiny scripted fake model stands in for a
real provider, deterministically, so this example is reproducible in CI).
Pass --live to use a real OpenAI model instead (requires OPENAI_API_KEY and
`pip install agent-observability-trace-cli[agno] openai`).

Prerequisites:
    pip install agent-observability-trace-cli[agno]

Usage:
    python examples/04-agno-agent-team/example.py
    python examples/04-agno-agent-team/example.py --live
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

try:
    from agno.agent.agent import Agent
    from agno.models.base import Model
    from agno.models.response import ModelResponse
    from agno.team.team import Team
except ImportError:
    sys.exit("agno is not installed.\nRun: pip install agent-observability-trace-cli[agno]")

from agent_trace import Tracer
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.agno import instrument_agent_arun

# ---------------------------------------------------------------------------
# A tiny scripted model — no network calls, no API key required.
#
# It overrides Model.invoke_stream/ainvoke_stream, the code path Agno's own
# run loop actually calls when stream=True (confirmed against agno==2.7.1's
# agent/_run.py) — the same integration point AgnoTracer observes via the
# ModelRequestStarted/ModelRequestCompleted events.
# ---------------------------------------------------------------------------


@dataclass
class ScriptedModel(Model):
    id: str = "scripted-model"
    script: list[ModelResponse] = field(default_factory=list)
    _call_count: int = field(default=0, init=False, repr=False)

    def _next_response(self) -> ModelResponse:
        idx = min(self._call_count, len(self.script) - 1)
        self._call_count += 1
        return self.script[idx]

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next_response()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next_response()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._next_response()

    async def ainvoke_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ModelResponse]:
        yield self._next_response()

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    return str(eval(expression))


def _leader_model(live: bool) -> Model:
    if live:
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id="gpt-4o-mini")
    return ScriptedModel(
        id="scripted-leader",
        script=[
            ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_delegate",
                        "type": "function",
                        "function": {
                            "name": "delegate_task_to_member",
                            "arguments": (
                                '{"member_id": "researcher", '
                                '"task": "summarize agent-trace"}'
                            ),
                        },
                    }
                ],
            ),
            ModelResponse(role="assistant", content="Here is the team's final answer."),
        ],
    )


def _member_model(live: bool) -> Model:
    if live:
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id="gpt-4o-mini")
    return ScriptedModel(
        id="scripted-member",
        script=[
            ModelResponse(
                role="assistant",
                content="agent-trace records and replays agent runs.",
            )
        ],
    )


def _solo_model(live: bool) -> Model:
    if live:
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id="gpt-4o-mini")
    return ScriptedModel(
        id="scripted-solo",
        script=[
            ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "6*7"}',
                        },
                    }
                ],
            ),
            ModelResponse(role="assistant", content="6 * 7 is 42."),
        ],
    )


async def run_solo_agent(live: bool) -> None:
    print("=" * 70)
    print("1. Solo Agent run — with a tool call")
    print("=" * 70)

    agent = Agent(model=_solo_model(live), name="math-agent", tools=[calculator])

    t = Tracer()
    with t.start_trace("agno-example-solo") as trace:
        result = await instrument_agent_arun(
            agent, "what is 6*7?", tracer=t, trace=trace
        )

    print(f"Result: {result.content}\n")
    StdoutExporter().export(trace)
    print()


async def run_team_delegation(live: bool) -> None:
    print("=" * 70)
    print("2. Team run — leader delegates to a member Agent")
    print("=" * 70)

    member = Agent(model=_member_model(live), name="researcher", id="researcher")
    team = Team(
        model=_leader_model(live),
        name="research-team",
        members=[member],
        stream_member_events=True,
    )

    t = Tracer()
    with t.start_trace("agno-example-team") as trace:
        result = await instrument_agent_arun(
            team, "tell me about agent-trace", tracer=t, trace=trace
        )

    print(f"Result: {result.content}\n")
    StdoutExporter().export(trace)

    # The member's own run span is a *child* of the team's run span, and the
    # tool span for the delegation itself carries agno.child_run_id — the
    # correlation this integration exists to provide (upstream issue #5326).
    team_span = next(s for s in trace.spans if s.name.startswith("team:"))
    member_spans = [s for s in trace.spans if s.name.startswith("agent:")]
    print(f"\nTeam span:   {team_span.name} ({team_span.span_id})")
    for m in member_spans:
        print(f"Member span: {m.name} ({m.span_id}) parent={m.parent_id}")
    print()


async def run_in_process_crash() -> None:
    print("=" * 70)
    print("3. In-process exception — never reaches the HTTP layer at all")
    print("=" * 70)

    @dataclass
    class CrashingModel(ScriptedModel):
        def _next_response(self) -> ModelResponse:  # type: ignore[override]
            # Simulates the exact failure class behind upstream issue #5298:
            # an UnboundLocalError raised entirely inside agno/models/base.py,
            # with zero HTTP traffic — something the framework-agnostic HTTP
            # interceptor alone could never see.
            raise UnboundLocalError("simulated crash inside Agno's own model code")

    agent = Agent(model=CrashingModel(id="crashing-model"), name="crash-agent")

    t = Tracer()
    with t.start_trace("agno-example-crash") as trace:
        result = await instrument_agent_arun(agent, "hi", tracer=t, trace=trace)

    print(f"Result: {result!r} (None — the run errored)\n")
    StdoutExporter().export(trace)
    error_span = next(s for s in trace.spans if s.name.startswith("agent:"))
    print(f"\nSpan status: {error_span.status.value}")
    for event in error_span.events:
        if event.name == "exception":
            message = event.attributes["exception.message"]
            print(f"Captured exception.message: {message}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Use a real OpenAI model instead of the bundled scripted fake "
            "(requires OPENAI_API_KEY)"
        ),
    )
    args = parser.parse_args()

    if args.live and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("--live requires OPENAI_API_KEY to be set.")

    await run_solo_agent(args.live)
    await run_team_delegation(args.live)
    await run_in_process_crash()  # always scripted — the crash is deterministic


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
