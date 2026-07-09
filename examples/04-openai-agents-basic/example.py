"""
OpenAI Agents SDK — worked example.

This example demonstrates agent-trace's OpenAI Agents SDK integration
(`agent_trace.integrations.openai_agents`): a `researcher` agent that calls a
function tool, then hands off to a `writer` agent to produce the final
answer.  Instrumented via `instrument_runner()`, this produces an
`agent_run` root span containing:

    agent_run
    ├── agent:researcher
    │   ├── llm:<model>            (one or more, per turn)
    │   ├── tool:lookup_fact
    │   └── handoff:researcher->writer   (duration-based, see below)
    └── agent:writer
        └── llm:<model>

Prerequisites:
    pip install agent-trace[openai-agents]
    export OPENAI_API_KEY=your-key   # only needed for the record step

Step 1 — Record (requires API key):
    python examples/04-openai-agents-basic/example.py record

Step 2 — Replay (no API key needed):
    python examples/04-openai-agents-basic/example.py replay <run_id>

Step 3 — Streamed variant (requires API key):
    python examples/04-openai-agents-basic/example.py streamed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    from agents import Agent, function_tool
except ImportError:
    sys.exit(
        "openai-agents is not installed.\nRun: pip install agent-trace[openai-agents]"
    )

from agent_trace import Tracer, replay
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.openai_agents import (
    instrument_runner,
    instrument_runner_streamed,
)

QUESTION = "What year was the Rust programming language first released?"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@function_tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a programming-language topic (stubbed for the demo)."""
    return (
        f"[stubbed lookup] {topic}: Rust 1.0 shipped in 2015; "
        "the project started in 2006."
    )


def build_pipeline() -> Agent:
    writer = Agent(
        name="writer",
        instructions=(
            "Using only the research handed to you, write a single, concise "
            "sentence answering the original question."
        ),
    )
    researcher = Agent(
        name="researcher",
        instructions=(
            "Call lookup_fact once to research the question, then hand off to "
            "the writer agent with what you found. Do not answer directly."
        ),
        tools=[lookup_fact],
        handoffs=[writer],
    )
    return researcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_span_tree(run_id: str) -> None:
    trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
    if not trace_path.exists():
        print(f"  (no trace.json found at {trace_path})")
        return
    trace_obj = Trace.from_dict(json.loads(trace_path.read_text()))
    StdoutExporter().export(trace_obj)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


async def _record() -> str:
    t = Tracer()
    agent = build_pipeline()
    with t.start_trace("openai-agents-basic", record=True) as trace:
        result = await instrument_runner(
            agent, QUESTION, tracer=t, trace=trace, max_turns=4
        )
        run_id = trace.run_id
        print(f"Final output: {result.final_output}")
    return run_id


def cmd_record() -> None:
    print(f"Recording run for: {QUESTION!r}")
    print("(This makes real API calls. Ensure OPENAI_API_KEY is set.)\n")

    run_id = asyncio.run(_record())

    print(f"\nRun ID: {run_id}")
    print_span_tree(run_id)
    print("\nReplay with:")
    print(f"  python examples/04-openai-agents-basic/example.py replay {run_id}")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


async def _replay(run_id: str) -> None:
    agent = build_pipeline()
    t = Tracer()
    with replay(run_id) as ctx:
        with t.start_trace("openai-agents-basic-replay") as trace:
            result = await instrument_runner(
                agent, QUESTION, tracer=t, trace=trace, max_turns=4
            )
            print(f"Final output: {result.final_output}")
        print(f"\nExchanges consumed: {ctx.fixture.exchange_count()}")


def cmd_replay(run_id: str) -> None:
    print(f"Replaying run: {run_id}")
    print("(No API calls will be made — all responses served from fixture.db)\n")
    try:
        asyncio.run(_replay(run_id))
    except FileNotFoundError:
        sys.exit(
            f"No fixture found for run {run_id!r}.\n"
            "Check the run ID with: agent-trace list"
        )


# ---------------------------------------------------------------------------
# Streamed variant — Runner.run_streamed()/stream_events() support
# ---------------------------------------------------------------------------


async def _streamed() -> None:
    t = Tracer()
    agent = build_pipeline()
    with t.start_trace("openai-agents-streamed", record=True) as trace:
        streamed = await instrument_runner_streamed(
            agent, QUESTION, tracer=t, trace=trace, max_turns=4
        )
        async for event in streamed.stream_events():
            print(f"  event: {getattr(event, 'type', type(event).__name__)}")
        print(f"\nFinal output: {streamed.final_output}")
        run_id = trace.run_id
    print(f"\nRun ID: {run_id}")
    print_span_tree(run_id)


def cmd_streamed() -> None:
    print(f"Streaming run for: {QUESTION!r}")
    print("(This makes real API calls. Ensure OPENAI_API_KEY is set.)\n")
    asyncio.run(_streamed())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI Agents SDK worked example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("record", help="Record a new run (requires OPENAI_API_KEY)")

    rep = sub.add_parser("replay", help="Replay a recorded run offline")
    rep.add_argument("run_id", help="Run ID from the record step (e.g. run_abc123)")

    sub.add_parser(
        "streamed",
        help="Run via Runner.run_streamed()/stream_events() (requires OPENAI_API_KEY)",
    )

    args = parser.parse_args()

    if args.command == "record":
        cmd_record()
    elif args.command == "replay":
        cmd_replay(args.run_id)
    elif args.command == "streamed":
        cmd_streamed()


if __name__ == "__main__":
    main()
