"""
crewAI research-crew record/replay example.

Demonstrates the CrewAITracer integration: a two-agent, two-task sequential
crew (researcher -> writer) instrumented with agent-trace, recorded once
against the real OpenAI API, then replayed offline at zero API cost.

crewAI's default ``llm="gpt-4o-mini"`` string resolves to crewAI's native
``OpenAICompletion`` class (confirmed via a live install — see
CrewAITracer's module docstring for how span pairing works), which talks to
OpenAI over ``httpx.Client`` — the same transport agent-trace's generic
HTTP interceptor already patches. Record/replay therefore works exactly like
the LangGraph examples: no crewAI-specific fixture handling needed.

Prerequisites:
    pip install agent-observability-trace-cli[crewai]
    export OPENAI_API_KEY=your-key  # only needed for the record step

Step 1 — Record (requires API key):
    python examples/04-crewai-research-crew/example.py record

Step 2 — Replay (no API key needed):
    python examples/04-crewai-research-crew/example.py replay <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from crewai import Agent, Crew, Process, Task  # type: ignore[import]
except ImportError:
    sys.exit("crewai is not installed.\nRun: pip install agent-observability-trace-cli[crewai]")

from agent_trace import replay, tracer
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.crewai import CrewAITracer

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Crew definition — researcher hands notes to writer, sequential process
# ---------------------------------------------------------------------------


def build_crew(topic: str) -> Crew:
    researcher = Agent(
        role="Researcher",
        goal=f"Gather three concise facts about {topic}",
        backstory="A careful analyst who only states verifiable facts.",
        llm="gpt-4o-mini",
        verbose=False,
    )
    writer = Agent(
        role="Writer",
        goal="Turn research notes into a two-sentence summary",
        backstory="A clear, concise technical writer.",
        llm="gpt-4o-mini",
        verbose=False,
    )

    research_task = Task(
        description=f"Research {topic} and list three concise facts.",
        expected_output="Three bullet-point facts.",
        agent=researcher,
    )
    write_task = Task(
        description="Summarize the research notes in exactly two sentences.",
        expected_output="A two-sentence summary.",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
    )


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


def cmd_record(topic: str) -> None:
    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set. Export it before running record.")

    print(f"Recording crew run for topic: {topic!r}")
    print("(This makes real API calls. Ensure OPENAI_API_KEY is set.)\n")

    crew = build_crew(topic)
    run_id = ""

    with tracer.start_trace("crewai-research-crew", record=True) as trace:
        with CrewAITracer(tracer=tracer, trace=trace):
            run_id = trace.run_id
            result = crew.kickoff()
            print(f"Result: {result}")

    print(f"\nRun ID: {run_id}")
    print_span_tree(run_id)
    print("\nReplay with:")
    print(f"  python examples/04-crewai-research-crew/example.py replay {run_id}")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def cmd_replay(run_id: str) -> None:
    print(f"Replaying run: {run_id}")
    print("(No API calls will be made — all responses served from fixture.db)\n")

    from agent_trace import Fixture

    fix_path = Path.home() / ".agent-trace" / "runs" / run_id / "fixture.db"
    if not fix_path.exists():
        sys.exit(
            f"No fixture found for run {run_id!r}.\n"
            "Check the run ID with: agent-trace list"
        )
    with Fixture(fix_path) as f:
        exchange_count = f.exchange_count()
    print(f"Fixture has {exchange_count} recorded HTTP exchange(s).\n")

    crew = build_crew("agent-trace record/replay")

    # Replay doesn't need CrewAITracer attached — it only needs the fixture's
    # recorded HTTP responses served in place of live network calls (via the
    # FixtureClock/replay transport installed by `replay()`). The span tree
    # printed below is the one captured during the original `record` run.
    with replay(run_id) as ctx:
        result = crew.kickoff()
        print(f"Result: {result}")

    print(f"\nExchanges consumed: {ctx.fixture.exchange_count()}")
    print_span_tree(run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="crewAI research-crew record/replay example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record a new run (requires OPENAI_API_KEY)")
    rec.add_argument("--topic", default="the CAP theorem", help="Research topic")

    rep = sub.add_parser("replay", help="Replay a recorded run offline")
    rep.add_argument("run_id", help="Run ID from the record step (e.g. run_abc123)")

    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args.topic)
    elif args.command == "replay":
        cmd_replay(args.run_id)


if __name__ == "__main__":
    main()
