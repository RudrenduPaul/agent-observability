"""
LangGraph failure replay example.

This example demonstrates recording a LangGraph run that encounters an error,
and replaying it offline without making API calls.

Prerequisites:
    pip install agent-observability-trace-cli[langgraph]
    export OPENAI_API_KEY=your-key  # only needed for the record step

Step 1 — Record (requires API key):
    python examples/02-langgraph-failure-replay/example.py record

Step 2 — Replay (no API key needed):
    python examples/02-langgraph-failure-replay/example.py replay <run_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

# ---------------------------------------------------------------------------
# Minimal guard: give a clear error if langgraph is not installed
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import END, StateGraph  # type: ignore[import]
except ImportError:
    sys.exit(
        "langgraph is not installed.\n"
        "Run: pip install agent-observability-trace-cli[langgraph]"
    )

import httpx

from agent_trace import replay, tracer
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    question: str
    research_notes: str
    analysis: str
    response: str


def research_node(state: ResearchState) -> ResearchState:
    """Call the LLM to gather research notes on the question."""
    with tracer.span("research") as span:
        span.set_attribute("langgraph.node_name", "research")
        span.set_attribute("llm.model", "gpt-4o-mini")

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Export it before running the record step."
            )

        resp = httpx.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research assistant. Be concise.",
                    },
                    {
                        "role": "user",
                        "content": f"Research: {state['question']}",
                    },
                ],
                "max_tokens": 150,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        span.set_attribute("llm.token_count.total", usage.get("total_tokens", 0))

        notes = data["choices"][0]["message"]["content"]
        return {"research_notes": notes}


def analyze_node(state: ResearchState) -> ResearchState:
    """Analyze the research notes to extract key insights."""
    with tracer.span("analyze") as span:
        span.set_attribute("langgraph.node_name", "analyze")
        span.set_attribute("llm.model", "gpt-4o-mini")

        resp = httpx.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Notes: {state['research_notes']}\n\n"
                            "List 3 key insights as bullet points."
                        ),
                    }
                ],
                "max_tokens": 150,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        span.set_attribute("llm.token_count.total", usage.get("total_tokens", 0))

        analysis = data["choices"][0]["message"]["content"]
        return {"analysis": analysis}


def respond_node(state: ResearchState) -> ResearchState:
    """Write a final response using the research and analysis."""
    with tracer.span("respond") as span:
        span.set_attribute("langgraph.node_name", "respond")
        span.set_attribute("llm.model", "gpt-4o-mini")

        resp = httpx.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['question']}\n"
                            f"Analysis: {state['analysis']}\n\n"
                            "Write a two-sentence answer."
                        ),
                    }
                ],
                "max_tokens": 100,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        span.set_attribute("llm.token_count.total", usage.get("total_tokens", 0))

        response = data["choices"][0]["message"]["content"]
        return {"response": response}


def build_graph() -> object:
    return (
        StateGraph(ResearchState)
        .add_node("research", research_node)
        .add_node("analyze", analyze_node)
        .add_node("respond", respond_node)
        .add_edge("research", "analyze")
        .add_edge("analyze", "respond")
        .add_edge("respond", END)
        .set_entry_point("research")
        .compile()
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


def cmd_record(question: str) -> None:
    print(f"Recording run for: {question!r}")
    print("(This makes real API calls. Ensure OPENAI_API_KEY is set.)\n")

    graph = build_graph()
    run_id: str = ""

    with tracer.start_trace("langgraph-failure-replay", record=True) as trace:
        try:
            result = graph.invoke({"question": question})
            run_id = trace.run_id
            print(f"Research notes: {result['research_notes'][:80]}...")
            print(f"Analysis:       {result['analysis'][:80]}...")
            print(f"Response:       {result['response']}")
        except Exception as exc:
            run_id = trace.run_id
            print(f"Run failed with: {type(exc).__name__}: {exc}")
            print("(Failure was recorded — you can replay it to debug.)")

    print(f"\nRun ID: {run_id}")
    print_span_tree(run_id)
    print(f"\nReplay with:")
    print(f"  python examples/02-langgraph-failure-replay/example.py replay {run_id}")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def cmd_replay(run_id: str) -> None:
    print(f"Replaying run: {run_id}")
    print("(No API calls will be made — all responses served from fixture.db)\n")

    # Load the original question from the fixture metadata if available,
    # otherwise use a placeholder — the HTTP responses are what matter.
    try:
        from agent_trace import Fixture
        fix_path = Path.home() / ".agent-trace" / "runs" / run_id / "fixture.db"
        with Fixture(fix_path) as f:
            question = f.get_metadata("question") or "What is agent-trace?"
            exchange_count = f.exchange_count()
    except FileNotFoundError:
        sys.exit(
            f"No fixture found for run {run_id!r}.\n"
            "Check the run ID with: agent-trace list"
        )

    print(f"Fixture has {exchange_count} recorded HTTP exchange(s).")
    print(f"Question: {question!r}\n")

    graph = build_graph()

    with replay(run_id) as ctx:
        try:
            result = graph.invoke({"question": question})
            print(f"Research notes: {result['research_notes'][:80]}...")
            print(f"Analysis:       {result['analysis'][:80]}...")
            print(f"Response:       {result['response']}")
        except Exception as exc:
            print(f"Replay reproduced the same failure: {type(exc).__name__}: {exc}")

    print(f"\nExchanges consumed: {ctx.fixture.exchange_count()}")
    print_span_tree(run_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph failure replay example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record a new run (requires OPENAI_API_KEY)")
    rec.add_argument(
        "--question",
        default="What is the difference between LangGraph and LangChain?",
        help="Question to research",
    )

    rep = sub.add_parser("replay", help="Replay a recorded run offline")
    rep.add_argument("run_id", help="Run ID from the record step (e.g. run_abc123)")

    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args.question)
    elif args.command == "replay":
        cmd_replay(args.run_id)


if __name__ == "__main__":
    main()
