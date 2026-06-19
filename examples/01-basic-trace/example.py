"""
Basic trace example.
Run: uv run python examples/01-basic-trace/example.py

This example traces a simple function call with manual spans.
No LLM API calls required — the work is simulated with time.sleep()
so you can see realistic span durations without any credentials.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent_trace import tracer
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter


def extract_entities(doc: str) -> list[str]:
    """Simulate entity extraction (no real NLP here)."""
    # In a real agent this would call an NLP API or LLM.
    time.sleep(0.05)  # simulate 50 ms of work
    words = [w for w in doc.split() if len(w) > 5]
    return words[:5]


def summarize_text(doc: str) -> str:
    """Simulate summarization."""
    time.sleep(0.08)  # simulate 80 ms of work
    sentences = doc.split(".")
    return sentences[0].strip() + "." if sentences else doc


def score_document(doc: str, entities: list[str]) -> float:
    """Simulate document quality scoring."""
    time.sleep(0.03)  # simulate 30 ms of work
    # Simple heuristic: ratio of entity words to total words
    words = doc.split()
    if not words:
        return 0.0
    return min(1.0, len(entities) / len(words) * 10)


def process_document(doc: str) -> dict[str, object]:
    """
    Process a document through three sub-steps, each traced as a child span.

    Returns a dict with keys: entities, summary, score.
    """
    with tracer.start_trace("process_document") as trace:
        result: dict[str, object] = {}

        with tracer.span("extract-entities") as span:
            span.set_attribute("doc.length_chars", len(doc))
            entities = extract_entities(doc)
            span.set_attribute("entity_count", len(entities))
            result["entities"] = entities

        with tracer.span("summarize") as span:
            span.set_attribute("doc.length_chars", len(doc))
            summary = summarize_text(doc)
            span.set_attribute("summary.length_chars", len(summary))
            result["summary"] = summary

        with tracer.span("score") as span:
            span.set_attribute("entity_count", len(entities))
            score = score_document(doc, entities)
            span.set_attribute("score", score)
            result["score"] = score

        # Export the trace to stdout immediately after it closes
        run_id = trace.run_id

    # The trace context has exited — load the saved trace.json and display it.
    trace_path = (
        Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
    )
    loaded_trace = Trace.from_dict(json.loads(trace_path.read_text()))

    print("\n--- Span tree ---")
    StdoutExporter().export(loaded_trace)
    print(f"\nTrace saved to: {trace_path.parent}")

    return result


def main() -> None:
    doc = (
        "agent-trace is an observability library for AI agents. "
        "It records every outbound HTTP call your agent makes and stores "
        "the request and response in a local SQLite database. "
        "You can replay any recorded run offline without making API calls, "
        "which makes debugging non-deterministic failures much faster."
    )

    print("Processing document...")
    result = process_document(doc)

    print("\n--- Result ---")
    print(f"Entities: {result['entities']}")
    print(f"Summary:  {result['summary']}")
    print(f"Score:    {result['score']:.3f}")


if __name__ == "__main__":
    main()
