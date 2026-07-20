"""
Haystack pipeline tracing example.

Run: uv run python examples/04-haystack-pipeline/example.py

Demonstrates capturing a Haystack 2.x Pipeline run with agent-trace, using
Haystack's own native tracing.Tracer/tracing.Span instrumentation surface
(there is no callback list to pass in, unlike LangGraph — a tracer is
registered globally with haystack.tracing.enable_tracing(...)).

No LLM API calls required — every component here is pure Python, so you can
run this end to end with no credentials.

This example also demonstrates the exact capability gap issue #4574
(https://github.com/deepset-ai/haystack/issues/4574) exposes: a `params`
dict passed into one component not reaching the component it was intended
for is an in-process Python argument-propagation bug, invisible to
agent-trace's HTTP interceptor (no network call is involved) but fully
visible once Haystack's own component-level tracing hook is wired up —
set HAYSTACK_CONTENT_TRACING_ENABLED=1 to also capture each component's
*actual received arguments*, not just their types.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

# ---------------------------------------------------------------------------
# Minimal guard: give a clear error if haystack-ai is not installed
# ---------------------------------------------------------------------------
try:
    import haystack.tracing
    from haystack import Pipeline, component
except ImportError:
    import sys

    sys.exit("haystack-ai is not installed.\nRun: pip install agent-observability-trace-cli[haystack]")

from agent_trace import SpanStatus, tracer
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.haystack import HaystackTracer

# ---------------------------------------------------------------------------
# Pipeline components — pure Python, no network calls
# ---------------------------------------------------------------------------


@component
class DocumentSplitter:
    """Splits raw text into short 'chunks' (a stand-in for a real splitter)."""

    @component.output_types(chunks=list[str])
    def run(self, text: str, chunk_size: int = 40) -> dict[str, list[str]]:
        words = text.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        return {"chunks": chunks}


@component
class KeywordScorer:
    """Scores each chunk by how many of a fixed keyword set it contains."""

    _KEYWORDS: ClassVar[set[str]] = {
        "agent",
        "trace",
        "observability",
        "replay",
        "debugging",
    }

    @component.output_types(scored_chunks=list[dict[str, object]])
    def run(self, chunks: list[str]) -> dict[str, list[dict[str, object]]]:
        scored = []
        for chunk in chunks:
            words = {w.strip(".,").lower() for w in chunk.split()}
            hits = words & self._KEYWORDS
            scored.append({"text": chunk, "score": len(hits), "keywords": sorted(hits)})
        return {"scored_chunks": scored}


@component
class TopChunkSelector:
    """Picks the highest-scoring chunk."""

    @component.output_types(best_chunk=dict[str, object])
    def run(
        self, scored_chunks: list[dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        best = max(scored_chunks, key=lambda c: c["score"])
        return {"best_chunk": best}


def build_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("splitter", DocumentSplitter())
    pipeline.add_component("scorer", KeywordScorer())
    pipeline.add_component("selector", TopChunkSelector())
    pipeline.connect("splitter.chunks", "scorer.chunks")
    pipeline.connect("scorer.scored_chunks", "selector.scored_chunks")
    return pipeline


def main() -> None:
    # Set HAYSTACK_CONTENT_TRACING_ENABLED=1 before running to also capture
    # each component's actual received arguments/returned output onto the
    # spans below (gated by Haystack itself, off by default for privacy).
    if os.environ.get("HAYSTACK_CONTENT_TRACING_ENABLED", "").lower() in (
        "1",
        "true",
    ):
        haystack.tracing.tracer.is_content_tracing_enabled = True

    text = (
        "agent-trace is an observability library for AI agents. "
        "It records every outbound HTTP call your agent makes and replays "
        "it offline without making API calls, which makes debugging "
        "non-deterministic failures much faster. "
        "This example traces a Haystack pipeline instead of an HTTP call, "
        "showing agent-trace's framework-level instrumentation."
    )

    pipeline = build_pipeline()

    print("Running Haystack pipeline...")
    with tracer.start_trace("haystack_pipeline_demo") as trace:
        haystack.tracing.enable_tracing(HaystackTracer(tracer=tracer, trace=trace))
        try:
            result = pipeline.run({"splitter": {"text": text, "chunk_size": 15}})
        finally:
            haystack.tracing.disable_tracing()

        run_id = trace.run_id

    trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
    loaded_trace = Trace.from_dict(json.loads(trace_path.read_text()))

    print("\n--- Span tree ---")
    StdoutExporter().export(loaded_trace)

    error_count = sum(1 for s in loaded_trace.spans if s.status == SpanStatus.ERROR)
    print(f"\nTrace saved to: {trace_path.parent}")
    print(f"Spans captured: {len(loaded_trace.spans)}  (errors: {error_count})")

    print("\n--- Result ---")
    print(f"Best chunk: {result['selector']['best_chunk']}")


if __name__ == "__main__":
    main()
