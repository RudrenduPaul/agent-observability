"""
CI pipeline example — record once, replay in every test run.

This example shows how to use agent-trace in a CI pipeline:
- Record a real agent run once (locally or in a setup step)
- Commit the fixture to your repo (or cache it in CI)
- Replay in every CI test run at zero API cost

Run the recording step:
    python examples/03-ci-pipeline/example.py record

Run the test (uses fixture):
    python examples/03-ci-pipeline/example.py test

Run with pytest (preferred):
    pytest examples/03-ci-pipeline/ -v
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

from agent_trace import replay, tracer
from agent_trace.replay.fixture import Fixture

# The fixture is stored next to this script so it can be committed to the repo.
# In a real project, put it in a dedicated fixtures/ directory.
FIXTURE_DIR = Path(__file__).parent
FIXTURE_PATH = FIXTURE_DIR / "fixture.db"

# The "API" this example calls.  In a real pipeline this would be OpenAI,
# Anthropic, or your own microservice.  Here we use httpbin.org, which is a
# public HTTP testing service that echoes request data back.
HTTPBIN_URL = "https://httpbin.org/post"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------


def classify_document(doc: str) -> dict[str, object]:
    """
    Classify a document by calling an external API.

    During recording this hits the real endpoint.
    During replay the response is served from fixture.db.
    """
    with tracer.span("classify") as span:
        span.set_attribute("doc.length_chars", len(doc))

        # In a real agent you would call the LLM API here.
        # For this demo we call httpbin.org so no API key is needed.
        resp = httpx.post(
            HTTPBIN_URL,
            json={"text": doc, "task": "classify"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # httpbin echoes the JSON we sent; pretend the "classification" is
        # derived from the response length.
        classification = "short" if len(doc) < 100 else "long"
        confidence = min(0.95, len(doc) / 500)

        span.set_attribute("classification", classification)
        span.set_attribute("confidence", confidence)

        return {
            "classification": classification,
            "confidence": confidence,
            "api_response_size": len(resp.content),
        }


def run_agent(doc: str) -> dict[str, object]:
    """Run the classification agent on doc."""
    with tracer.start_trace("ci-pipeline-agent", record=False) as _trace:
        return classify_document(doc)


# ---------------------------------------------------------------------------
# Record command
# ---------------------------------------------------------------------------


def cmd_record() -> None:
    """Record a real run and save the fixture next to this script."""
    if FIXTURE_PATH.exists():
        print(f"Fixture already exists at {FIXTURE_PATH}")
        print("Delete it first if you want to re-record.")
        sys.exit(1)

    doc = (
        "agent-trace is an observability tool for AI agents. "
        "It records every HTTP call and lets you replay runs offline."
    )

    print(f"Recording to: {FIXTURE_PATH}")
    print(f"Document: {doc[:60]}...")
    print()

    with tracer.start_trace("ci-pipeline-agent", record=True) as trace:
        # Save the input document in the fixture metadata so tests can verify
        # they are running against the right fixture.
        fixture_path_tmp = (
            Path.home() / ".agent-trace" / "runs" / trace.run_id / "fixture.db"
        )
        result = classify_document(doc)

    # Copy the fixture from the default trace dir to the example directory
    # so it can be committed to the repo.
    import shutil
    shutil.copy2(fixture_path_tmp, FIXTURE_PATH)

    # Store the input in the local fixture's metadata table
    with Fixture(FIXTURE_PATH) as f:
        f.set_metadata("input_doc", doc)
        f.set_metadata("expected_classification", str(result["classification"]))
        print(f"Stored {f.exchange_count()} HTTP exchange(s) in fixture")

    print(f"\nResult: {result}")
    print(f"\nFixture written to: {FIXTURE_PATH}")
    print("Commit this file to your repo so CI can use it.")
    print("\nRun the test with:")
    print("  pytest examples/03-ci-pipeline/ -v")


# ---------------------------------------------------------------------------
# Test command (manual, without pytest)
# ---------------------------------------------------------------------------


def cmd_test() -> None:
    """Replay the recorded run and verify the output."""
    if not FIXTURE_PATH.exists():
        print(f"No fixture found at {FIXTURE_PATH}")
        print("Run the record step first:")
        print("  python examples/03-ci-pipeline/example.py record")
        sys.exit(1)

    with Fixture(FIXTURE_PATH) as f:
        doc = f.get_metadata("input_doc") or (
            "agent-trace is an observability tool for AI agents. "
            "It records every HTTP call and lets you replay runs offline."
        )
        expected_classification = f.get_metadata("expected_classification") or "long"
        exchange_count = f.exchange_count()

    print(f"Replaying {exchange_count} exchange(s) from: {FIXTURE_PATH}")
    print(f"Document: {doc[:60]}...")
    print()

    start = time.perf_counter()

    with replay(FIXTURE_PATH) as ctx:
        result = classify_document(doc)

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Result: {result}")
    print(f"Elapsed: {elapsed_ms:.1f} ms (no network I/O)")
    print()

    # Assertions
    assert result["classification"] == expected_classification, (
        f"Expected {expected_classification!r}, got {result['classification']!r}"
    )
    assert result["confidence"] > 0.0, "Confidence should be positive"
    assert ctx.fixture.exchange_count() == exchange_count, (
        "Replay should consume all recorded exchanges"
    )

    print("All assertions passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI pipeline example — record once, replay forever",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("record", help="Record a real run and save fixture.db here")
    sub.add_parser("test", help="Replay the fixture and verify output (no pytest)")

    args = parser.parse_args()

    if args.command == "record":
        cmd_record()
    elif args.command == "test":
        cmd_test()


if __name__ == "__main__":
    main()
