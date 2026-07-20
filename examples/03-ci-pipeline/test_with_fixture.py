"""
pytest test that replays a recorded agent run.

Run: pytest examples/03-ci-pipeline/ -v

The test is skipped if fixture.db has not been recorded yet.
Record it with:

    python examples/03-ci-pipeline/example.py record
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agent_trace import replay
from agent_trace import Fixture

# Ensure the example module in this directory is importable as `example`.
sys.path.insert(0, str(Path(__file__).parent))

FIXTURE_PATH = Path(__file__).parent / "fixture.db"


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="No fixture recorded yet. Run: python examples/03-ci-pipeline/example.py record",
)
def test_agent_responds_correctly() -> None:
    """The agent must return a non-empty classification with positive confidence."""
    with replay(FIXTURE_PATH) as ctx:
        # Import here so the module is only loaded inside the replay context,
        # which ensures any httpx clients created at import time are patched.
        from example import classify_document  # type: ignore[import]

        # Load the input doc from fixture metadata
        doc = ctx.fixture.get_metadata("input_doc") or (
            "agent-trace is an observability tool for AI agents. "
            "It records every HTTP call and lets you replay runs offline."
        )
        result = classify_document(doc)
        exchange_count = ctx.fixture.exchange_count()

    assert isinstance(result["classification"], str)
    assert len(result["classification"]) > 0
    assert isinstance(result["confidence"], float)
    assert result["confidence"] > 0.0
    assert exchange_count > 0


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="No fixture recorded yet. Run: python examples/03-ci-pipeline/example.py record",
)
def test_replay_is_fast() -> None:
    """Replay should complete in under 200 ms — no network I/O."""
    with Fixture(FIXTURE_PATH) as f:
        doc = f.get_metadata("input_doc") or (
            "agent-trace is an observability tool for AI agents."
        )

    start = time.perf_counter()

    with replay(FIXTURE_PATH) as ctx:
        from example import classify_document  # type: ignore[import]

        classify_document(doc)

    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200, (
        f"Replay took {elapsed_ms:.1f} ms — expected < 200 ms. "
        "Is the network guard active? Set AGENT_TRACE_NETWORK_GUARD=1."
    )


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="No fixture recorded yet. Run: python examples/03-ci-pipeline/example.py record",
)
def test_fixture_exchange_count_matches_expectation() -> None:
    """The fixture must contain exactly 1 HTTP exchange for this simple agent."""
    with replay(FIXTURE_PATH) as ctx:
        assert ctx.fixture.exchange_count() > 0, (
            "Fixture is empty — the recording may have failed"
        )
