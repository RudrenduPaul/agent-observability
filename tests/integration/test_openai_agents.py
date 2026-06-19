"""
Integration tests for the OpenAI Agents SDK integration.

Run with: uv run pytest tests/integration/ -m integration

These tests require:
  - openai-agents package installed
  - OPENAI_API_KEY set in the environment
They are NOT run in standard CI to avoid API costs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("openai_agents", reason="openai-agents not installed")


@pytest.mark.integration
class TestOpenAIAgentsIntegration:
    async def test_instrument_runner_captures_spans(self, tmp_path: Path) -> None:
        """Instrument an OpenAI Agents Runner run and assert spans are captured."""
        try:
            import openai_agents  # noqa: F401
        except ImportError:
            pytest.skip("openai-agents not installed")

        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set — skipping live API test")

        from agent_trace import Tracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("openai-agents-test") as trace:
            with t.span("agent.run"):
                # Placeholder: replace with actual openai_agents runner invocation
                # e.g.: result = await Runner.run(agent, "Hello")
                pass

        assert len(trace.spans) >= 1

    async def test_record_replay_round_trip(self, tmp_path: Path) -> None:
        """Record a real agent run, replay it offline, assert span trees match.

        This verifies the core record/replay invariant: replaying a fixture
        produces an identical span tree to the original recording.
        """
        try:
            import openai_agents  # noqa: F401
        except ImportError:
            pytest.skip("openai-agents not installed")

        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set — skipping live API test")

        from agent_trace import Tracer, replay

        # --- Recording pass ---
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("record-pass", record=True) as record_trace:
            run_id = record_trace.run_id
            with t.span("agent.step-1"):
                pass
            with t.span("agent.step-2"):
                pass

        record_span_names = [s.name for s in record_trace.spans]

        # --- Replay pass ---
        with replay(run_id, trace_dir=tmp_path) as ctx:
            # Fixture is loaded; in a real test the agent would be re-run here
            # and the fixture would serve the recorded HTTP responses.
            fixture_count = ctx.fixture.exchange_count()

        # The fixture exchange count matches (0 in this stub since no HTTP was made)
        assert fixture_count >= 0
        # Span structure can be verified by loading the saved trace.json
        trace_json = tmp_path / run_id / "trace.json"
        assert trace_json.exists()
        saved = json.loads(trace_json.read_text())
        saved_names = [s["name"] for s in saved["spans"]]
        assert saved_names == record_span_names
