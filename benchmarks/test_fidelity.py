"""
Benchmark/correctness check: replay fidelity.

Records a synthetic 5-step fixture (mix of LLM calls and tool calls).
Replays it. Diffs the recorded exchange sequence against the replayed sequence.
Target: 100% exchange match.

Run with:
    uv run pytest benchmarks/test_fidelity.py -v --benchmark-only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from agent_trace._replay.engine import ReplayEngine
from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.httpx_hook import ReplayTransport

# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

_LLM_URL = "https://api.openai.com/v1/chat/completions"
_TOOL_URL = "https://api.example.com/tool"


def _llm_body(step: int) -> str:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {"content": f"llm-response-{step}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )


def _tool_body(step: int) -> str:
    return json.dumps({"tool_result": f"tool-output-{step}"})


def _create_test_fixture(tmp_path: Path) -> Path:
    """Create a Fixture with 5 exchanges: 3 LLM calls + 2 tool calls.

    Returns the path to the fixture.db file.
    """
    db_path = tmp_path / "fidelity_fixture.db"
    with Fixture(db_path, trace_id="bench-trace") as f:
        # LLM call 1
        f.record_exchange(
            url=_LLM_URL,
            method="POST",
            request_headers={"content-type": "application/json"},
            request_body=json.dumps({"model": "gpt-4o", "step": 1}),
            response_status=200,
            response_headers={"content-type": "application/json"},
            response_body=_llm_body(1),
        )
        # Tool call 1
        f.record_exchange(
            url=_TOOL_URL,
            method="GET",
            request_headers={"accept": "application/json"},
            request_body="",
            response_status=200,
            response_headers={"content-type": "application/json"},
            response_body=_tool_body(1),
        )
        # LLM call 2
        f.record_exchange(
            url=_LLM_URL,
            method="POST",
            request_headers={"content-type": "application/json"},
            request_body=json.dumps({"model": "gpt-4o", "step": 2}),
            response_status=200,
            response_headers={"content-type": "application/json"},
            response_body=_llm_body(2),
        )
        # Tool call 2
        f.record_exchange(
            url=_TOOL_URL,
            method="GET",
            request_headers={"accept": "application/json"},
            request_body="",
            response_status=200,
            response_headers={"content-type": "application/json"},
            response_body=_tool_body(2),
        )
        # LLM call 3
        f.record_exchange(
            url=_LLM_URL,
            method="POST",
            request_headers={"content-type": "application/json"},
            request_body=json.dumps({"model": "gpt-4o", "step": 3}),
            response_status=200,
            response_headers={"content-type": "application/json"},
            response_body=_llm_body(3),
        )
        f.set_metadata("fixture_version", "1")

    return db_path


def _replay_all_exchanges(db_path: Path) -> list[dict[str, Any]]:
    """Replay all 5 exchanges in the fixture and return the responses."""
    replayed: list[dict[str, Any]] = []

    with Fixture(db_path) as f:
        f.reset_read_cursor()
        transport = ReplayTransport(f)

        # Replay in recorded order: LLM1, Tool1, LLM2, Tool2, LLM3
        for url, method in [
            (_LLM_URL, "POST"),
            (_TOOL_URL, "GET"),
            (_LLM_URL, "POST"),
            (_TOOL_URL, "GET"),
            (_LLM_URL, "POST"),
        ]:
            request = httpx.Request(method, url)
            response = transport.handle_request(request)
            replayed.append(
                {
                    "url": url,
                    "method": method,
                    "status": response.status_code,
                    "body": response.text,
                }
            )

    return replayed


# ---------------------------------------------------------------------------
# Fidelity correctness test
# ---------------------------------------------------------------------------


def test_fidelity_exchange_count(benchmark: Any, tmp_path: Path) -> None:
    """Record 5 exchanges then replay — assert counts match."""
    db_path = _create_test_fixture(tmp_path)

    engine = ReplayEngine(db_path)
    recorded_count = engine.fixture_exchange_count()
    assert recorded_count == 5

    def _do_replay() -> list[dict[str, Any]]:
        return _replay_all_exchanges(db_path)

    replayed = benchmark(_do_replay)
    assert len(replayed) == recorded_count


def test_fidelity_response_bodies(tmp_path: Path) -> None:
    """Assert each replayed response body exactly matches what was recorded."""
    db_path = _create_test_fixture(tmp_path)

    # Collect recorded bodies in order
    with Fixture(db_path) as f:
        recorded = f.all_exchanges()

    recorded_bodies = [ex["response_body"] for ex in recorded]

    replayed = _replay_all_exchanges(db_path)
    replayed_bodies = [r["body"] for r in replayed]

    assert len(replayed_bodies) == len(recorded_bodies) == 5

    for i, (recorded_body, replayed_body) in enumerate(
        zip(recorded_bodies, replayed_bodies, strict=True)
    ):
        assert recorded_body == replayed_body, (
            f"Exchange {i}: body mismatch.\n"
            f"  recorded: {recorded_body!r}\n"
            f"  replayed: {replayed_body!r}"
        )


def test_fidelity_pct_is_100(tmp_path: Path) -> None:
    """Compute fidelity percentage: must be 100%."""
    db_path = _create_test_fixture(tmp_path)

    with Fixture(db_path) as f:
        recorded = f.all_exchanges()

    replayed = _replay_all_exchanges(db_path)

    matches = sum(
        1
        for r_ex, rep in zip(recorded, replayed, strict=False)
        if r_ex["response_body"] == rep["body"]
    )
    fidelity_pct = matches / len(recorded) * 100
    assert fidelity_pct == 100.0, f"Fidelity {fidelity_pct:.1f}% — expected 100%"


def test_replay_speed(benchmark: Any, tmp_path: Path) -> None:
    """Benchmark replay of a 5-exchange fixture.

    Target: < 10ms per full replay pass.
    The speed assertion is only applied when running in benchmark mode
    (benchmark.stats is None in --benchmark-disable mode).
    """
    db_path = _create_test_fixture(tmp_path)

    def _full_replay() -> list[dict[str, Any]]:
        return _replay_all_exchanges(db_path)

    result = benchmark(_full_replay)
    assert len(result) == 5

    # Assert target speed only when benchmark stats are available
    # (stats is None when running with --benchmark-disable)
    stats = getattr(benchmark, "stats", None)
    if stats is not None:
        mean_ms = stats.get("mean", 0.0) * 1000
        if mean_ms > 0:
            assert mean_ms < 50.0, (
                f"Replay too slow: {mean_ms:.2f}ms mean (target < 10ms, CI limit 50ms)"
            )
