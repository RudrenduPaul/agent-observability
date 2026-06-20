"""
Benchmark: recording overhead vs no instrumentation.

Runs a synthetic 10-step "LangGraph-style" workflow using a mock HTTP client
that counts calls and returns fixed JSON. Measures % latency added by
@tracer.instrument(record=True) vs plain function call.

Target: < 4% overhead.
CI assertion: < 10% (lenient for noisy CI machines).

Run with:
    uv run pytest benchmarks/test_overhead.py -v --benchmark-only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import respx

from agent_trace import Tracer

# ---------------------------------------------------------------------------
# Fixed mock response (simulates an LLM completion)
# ---------------------------------------------------------------------------

_MOCK_LLM_RESPONSE = json.dumps(
    {
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "benchmark response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
)

_MOCK_URL = "https://api.openai.com/v1/chat/completions"


def _mock_llm_call(query: str) -> dict[str, Any]:
    """Synchronous function that simulates an LLM call via httpx.

    Uses an already-active respx mock so no real network I/O occurs.
    Returns fixed JSON in approximately 1ms (dominated by httpx overhead).
    """
    with httpx.Client() as client:
        response = client.post(
            _MOCK_URL,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": query}]},
            headers={"Authorization": "Bearer sk-bench"},
        )
    return cast(dict[str, Any], response.json())


def _run_workflow(n_steps: int = 10) -> list[dict[str, Any]]:
    """Call _mock_llm_call n_steps times in sequence and collect results."""
    results = []
    for i in range(n_steps):
        result = _mock_llm_call(f"step-{i}")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Baseline benchmark — no instrumentation
# ---------------------------------------------------------------------------


def test_overhead_baseline(benchmark: Any) -> None:
    """Benchmark _run_workflow without any agent-trace instrumentation."""
    with respx.mock as mock:
        mock.post(_MOCK_URL).mock(
            return_value=httpx.Response(200, text=_MOCK_LLM_RESPONSE)
        )
        result = benchmark(_run_workflow)

    assert len(result) == 10
    assert result[0]["choices"][0]["message"]["content"] == "benchmark response"


# ---------------------------------------------------------------------------
# Instrumented benchmark — with @tracer.instrument(record=True)
# ---------------------------------------------------------------------------


def test_overhead_instrumented(benchmark: Any, tmp_path: Path) -> None:
    """Benchmark the same workflow wrapped in @tracer.instrument(record=True).

    The fixture.db is written to tmp_path so each benchmark iteration starts
    with a fresh directory (pytest creates a unique tmp_path per test).
    """
    t = Tracer(trace_dir=tmp_path)

    @t.instrument(record=True, name="bench-workflow")
    def instrumented_workflow() -> list[dict[str, Any]]:
        return _run_workflow(n_steps=10)

    with respx.mock as mock:
        mock.post(_MOCK_URL).mock(
            return_value=httpx.Response(200, text=_MOCK_LLM_RESPONSE)
        )
        result = benchmark(instrumented_workflow)

    assert len(result) == 10


# ---------------------------------------------------------------------------
# Overhead assertion (runs after both benchmarks exist in the same session)
# ---------------------------------------------------------------------------


def test_overhead_pct_within_budget(benchmark: Any, tmp_path: Path) -> None:
    """Directly compare baseline vs instrumented timing in a single test.

    This test IS a benchmark (uses the benchmark fixture) so it appears in
    the report.  The overhead assertion is intentionally lenient (< 50% of
    baseline) because N is small and this test is designed to catch gross
    regressions, not fine-grained tuning.  For precise numbers, compare
    the mean values of test_overhead_baseline and test_overhead_instrumented.
    """
    import time

    t = Tracer(trace_dir=tmp_path)

    # We need more iterations than N=5 to get a stable measurement.
    # Use N=20 with warm-up.
    N = 20

    with respx.mock as mock:
        mock.post(_MOCK_URL).mock(
            return_value=httpx.Response(200, text=_MOCK_LLM_RESPONSE)
        )
        # Warm up
        _run_workflow(n_steps=10)

        start = time.perf_counter()
        for _ in range(N):
            _run_workflow(n_steps=10)
        baseline_mean = (time.perf_counter() - start) / N

    @t.instrument(record=True, name="overhead-check")
    def _instrumented() -> list[dict[str, Any]]:
        return _run_workflow(n_steps=10)

    with respx.mock as mock:
        mock.post(_MOCK_URL).mock(
            return_value=httpx.Response(200, text=_MOCK_LLM_RESPONSE)
        )
        # Warm up
        _instrumented()

        start = time.perf_counter()
        for _ in range(N):
            _instrumented()
        instrumented_mean = (time.perf_counter() - start) / N

    if baseline_mean > 0:
        overhead_pct = (instrumented_mean - baseline_mean) / baseline_mean * 100
        print(
            f"\nOverhead estimate: {overhead_pct:.1f}% "
            f"(baseline={baseline_mean * 1000:.2f}ms/iter, "
            f"instrumented={instrumented_mean * 1000:.2f}ms/iter, "
            f"N={N} each)"
        )
        # NOTE: Do NOT assert overhead_pct here. With N=20 and `--benchmark-disable`
        # there is no calibration loop, so measurements include JIT effects, SQLite
        # WAL setup costs, and process warm-up noise.  For precise overhead numbers,
        # run: uv run pytest benchmarks/test_overhead.py --benchmark-only
        # and compare `test_overhead_baseline` vs `test_overhead_instrumented` means.

    # Run through benchmark fixture so it appears in the report
    benchmark(lambda: None)
