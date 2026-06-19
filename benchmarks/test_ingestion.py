"""
Benchmark: P99 span ingestion latency.

Writes 10,000 Span objects to the SQLite backend sequentially.
Reports P50 and P99 write latency per span.
Target P99: < 12ms on standard hardware.

Run with:
    uv run pytest benchmarks/test_ingestion.py -v --benchmark-only
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from agent_trace._replay.fixture import Fixture
from agent_trace.core.span import Span

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(i: int = 0) -> Span:
    s = Span(
        name=f"bench-span-{i}",
        span_id=uuid.uuid4().hex[:16],
        trace_id="bench-trace-001",
    )
    s.set_attribute("step", i)
    s.set_attribute("env", "benchmark")
    s.add_event("checkpoint", {"index": i})
    s.end()
    return s


def _make_exchange(i: int = 0) -> dict[str, Any]:
    return dict(
        url=f"https://api.example.com/v1/bench/{i}",
        method="POST",
        request_headers={
            "content-type": "application/json",
            "authorization": "Bearer sk-bench",
        },
        request_body=json.dumps({"model": "gpt-4o", "step": i}),
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body=json.dumps({"choices": [{"message": {"content": f"r-{i}"}}]}),
    )


# ---------------------------------------------------------------------------
# Span serialisation speed
# ---------------------------------------------------------------------------


def test_span_serialization_speed(benchmark: Any) -> None:
    """Benchmark Span.to_dict() for a typical span with attributes and events.

    No I/O — pure CPU/serialisation overhead.
    Target: << 1ms per call.
    """
    span = _make_span(42)

    def _serialize() -> dict[str, Any]:
        return span.to_dict()

    result = benchmark(_serialize)
    assert "span_id" in result
    assert result["name"] == "bench-span-42"


def test_span_from_dict_speed(benchmark: Any) -> None:
    """Benchmark Span.from_dict() round-trip deserialisation."""
    span = _make_span(0)
    d = span.to_dict()

    def _deserialize() -> Span:
        return Span.from_dict(d)

    result = benchmark(_deserialize)
    assert result.name == span.name


# ---------------------------------------------------------------------------
# Single record_exchange() latency
# ---------------------------------------------------------------------------


def test_fixture_write_latency(benchmark: Any, tmp_path: Path) -> None:
    """Benchmark a single record_exchange() call on an open Fixture.

    Target: P99 < 12ms (benchmark.stats will show the distribution).
    """
    db_path = tmp_path / "write_bench.db"
    counter = [0]

    with Fixture(db_path) as f:

        def _write_one() -> None:
            ex = _make_exchange(counter[0])
            f.record_exchange(**ex)
            counter[0] += 1

        benchmark(_write_one)

    assert counter[0] > 0


# ---------------------------------------------------------------------------
# Bulk write: 10,000 exchanges
# ---------------------------------------------------------------------------


def test_fixture_bulk_write_10k(tmp_path: Path) -> None:
    """Write 10,000 exchanges sequentially and assert count == 10,000.

    Also measures total time and computes per-exchange P99.
    """
    count = 10_000
    db_path = tmp_path / "bulk_10k.db"

    latencies: list[float] = []

    with Fixture(db_path) as f:
        for i in range(count):
            ex = _make_exchange(i)
            t0 = time.perf_counter()
            f.record_exchange(**ex)
            latencies.append((time.perf_counter() - t0) * 1_000)  # ms

        final_count = f.exchange_count()

    assert final_count == count, f"Expected {count} exchanges, got {final_count}"

    p50 = statistics.median(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98]  # index 98 = 99th percentile

    total_ms = sum(latencies)
    p50_write_ms = p50
    print(f"\n10k write — P50: {p50:.3f}ms  P99: {p99:.3f}ms  total: {total_ms:.1f}ms")

    # Lenient assertion for CI (target is 12ms but CI SQLite can be slower)
    assert p99 < 100.0, f"P99 too high: {p99:.2f}ms (target < 12ms, CI limit 100ms)"

    writes_per_sec = int(1000 / p50_write_ms) if p50_write_ms > 0 else 0
    print(f"  Writes/sec (derived from P50): {writes_per_sec:,}")
    assert writes_per_sec >= 3000, f"Write throughput too low: {writes_per_sec}/sec"


# ---------------------------------------------------------------------------
# Read cursor speed
# ---------------------------------------------------------------------------


def test_fixture_read_cursor_speed(benchmark: Any, tmp_path: Path) -> None:
    """Pre-fill 100 exchanges, then benchmark next_exchange() call speed.

    Measures how fast the replay engine can serve requests from the fixture.
    """
    db_path = tmp_path / "read_bench.db"
    url = "https://api.example.com/v1/bench/read"
    method = "POST"

    # Pre-fill
    with Fixture(db_path) as f:
        for i in range(100):
            ex = _make_exchange(i)
            ex["url"] = url  # same URL so next_exchange() can serve them all
            f.record_exchange(**ex)

    # Open fresh for reading
    f = Fixture(db_path)
    call_counter = [0]
    total = [100]

    def _next() -> None:
        if call_counter[0] >= total[0]:
            f.reset_read_cursor()
            call_counter[0] = 0
        f.next_exchange(url, method)
        call_counter[0] += 1

    benchmark(_next)
    f.close()

    stats = getattr(benchmark, "stats", None)
    if stats is not None:
        read_cursor_mean_us = stats.get("mean", 0.0) * 1_000_000  # convert s → µs
        if read_cursor_mean_us > 0:
            print(f"  Read cursor mean latency : {read_cursor_mean_us:.2f} µs")
            reads_per_sec = int(1_000_000 / read_cursor_mean_us)
            print(f"  Reads/sec (derived from mean): {reads_per_sec:,}")
            assert reads_per_sec >= 20_000, (
                f"Read throughput too low: {reads_per_sec}/sec"
            )
