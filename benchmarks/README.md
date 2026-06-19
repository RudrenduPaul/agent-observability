# agent-trace Benchmarks

This directory contains performance benchmarks for the agent-trace library.
All benchmarks use `pytest-benchmark` and can be run without any real API keys.

## Quick start

```bash
# Run all benchmarks
uv run pytest benchmarks/ -v --benchmark-only

# Run a specific benchmark file
uv run pytest benchmarks/test_overhead.py -v --benchmark-only

# Run with histogram output
uv run pytest benchmarks/ --benchmark-histogram
```

## What each benchmark measures

### `test_overhead.py` — Recording overhead

Measures the latency overhead introduced by `@tracer.instrument(record=True)`
versus a plain undecorated function call.

A synthetic 10-step workflow calls a mock LLM endpoint (served by `respx`)
repeatedly. The "baseline" benchmark measures raw throughput; the "instrumented"
benchmark measures the same workflow wrapped in a trace with SQLite recording
enabled.

**Target**: < 4% overhead on an M-series Mac.
**CI assertion**: < 10% overhead (GitHub Actions machines are noisier).

Metrics to watch:
- `mean`: average total time per 10-step run
- `stddev`: variability — high stddev on CI is normal
- `overhead_pct`: computed as `(instrumented_mean - baseline_mean) / baseline_mean * 100`

### `test_fidelity.py` — Replay fidelity

Verifies that a fixture replayed through `ReplayEngine` produces byte-for-byte
identical responses to what was recorded.

Creates a 5-exchange fixture (3 mock LLM calls + 2 tool calls), replays it,
and diffs every response body. Also benchmarks the raw replay speed.

**Target**: 100% exchange match.
**Target**: < 10ms to replay a 5-exchange fixture.

### `test_ingestion.py` — Span ingestion latency

Stress-tests the SQLite write path by writing 10,000 `Span` objects and
measuring P50 / P99 latency per write.

Also benchmarks:
- `Span.to_dict()` serialisation speed (no I/O)
- Single `record_exchange()` call latency
- `next_exchange()` read cursor speed on a 100-exchange fixture

**Target P99**: < 12ms on standard hardware.

## How to reproduce README numbers

The numbers in the main `README.md` were captured on GitHub Actions
`ubuntu-latest` (2-core, 7 GB RAM). To reproduce in under 5 minutes:

```bash
# 1. Install dev dependencies
uv sync --all-extras

# 2. Run benchmarks and save results
uv run pytest benchmarks/ \
  --benchmark-only \
  --benchmark-json=benchmarks/results/latest.json \
  -v

# 3. Print a summary
uv run pytest benchmarks/ --benchmark-only --benchmark-compare
```

The three key numbers from `baseline.json`:
| Metric | Baseline value | How to rerun |
|--------|---------------|--------------|
| `overhead_pct` | See `baseline.json` | `uv run pytest benchmarks/test_overhead.py --benchmark-only` |
| `fidelity_pct` | 100 | `uv run pytest benchmarks/test_fidelity.py --benchmark-only` |
| `p99_ingestion_ms` | See `baseline.json` | `uv run pytest benchmarks/test_ingestion.py --benchmark-only` |

## Interpreting results

- **mean / median**: the central tendency — this is the headline number.
- **stddev**: noise in the run. On CI, 20–30% stddev is normal and not alarming.
- **min**: the best-case floor — useful for comparing across machines.
- **rounds / iterations**: `pytest-benchmark` auto-calibrates these. Fewer rounds
  means higher variance; if results look unstable, pass `--benchmark-min-rounds=10`.

## Warning: environment differences

All numbers in `benchmarks/results/baseline.json` were captured on
**GitHub Actions `ubuntu-latest`**. Local M-series Mac results will typically
be 2–5x faster for CPU-bound work and similar for I/O-bound (SQLite) work.

Do NOT compare local numbers directly to the baseline JSON — run the full
suite on your machine and compare relative percentages (overhead_pct, fidelity_pct)
rather than absolute timings.
