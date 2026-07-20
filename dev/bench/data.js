window.BENCHMARK_DATA = {
  "lastUpdate": 1784591887685,
  "repoUrl": "https://github.com/RudrenduPaul/agent-observability",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "RudrenduPaul@users.noreply.github.com",
            "name": "Rudrendu",
            "username": "RudrenduPaul"
          },
          "committer": {
            "email": "RudrenduPaul@users.noreply.github.com",
            "name": "Rudrendu",
            "username": "RudrenduPaul"
          },
          "distinct": true,
          "id": "8c50bed5b7089dfb4a1c9159882dd457fabb019c",
          "message": "Fix coverage gate (mcp missing from dev extra) and a fragile Windows/py3.13 test\n\nCoverage gate — replay/ and interceptor/ each >=90%: was also never\nvalidated by CI (same root cause as lint/mypy). Failed for real once\nreachable: interceptor/stdio_hook.py had 0% coverage because mcp isn't\nin the dev extra, so its real unit test (tests/unit/test_stdio_hook.py)\nwas silently skipped rather than actually running. Added mcp to dev,\nmatching the existing pattern for grpc/aiohttp/boto3/websockets (each\nadded specifically for its own interceptor's unit tests). Coverage\ngate passes locally now: 92.88% (was 85.85%).\n\nAlso hardened 4 loop-guard tests in test_httpx_hook.py that asserted\n`len(recwarn) == 0` -- fragile to ANY unrelated warning landing in the\nrecorder (e.g. a delayed ResourceWarning from a prior test's GC), not\njust the loop-guard warning they're actually testing for. One of them\nfailed on windows-latest/Python 3.13 specifically with 10 unexplained\nwarnings captured; the loop-guard counting logic itself is correct\n(verified passing on 7 other platform/version combinations). Added a\n_loop_guard_warnings() filter so these assert on the specific warning\nthey claim to test, not total warning count.",
          "timestamp": "2026-07-20T16:52:25-07:00",
          "tree_id": "7c97d3fbd121b437c7dc7505edd9a04d9f9cafd8",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/8c50bed5b7089dfb4a1c9159882dd457fabb019c"
        },
        "date": 1784591886877,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 897.0075645818189,
            "unit": "iter/sec",
            "range": "stddev: 0.000019143554183484695",
            "extra": "mean: 1.1148178003004867 msec\nrounds: 666"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 903.0919806892377,
            "unit": "iter/sec",
            "range": "stddev: 0.000022759115043652475",
            "extra": "mean: 1.1073069204277535 msec\nrounds: 842"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 926204.9926910105,
            "unit": "iter/sec",
            "range": "stddev: 3.3315428505641136e-7",
            "extra": "mean: 1.0796745945998243 usec\nrounds: 151700"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 375443.1587814094,
            "unit": "iter/sec",
            "range": "stddev: 5.915330712010732e-7",
            "extra": "mean: 2.6635190350670905 usec\nrounds: 83320"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3381.028036547994,
            "unit": "iter/sec",
            "range": "stddev: 0.00016766871761676033",
            "extra": "mean: 295.7680294840125 usec\nrounds: 1628"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31764.40676407289,
            "unit": "iter/sec",
            "range": "stddev: 0.000003825440615117568",
            "extra": "mean: 31.481777935517727 usec\nrounds: 9862"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.652176876340663,
            "unit": "iter/sec",
            "range": "stddev: 0.00036181532138084227",
            "extra": "mean: 176.92298416666338 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.16948658024021,
            "unit": "iter/sec",
            "range": "stddev: 0.0035419212164371537",
            "extra": "mean: 193.4428080000032 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16936261.62449471,
            "unit": "iter/sec",
            "range": "stddev: 7.135684093622005e-9",
            "extra": "mean: 59.04490744012316 nsec\nrounds: 161787"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 400.34999198980944,
            "unit": "iter/sec",
            "range": "stddev: 0.00003606157527543158",
            "extra": "mean: 2.497814462365355 msec\nrounds: 372"
          }
        ]
      }
    ]
  }
}