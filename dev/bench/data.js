window.BENCHMARK_DATA = {
  "lastUpdate": 1784600674675,
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
      },
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
          "id": "038ce4c1e5f944ebf6013a02faca631c195b9f5f",
          "message": "Fix demo-1-record-replay.gif: remove leaked private repo path\n\nThe recorded command referenced an absolute scratchpad path inside a\nprivate, unrelated repo. Rebuilt the terminal mockup with the real\nagent-trace CLI syntax (agent-trace run --name ... -- <cmd>, then\nagent-trace replay <run_id>) cd'd into this public repo instead, and\nmatched the '>' prompt style already used by demo-2/demo-3. Same\nrecord/replay content as before, no leaked path.",
          "timestamp": "2026-07-20T19:23:57-07:00",
          "tree_id": "c67492f63054f762b56e16d611362d823b0c2a66",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/038ce4c1e5f944ebf6013a02faca631c195b9f5f"
        },
        "date": 1784600674389,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 942.2202374066458,
            "unit": "iter/sec",
            "range": "stddev: 0.00004013404836457129",
            "extra": "mean: 1.0613229904214183 msec\nrounds: 522"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 949.4736598902748,
            "unit": "iter/sec",
            "range": "stddev: 0.00003262584582812166",
            "extra": "mean: 1.0532151045828528 msec\nrounds: 851"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 964266.5190696772,
            "unit": "iter/sec",
            "range": "stddev: 4.373492915471483e-7",
            "extra": "mean: 1.037057680862754 usec\nrounds: 122519"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 353939.2006648458,
            "unit": "iter/sec",
            "range": "stddev: 6.089118330958627e-7",
            "extra": "mean: 2.825344008579953 usec\nrounds: 61638"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 4120.342993651645,
            "unit": "iter/sec",
            "range": "stddev: 0.00011535638145930198",
            "extra": "mean: 242.6982417582067 usec\nrounds: 2002"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 32516.814857153902,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027909802704910888",
            "extra": "mean: 30.753319609961544 usec\nrounds: 8307"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.945156796544792,
            "unit": "iter/sec",
            "range": "stddev: 0.00041790440397424195",
            "extra": "mean: 168.2041423333326 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.482484466520621,
            "unit": "iter/sec",
            "range": "stddev: 0.0005276063499013042",
            "extra": "mean: 182.3990576 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16049669.5637403,
            "unit": "iter/sec",
            "range": "stddev: 8.182890654192334e-9",
            "extra": "mean: 62.30657871357164 nsec\nrounds: 149701"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 479.20137822922027,
            "unit": "iter/sec",
            "range": "stddev: 0.00024290507889896456",
            "extra": "mean: 2.0868053503837416 msec\nrounds: 391"
          }
        ]
      }
    ]
  }
}