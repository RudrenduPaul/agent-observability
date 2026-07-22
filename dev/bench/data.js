window.BENCHMARK_DATA = {
  "lastUpdate": 1784689603692,
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
          "id": "00947952c4e1f61af2ff747aea8f89f3328bea12",
          "message": "Fix chronically flaky P99 assertion in recording-overhead benchmark\n\ntest_recording_overhead_per_exchange has been failing intermittently\non CI for a while (10+ failures across unrelated commits going back\nthrough this session's history), most recently on the json-repair\nfix push. The P99 write latency on a single SQLite WAL fsync, over\nonly 200 samples, is dominated by shared-runner I/O jitter rather\nthan actual code performance -- observed failures ranged from 8ms to\n262ms with no code change in between. Report P99 but stop asserting\non it; P50 (already asserted, never the failing check) is the\nstable, meaningful regression signal.",
          "timestamp": "2026-07-20T19:45:18-07:00",
          "tree_id": "2e935ec3444b02a1ee66342e072d1b17cc6fd704",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/00947952c4e1f61af2ff747aea8f89f3328bea12"
        },
        "date": 1784601956313,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 1329.3086437666734,
            "unit": "iter/sec",
            "range": "stddev: 0.000038877435456370805",
            "extra": "mean: 752.2707421554425 usec\nrounds: 733"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 1361.3098458979407,
            "unit": "iter/sec",
            "range": "stddev: 0.000033594328167459804",
            "extra": "mean: 734.5866211232644 usec\nrounds: 1193"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 1189773.8414807557,
            "unit": "iter/sec",
            "range": "stddev: 3.738803396746666e-7",
            "extra": "mean: 840.4958700011683 nsec\nrounds: 114286"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 469513.7020281133,
            "unit": "iter/sec",
            "range": "stddev: 4.526404141742932e-7",
            "extra": "mean: 2.1298632940431683 usec\nrounds: 79777"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3317.0579566872584,
            "unit": "iter/sec",
            "range": "stddev: 0.0005694988693444944",
            "extra": "mean: 301.47197096269576 usec\nrounds: 1963"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 48504.89948989109,
            "unit": "iter/sec",
            "range": "stddev: 0.000001949994220715875",
            "extra": "mean: 20.616474016370454 usec\nrounds: 9506"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 6.674660774383125,
            "unit": "iter/sec",
            "range": "stddev: 0.0009084949700997256",
            "extra": "mean: 149.82034799999562 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 3.5639826327240107,
            "unit": "iter/sec",
            "range": "stddev: 0.21762435381049197",
            "extra": "mean: 280.58498120000195 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 25590271.299118537,
            "unit": "iter/sec",
            "range": "stddev: 4.347869185795725e-9",
            "extra": "mean: 39.07735046304278 nsec\nrounds: 199721"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 612.07018932685,
            "unit": "iter/sec",
            "range": "stddev: 0.00009792340907092537",
            "extra": "mean: 1.633799550178701 msec\nrounds: 558"
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
          "id": "316d401de15ba8ac9b099d9e58c8cea34171b8c9",
          "message": "Stop hard-failing CI on single-sample benchmark noise\n\nalert-threshold 120% + fail-on-alert true was comparing one run\nagainst exactly one prior run with no statistical tolerance, and has\nbeen failing repeatedly on commits unrelated to the flagged tests\n(observed ratios 1.24-1.54 from shared-runner variance alone). Widen\nthe threshold to 160% and stop hard-failing; comment-on-alert stays\non so real regressions are still visible.",
          "timestamp": "2026-07-20T19:49:56-07:00",
          "tree_id": "fbce8d3e2c90281bf8caa7fcaec2ac83010e9506",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/316d401de15ba8ac9b099d9e58c8cea34171b8c9"
        },
        "date": 1784602232860,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 957.8787977127474,
            "unit": "iter/sec",
            "range": "stddev: 0.000022649392762956754",
            "extra": "mean: 1.0439734154131304 msec\nrounds: 532"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 970.4268414882575,
            "unit": "iter/sec",
            "range": "stddev: 0.000026411092731632653",
            "extra": "mean: 1.0304743822485256 msec\nrounds: 845"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 934766.7919185954,
            "unit": "iter/sec",
            "range": "stddev: 4.670796386486171e-7",
            "extra": "mean: 1.0697855429240424 usec\nrounds: 123717"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 382529.07352791977,
            "unit": "iter/sec",
            "range": "stddev: 5.226141101377693e-7",
            "extra": "mean: 2.61418038314678 usec\nrounds: 73100"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 4340.390328625364,
            "unit": "iter/sec",
            "range": "stddev: 0.00009340390936988449",
            "extra": "mean: 230.3940254877279 usec\nrounds: 2511"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 33059.15313517688,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028675325259916797",
            "extra": "mean: 30.248808731157162 usec\nrounds: 8773"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.968742198480555,
            "unit": "iter/sec",
            "range": "stddev: 0.0002706614744862953",
            "extra": "mean: 167.53948600000967 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.259267791505575,
            "unit": "iter/sec",
            "range": "stddev: 0.01461177775650172",
            "extra": "mean: 190.14053660000627 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16290774.432123723,
            "unit": "iter/sec",
            "range": "stddev: 8.369952167833733e-9",
            "extra": "mean: 61.3844359681332 nsec\nrounds: 151516"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 500.1134827465725,
            "unit": "iter/sec",
            "range": "stddev: 0.0000897932092443193",
            "extra": "mean: 1.9995461720170018 msec\nrounds: 436"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "38769913+RudrenduPaul@users.noreply.github.com",
            "name": "Rudrendu Paul",
            "username": "RudrenduPaul"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bb877abcd73d21753314f20474e7adeca7ac74da",
          "message": "Document all _cli.py subcommand flags in README CLI reference (#18)\n\ninspect had 7 undocumented flags (--registered-tools, --configured-host,\n--check-kwarg, --diff-field, --diff-get-post-field,\n--diff-get-post-id-field, --diff-get-post-post-id-field) and run had 2\n(--run-id, --name), none previously listed in README.md or npm/README.md.\nAdds a complete CLI reference covering all 7 subcommands (version, list,\nshow, replay, inspect, diff, run) with every flag, default, and behavior,\nverified directly against argparse definitions in src/agent_trace/_cli.py.\nAlso corrects an existing inaccuracy: README claimed `show` supports\n--json, but it has no such flag.\n\nCo-authored-by: Rudrendu <RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-07-21T19:26:11-07:00",
          "tree_id": "a40f6597b788ddabb4c3404bd7750ba2f1efe841",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/bb877abcd73d21753314f20474e7adeca7ac74da"
        },
        "date": 1784687207111,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 940.0852441757709,
            "unit": "iter/sec",
            "range": "stddev: 0.00004623606959615255",
            "extra": "mean: 1.063733322265642 msec\nrounds: 512"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 962.8049104095506,
            "unit": "iter/sec",
            "range": "stddev: 0.000037247320612441406",
            "extra": "mean: 1.0386320106890892 msec\nrounds: 842"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 934287.243285838,
            "unit": "iter/sec",
            "range": "stddev: 4.344188987071899e-7",
            "extra": "mean: 1.0703346397870679 usec\nrounds: 115835"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 379424.8336568058,
            "unit": "iter/sec",
            "range": "stddev: 6.424474174580671e-7",
            "extra": "mean: 2.6355681318015987 usec\nrounds: 60405"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3744.2064047368744,
            "unit": "iter/sec",
            "range": "stddev: 0.00016102883739441907",
            "extra": "mean: 267.0792931540523 usec\nrounds: 1709"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 32502.840007632636,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035509838556563344",
            "extra": "mean: 30.766542239544922 usec\nrounds: 8144"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.902331552290075,
            "unit": "iter/sec",
            "range": "stddev: 0.0005651345458091877",
            "extra": "mean: 169.42457249999876 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.703779119533583,
            "unit": "iter/sec",
            "range": "stddev: 0.06017168080500136",
            "extra": "mean: 212.59501660000524 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16227461.445180295,
            "unit": "iter/sec",
            "range": "stddev: 8.329005476492638e-9",
            "extra": "mean: 61.62393319363019 nsec\nrounds: 153847"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 498.5082765856667,
            "unit": "iter/sec",
            "range": "stddev: 0.000042823641626266156",
            "extra": "mean: 2.0059847488372724 msec\nrounds: 430"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "38769913+RudrenduPaul@users.noreply.github.com",
            "name": "Rudrendu Paul",
            "username": "RudrenduPaul"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b255b993b1ce63bb5f22e81d6f5344b88d4f1a27",
          "message": "Add --json to the replay command (#19)\n\nreplay was the one data-returning subcommand with no structured\noutput mode, unlike list/inspect/diff/run. Adds a JSON summary\n(fixture path, span/exchange counts, the original trace) alongside\nthe existing human-readable span tree, gated the same way the other\nsubcommands already do.\n\nVerified: 113/113 CLI unit tests pass; both modes smoke-tested\nagainst a real recorded run in a fresh venv.\n\nCo-authored-by: Rudrendu <RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-07-21T19:54:37-07:00",
          "tree_id": "db5af0e3dbd4e528e592b883305503fbd25eba39",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/b255b993b1ce63bb5f22e81d6f5344b88d4f1a27"
        },
        "date": 1784688959784,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 1106.1127562810861,
            "unit": "iter/sec",
            "range": "stddev: 0.000026187812692807827",
            "extra": "mean: 904.0669627228125 usec\nrounds: 617"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 1115.5185693151489,
            "unit": "iter/sec",
            "range": "stddev: 0.00007949528501452973",
            "extra": "mean: 896.4440642291869 usec\nrounds: 1012"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 834471.3581001633,
            "unit": "iter/sec",
            "range": "stddev: 3.959848475224992e-7",
            "extra": "mean: 1.1983634792171838 usec\nrounds: 87928"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 364008.77842440864,
            "unit": "iter/sec",
            "range": "stddev: 7.582896338547169e-7",
            "extra": "mean: 2.7471864945907165 usec\nrounds: 71675"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 1939.3776497406802,
            "unit": "iter/sec",
            "range": "stddev: 0.003287247048296535",
            "extra": "mean: 515.6293309524902 usec\nrounds: 840"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 37351.404000269555,
            "unit": "iter/sec",
            "range": "stddev: 0.000003704789361682357",
            "extra": "mean: 26.772755315778312 usec\nrounds: 9547"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.555863062287465,
            "unit": "iter/sec",
            "range": "stddev: 0.0003245003537186751",
            "extra": "mean: 179.99003733333177 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.062444836879977,
            "unit": "iter/sec",
            "range": "stddev: 0.0012336835208799804",
            "extra": "mean: 197.53301660000062 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 17342817.41880747,
            "unit": "iter/sec",
            "range": "stddev: 6.790187856571973e-9",
            "extra": "mean: 57.66075810240308 nsec\nrounds: 167141"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 497.38887707002965,
            "unit": "iter/sec",
            "range": "stddev: 0.00004379516477685291",
            "extra": "mean: 2.0104993217594718 msec\nrounds: 432"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "38769913+RudrenduPaul@users.noreply.github.com",
            "name": "Rudrendu Paul",
            "username": "RudrenduPaul"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f471dabe993b17d13a95e2192fd129d099d4ca36",
          "message": "Add Sourav Nandy as npm contributor (#20)\n\n* Add --json to the replay command\n\nreplay was the one data-returning subcommand with no structured\noutput mode, unlike list/inspect/diff/run. Adds a JSON summary\n(fixture path, span/exchange counts, the original trace) alongside\nthe existing human-readable span tree, gated the same way the other\nsubcommands already do.\n\nVerified: 113/113 CLI unit tests pass; both modes smoke-tested\nagainst a real recorded run in a fresh venv.\n\n* Add Sourav Nandy as npm contributor, matching PyPI's author listing\n\n---------\n\nCo-authored-by: Rudrendu <RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-07-21T20:01:35-07:00",
          "tree_id": "c67bb1a5a291b7790982c5384bd249fd1dc67d8e",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/f471dabe993b17d13a95e2192fd129d099d4ca36"
        },
        "date": 1784689338223,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 958.5584355570244,
            "unit": "iter/sec",
            "range": "stddev: 0.000022544965216741118",
            "extra": "mean: 1.0432332165736915 msec\nrounds: 531"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 964.6538791252476,
            "unit": "iter/sec",
            "range": "stddev: 0.00002156496119314993",
            "extra": "mean: 1.0366412468136286 msec\nrounds: 863"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 971099.1353319694,
            "unit": "iter/sec",
            "range": "stddev: 3.345234595185693e-7",
            "extra": "mean: 1.0297609828044496 usec\nrounds: 146843"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 376966.2255522086,
            "unit": "iter/sec",
            "range": "stddev: 5.56350255679586e-7",
            "extra": "mean: 2.6527575475365848 usec\nrounds: 65386"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 4275.436894426954,
            "unit": "iter/sec",
            "range": "stddev: 0.00008748307798511317",
            "extra": "mean: 233.89422524362445 usec\nrounds: 2353"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 32911.00124135539,
            "unit": "iter/sec",
            "range": "stddev: 0.000002824751880704947",
            "extra": "mean: 30.38497652096398 usec\nrounds: 8859"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.972970040836491,
            "unit": "iter/sec",
            "range": "stddev: 0.0002934765367589128",
            "extra": "mean: 167.4208966666697 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.533813613828711,
            "unit": "iter/sec",
            "range": "stddev: 0.00045664389673365794",
            "extra": "mean: 180.70720659999324 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16268245.751561007,
            "unit": "iter/sec",
            "range": "stddev: 7.99494228129163e-9",
            "extra": "mean: 61.46944269661318 nsec\nrounds: 154799"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 487.82252195454004,
            "unit": "iter/sec",
            "range": "stddev: 0.00021639761334513928",
            "extra": "mean: 2.049925854167901 msec\nrounds: 432"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "38769913+RudrenduPaul@users.noreply.github.com",
            "name": "Rudrendu Paul",
            "username": "RudrenduPaul"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a6f798df75922dee0a80f73cb81ccf979efb526d",
          "message": "Add missing PyPI Environment classifier (#21)\n\nCo-authored-by: Rudrendu <RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-07-21T20:04:00-07:00",
          "tree_id": "336505d63512fdc3eb1b1b6194bdbf63c1ad920c",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/a6f798df75922dee0a80f73cb81ccf979efb526d"
        },
        "date": 1784689602845,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 869.7299647910129,
            "unit": "iter/sec",
            "range": "stddev: 0.00003659261697518906",
            "extra": "mean: 1.1497821628352078 msec\nrounds: 522"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 892.2262906703752,
            "unit": "iter/sec",
            "range": "stddev: 0.000023295168169512373",
            "extra": "mean: 1.1207919005039058 msec\nrounds: 794"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 918762.7282464231,
            "unit": "iter/sec",
            "range": "stddev: 3.457707948316436e-7",
            "extra": "mean: 1.088420295312402 usec\nrounds: 133085"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 369119.67471588636,
            "unit": "iter/sec",
            "range": "stddev: 7.983622540806829e-7",
            "extra": "mean: 2.7091484645723805 usec\nrounds: 66366"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 2935.2551984444262,
            "unit": "iter/sec",
            "range": "stddev: 0.0002679931303979946",
            "extra": "mean: 340.68587989554095 usec\nrounds: 1532"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31893.265971176992,
            "unit": "iter/sec",
            "range": "stddev: 0.000003913132581360479",
            "extra": "mean: 31.35458127442117 usec\nrounds: 8459"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.302863077377049,
            "unit": "iter/sec",
            "range": "stddev: 0.0033378130334456736",
            "extra": "mean: 188.57737516666737 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.8966537765487494,
            "unit": "iter/sec",
            "range": "stddev: 0.002922136735000673",
            "extra": "mean: 204.22109580000125 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16920819.3351524,
            "unit": "iter/sec",
            "range": "stddev: 8.1801471787635e-9",
            "extra": "mean: 59.09879304263568 nsec\nrounds: 158680"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 397.85805471767344,
            "unit": "iter/sec",
            "range": "stddev: 0.00004600059723245016",
            "extra": "mean: 2.513459230351931 msec\nrounds: 369"
          }
        ]
      }
    ]
  }
}