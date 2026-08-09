window.BENCHMARK_DATA = {
  "lastUpdate": 1786311953892,
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
          "id": "d7e5a9df3b587d90fb2944fa67d482d344f7426e",
          "message": "Merge pull request #30 from RudrenduPaul/fix/repo-sanity-2026-08-02\n\nFix npm wrapper fork bomb, CI lint failure, and LICENSE drift",
          "timestamp": "2026-08-02T18:26:46-07:00",
          "tree_id": "0bdd5a971a3abcdc0276453fec2123e7b78e4f13",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/d7e5a9df3b587d90fb2944fa67d482d344f7426e"
        },
        "date": 1785720448870,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 1324.61882245291,
            "unit": "iter/sec",
            "range": "stddev: 0.00002866182286830667",
            "extra": "mean: 754.9341614731205 usec\nrounds: 706"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 1353.945743205281,
            "unit": "iter/sec",
            "range": "stddev: 0.000027160412455951247",
            "extra": "mean: 738.5820333041095 usec\nrounds: 1141"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 1231202.4248542215,
            "unit": "iter/sec",
            "range": "stddev: 2.2117031851692285e-7",
            "extra": "mean: 812.214124836867 nsec\nrounds: 162947"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 496583.98217277846,
            "unit": "iter/sec",
            "range": "stddev: 4.604348257526382e-7",
            "extra": "mean: 2.013758066912569 usec\nrounds: 80328"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 2998.6938806120083,
            "unit": "iter/sec",
            "range": "stddev: 0.0005408237797841239",
            "extra": "mean: 333.47852092055103 usec\nrounds: 1912"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 49012.04844069118,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018075777718147136",
            "extra": "mean: 20.403146406134944 usec\nrounds: 11311"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 6.784242657488696,
            "unit": "iter/sec",
            "range": "stddev: 0.0006791521183449747",
            "extra": "mean: 147.40038800000224 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.578847778119193,
            "unit": "iter/sec",
            "range": "stddev: 0.00976741385053676",
            "extra": "mean: 179.24848279999708 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 25345376.74359894,
            "unit": "iter/sec",
            "range": "stddev: 4.150080880190001e-9",
            "extra": "mean: 39.45492742586883 nsec\nrounds: 194440"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 616.0657021039581,
            "unit": "iter/sec",
            "range": "stddev: 0.000038139637335044287",
            "extra": "mean: 1.6232034936936888 msec\nrounds: 555"
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
          "id": "dd4cc8951cd5c12b2c745d794c6991e9ea7ea9e9",
          "message": "Remove stale version pin from SLSA provenance claim in README",
          "timestamp": "2026-08-08T19:54:33-07:00",
          "tree_id": "7ffefbe8052ecbacd3edafd00c1e3c0910885097",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/dd4cc8951cd5c12b2c745d794c6991e9ea7ea9e9"
        },
        "date": 1786244116460,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 875.7802050263267,
            "unit": "iter/sec",
            "range": "stddev: 0.00002342150483938794",
            "extra": "mean: 1.14183900739106 msec\nrounds: 541"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 885.8181642533386,
            "unit": "iter/sec",
            "range": "stddev: 0.00002320537474280943",
            "extra": "mean: 1.1288998581812848 msec\nrounds: 825"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 892398.4796598788,
            "unit": "iter/sec",
            "range": "stddev: 3.7872253635017955e-7",
            "extra": "mean: 1.1205756428239677 usec\nrounds: 117427"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 370013.1389556107,
            "unit": "iter/sec",
            "range": "stddev: 6.521619153117863e-7",
            "extra": "mean: 2.7026067312706075 usec\nrounds: 74997"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3424.4203636181087,
            "unit": "iter/sec",
            "range": "stddev: 0.00014816580869486443",
            "extra": "mean: 292.02022351702146 usec\nrounds: 1973"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31126.23191918538,
            "unit": "iter/sec",
            "range": "stddev: 0.000004440776975253415",
            "extra": "mean: 32.12724246855035 usec\nrounds: 9494"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.613605696125321,
            "unit": "iter/sec",
            "range": "stddev: 0.0002647351070049384",
            "extra": "mean: 178.138625000012 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.723388345983477,
            "unit": "iter/sec",
            "range": "stddev: 0.03830808105689293",
            "extra": "mean: 211.71242479995271 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16493289.305939365,
            "unit": "iter/sec",
            "range": "stddev: 8.482261745676649e-9",
            "extra": "mean: 60.63071964910554 nsec\nrounds: 160231"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 393.1710018809826,
            "unit": "iter/sec",
            "range": "stddev: 0.000039358518737800075",
            "extra": "mean: 2.543422569863664 msec\nrounds: 365"
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
          "id": "07c7c0b84271e79640715b93b31b5e7d58ab2487",
          "message": "Add missing npm badge for agent-observability-trace-cli",
          "timestamp": "2026-08-08T19:55:13-07:00",
          "tree_id": "bbd8441cd007c01ac8cf14c0ebec00ff1afe0321",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/07c7c0b84271e79640715b93b31b5e7d58ab2487"
        },
        "date": 1786244161392,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 878.1267730982964,
            "unit": "iter/sec",
            "range": "stddev: 0.00002249164249797551",
            "extra": "mean: 1.1387877361621694 msec\nrounds: 542"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 884.804194071157,
            "unit": "iter/sec",
            "range": "stddev: 0.00003301689797325295",
            "extra": "mean: 1.1301935577393734 msec\nrounds: 814"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 887154.6251106241,
            "unit": "iter/sec",
            "range": "stddev: 3.974446350681096e-7",
            "extra": "mean: 1.127199218372225 usec\nrounds: 120265"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 366718.5545690728,
            "unit": "iter/sec",
            "range": "stddev: 6.073993492055653e-7",
            "extra": "mean: 2.7268868388050063 usec\nrounds: 70837"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3461.6291150740503,
            "unit": "iter/sec",
            "range": "stddev: 0.00013707691438120525",
            "extra": "mean: 288.8813234339255 usec\nrounds: 2155"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 29670.617614746112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000074654256121223944",
            "extra": "mean: 33.703376619400274 usec\nrounds: 6869"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.533232880257244,
            "unit": "iter/sec",
            "range": "stddev: 0.0003663045621207946",
            "extra": "mean: 180.72617249999232 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.054485759267253,
            "unit": "iter/sec",
            "range": "stddev: 0.0013186935682621833",
            "extra": "mean: 197.84406320000585 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16672016.505105196,
            "unit": "iter/sec",
            "range": "stddev: 8.368517845441465e-9",
            "extra": "mean: 59.98074676172415 nsec\nrounds: 163613"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 389.46849787657175,
            "unit": "iter/sec",
            "range": "stddev: 0.000038330173320141666",
            "extra": "mean: 2.5676017584274926 msec\nrounds: 356"
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
          "id": "5c21114906f6b336638a6a6906c7b997a26189b7",
          "message": "Re-record dev-to-demos GIFs without leaking real filesystem path or username",
          "timestamp": "2026-08-09T00:25:38-07:00",
          "tree_id": "d1db06fb7b01c822027ffb642663d6579ec6e359",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/5c21114906f6b336638a6a6906c7b997a26189b7"
        },
        "date": 1786260380271,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 867.8776664716584,
            "unit": "iter/sec",
            "range": "stddev: 0.00002585933639505409",
            "extra": "mean: 1.1522361257036176 msec\nrounds: 533"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 876.6729217526643,
            "unit": "iter/sec",
            "range": "stddev: 0.000024098402545902694",
            "extra": "mean: 1.1406762718309782 msec\nrounds: 710"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 913880.1576102731,
            "unit": "iter/sec",
            "range": "stddev: 3.467070890989879e-7",
            "extra": "mean: 1.0942353783180103 usec\nrounds: 156446"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 370552.7031999732,
            "unit": "iter/sec",
            "range": "stddev: 6.507901101976736e-7",
            "extra": "mean: 2.6986714477166776 usec\nrounds: 74600"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3392.8871027557593,
            "unit": "iter/sec",
            "range": "stddev: 0.0002348982289484999",
            "extra": "mean: 294.73423951766136 usec\nrounds: 1741"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31499.083944508995,
            "unit": "iter/sec",
            "range": "stddev: 0.000004147122998199204",
            "extra": "mean: 31.746954983251904 usec\nrounds: 8619"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.480273640402217,
            "unit": "iter/sec",
            "range": "stddev: 0.0015444760036160015",
            "extra": "mean: 182.47264016666995 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.940116432260533,
            "unit": "iter/sec",
            "range": "stddev: 0.0011011353384528376",
            "extra": "mean: 202.42437879999784 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16509701.98523032,
            "unit": "iter/sec",
            "range": "stddev: 8.248161417539257e-9",
            "extra": "mean: 60.57044523847893 nsec\nrounds: 159465"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 386.9281174300993,
            "unit": "iter/sec",
            "range": "stddev: 0.0001660822795088215",
            "extra": "mean: 2.584459373595809 msec\nrounds: 356"
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
          "id": "07ebc3b3d61d60044dd9be2f30ef1b8cf46928e8",
          "message": "Fix GIF file-size bloat from a bad resize disposal setting",
          "timestamp": "2026-08-09T00:28:55-07:00",
          "tree_id": "7c6d2458481b7d6d5831d81591d1d9d933a1fa47",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/07ebc3b3d61d60044dd9be2f30ef1b8cf46928e8"
        },
        "date": 1786260588116,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 822.5586985337953,
            "unit": "iter/sec",
            "range": "stddev: 0.0001512961681063949",
            "extra": "mean: 1.2157187101449325 msec\nrounds: 483"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 856.6897712291166,
            "unit": "iter/sec",
            "range": "stddev: 0.00002709313135077846",
            "extra": "mean: 1.1672836930984625 msec\nrounds: 681"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 895823.7330897205,
            "unit": "iter/sec",
            "range": "stddev: 4.2076621738561537e-7",
            "extra": "mean: 1.1162910325572342 usec\nrounds: 118540"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 365616.5551531297,
            "unit": "iter/sec",
            "range": "stddev: 6.429027955550606e-7",
            "extra": "mean: 2.735105907830607 usec\nrounds: 69315"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3412.371944302097,
            "unit": "iter/sec",
            "range": "stddev: 0.0002029497570975234",
            "extra": "mean: 293.05128993038926 usec\nrounds: 1728"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31624.36398833495,
            "unit": "iter/sec",
            "range": "stddev: 0.000005311065633308014",
            "extra": "mean: 31.621189294711595 usec\nrounds: 7940"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.280790003227249,
            "unit": "iter/sec",
            "range": "stddev: 0.0008085408353488301",
            "extra": "mean: 189.36560616666634 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.728497861466865,
            "unit": "iter/sec",
            "range": "stddev: 0.002346456197660468",
            "extra": "mean: 211.4836527999998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 10795715.850868247,
            "unit": "iter/sec",
            "range": "stddev: 3.3581529038834616e-8",
            "extra": "mean: 92.629336841945 nsec\nrounds: 120846"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 384.8242563707772,
            "unit": "iter/sec",
            "range": "stddev: 0.000050630597209311106",
            "extra": "mean: 2.5985887933127128 msec\nrounds: 329"
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
          "id": "792e887a1bf2c40591ca842b4dca88757bfd6e61",
          "message": "Add remaining demo GIFs to README",
          "timestamp": "2026-08-09T02:09:36-07:00",
          "tree_id": "90ae99aa5edfedbf06f18a3eeee57ef9b189cc3d",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/792e887a1bf2c40591ca842b4dca88757bfd6e61"
        },
        "date": 1786266615460,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 951.8339856417941,
            "unit": "iter/sec",
            "range": "stddev: 0.000021521231093201845",
            "extra": "mean: 1.0506033773586356 msec\nrounds: 530"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 965.8925897513641,
            "unit": "iter/sec",
            "range": "stddev: 0.00002127633550339029",
            "extra": "mean: 1.0353118044496186 msec\nrounds: 854"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 949159.9808867519,
            "unit": "iter/sec",
            "range": "stddev: 3.269128151465056e-7",
            "extra": "mean: 1.0535631717908618 usec\nrounds: 109123"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 378817.7955862583,
            "unit": "iter/sec",
            "range": "stddev: 5.213452639091952e-7",
            "extra": "mean: 2.6397915083487575 usec\nrounds: 66300"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 4250.204026683198,
            "unit": "iter/sec",
            "range": "stddev: 0.00011039889772668579",
            "extra": "mean: 235.28282259437475 usec\nrounds: 2390"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 33234.34190947536,
            "unit": "iter/sec",
            "range": "stddev: 0.000002557761792060678",
            "extra": "mean: 30.089357650704446 usec\nrounds: 8581"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 6.004587216412981,
            "unit": "iter/sec",
            "range": "stddev: 0.00044263376853083896",
            "extra": "mean: 166.53934133333811 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.913651647804427,
            "unit": "iter/sec",
            "range": "stddev: 0.05203290138665676",
            "extra": "mean: 203.51463060000015 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 15764311.542524477,
            "unit": "iter/sec",
            "range": "stddev: 8.723931528477205e-9",
            "extra": "mean: 63.434422575479076 nsec\nrounds: 152929"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 488.44640118093236,
            "unit": "iter/sec",
            "range": "stddev: 0.00004194546186594405",
            "extra": "mean: 2.0473075399517087 msec\nrounds: 413"
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
          "id": "013198d6659f1745ab9ac9fc3809e276438dae2e",
          "message": "Update demo GIFs with richer, PH-quality recordings",
          "timestamp": "2026-08-09T11:32:48-07:00",
          "tree_id": "31a556397f62a51c7559a90b6d89c364830d66e8",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/013198d6659f1745ab9ac9fc3809e276438dae2e"
        },
        "date": 1786300409613,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 874.3678204075275,
            "unit": "iter/sec",
            "range": "stddev: 0.000025233516352227855",
            "extra": "mean: 1.1436834438096288 msec\nrounds: 525"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 869.1518377841356,
            "unit": "iter/sec",
            "range": "stddev: 0.000058104296191530156",
            "extra": "mean: 1.150546954545314 msec\nrounds: 770"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 915418.8351563532,
            "unit": "iter/sec",
            "range": "stddev: 3.676493906976314e-7",
            "extra": "mean: 1.0923961378063631 usec\nrounds: 133085"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 368664.75253004243,
            "unit": "iter/sec",
            "range": "stddev: 8.347812734185125e-7",
            "extra": "mean: 2.7124914794193953 usec\nrounds: 56745"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3346.813250955004,
            "unit": "iter/sec",
            "range": "stddev: 0.00018080129182871554",
            "extra": "mean: 298.7916937745638 usec\nrounds: 1783"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31101.507426204975,
            "unit": "iter/sec",
            "range": "stddev: 0.000005632137481173803",
            "extra": "mean: 32.15278238113427 usec\nrounds: 9172"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.571964511233108,
            "unit": "iter/sec",
            "range": "stddev: 0.00024698151890532443",
            "extra": "mean: 179.46991549999916 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.036100926895181,
            "unit": "iter/sec",
            "range": "stddev: 0.0009195840841389687",
            "extra": "mean: 198.5663144 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16665739.686918065,
            "unit": "iter/sec",
            "range": "stddev: 8.234308838672188e-9",
            "extra": "mean: 60.00333731271225 nsec\nrounds: 164177"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 390.31625502486156,
            "unit": "iter/sec",
            "range": "stddev: 0.00006089337906927413",
            "extra": "mean: 2.5620249915963766 msec\nrounds: 357"
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
          "id": "7d299d0f061add82e5cb1a4b835771c687a1bd25",
          "message": "Add traffic-light window dots to demo GIFs",
          "timestamp": "2026-08-09T12:06:56-07:00",
          "tree_id": "d82f3f8c2de0f985edc34f6bb1d5175ca2b4e004",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/7d299d0f061add82e5cb1a4b835771c687a1bd25"
        },
        "date": 1786302467478,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 869.3007074418,
            "unit": "iter/sec",
            "range": "stddev: 0.00007486914521890787",
            "extra": "mean: 1.1503499208493977 msec\nrounds: 518"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 885.4853906599916,
            "unit": "iter/sec",
            "range": "stddev: 0.000024995706136191377",
            "extra": "mean: 1.1293241091811301 msec\nrounds: 806"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 941734.5430698689,
            "unit": "iter/sec",
            "range": "stddev: 3.803512371176577e-7",
            "extra": "mean: 1.0618703618327487 usec\nrounds: 136557"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 371540.34255134023,
            "unit": "iter/sec",
            "range": "stddev: 6.315288610640121e-7",
            "extra": "mean: 2.69149776073595 usec\nrounds: 73015"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3299.1967028269046,
            "unit": "iter/sec",
            "range": "stddev: 0.00027007911069566867",
            "extra": "mean: 303.10408565307836 usec\nrounds: 1868"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31400.8676882299,
            "unit": "iter/sec",
            "range": "stddev: 0.000004118084808792732",
            "extra": "mean: 31.84625373823137 usec\nrounds: 8694"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.621195704737159,
            "unit": "iter/sec",
            "range": "stddev: 0.001124484420100713",
            "extra": "mean: 177.8980936666675 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.071509804626662,
            "unit": "iter/sec",
            "range": "stddev: 0.0026562023757509346",
            "extra": "mean: 197.1799402000002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16647712.27047383,
            "unit": "iter/sec",
            "range": "stddev: 7.688002851248072e-9",
            "extra": "mean: 60.06831351678195 nsec\nrounds: 158454"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 388.7685886488463,
            "unit": "iter/sec",
            "range": "stddev: 0.00008480704441531362",
            "extra": "mean: 2.572224272221864 msec\nrounds: 360"
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
          "id": "d6126617a2e0a1bcb684fa9da85e041fba795e86",
          "message": "Move demo GIF above the fold for engagement",
          "timestamp": "2026-08-09T12:34:53-07:00",
          "tree_id": "59dbd3e665bf5d6c35c4bfdc049ef0d8ed297284",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/d6126617a2e0a1bcb684fa9da85e041fba795e86"
        },
        "date": 1786304145621,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 1124.100063492104,
            "unit": "iter/sec",
            "range": "stddev: 0.00003544204990109111",
            "extra": "mean: 889.6005190974035 usec\nrounds: 576"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 1130.6207304569568,
            "unit": "iter/sec",
            "range": "stddev: 0.00006414354399279382",
            "extra": "mean: 884.4698961037407 usec\nrounds: 847"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 1040389.7649569417,
            "unit": "iter/sec",
            "range": "stddev: 2.681847572526166e-7",
            "extra": "mean: 961.1782369287213 nsec\nrounds: 136240"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 428218.1340684932,
            "unit": "iter/sec",
            "range": "stddev: 4.583557447784155e-7",
            "extra": "mean: 2.3352584125735567 usec\nrounds: 65527"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3476.1721882001502,
            "unit": "iter/sec",
            "range": "stddev: 0.0008422242958837552",
            "extra": "mean: 287.67274630252643 usec\nrounds: 1758"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 42166.87031372249,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021560032199075023",
            "extra": "mean: 23.715300484953637 usec\nrounds: 7012"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.812842441269276,
            "unit": "iter/sec",
            "range": "stddev: 0.0007615125668928434",
            "extra": "mean: 172.03287550000113 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.818229367623023,
            "unit": "iter/sec",
            "range": "stddev: 0.007234174538634143",
            "extra": "mean: 207.54512159999763 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 22026999.187526304,
            "unit": "iter/sec",
            "range": "stddev: 5.391131234214678e-9",
            "extra": "mean: 45.398830384771216 nsec\nrounds: 196426"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 535.1823668750085,
            "unit": "iter/sec",
            "range": "stddev: 0.00003798464741035407",
            "extra": "mean: 1.8685219504504889 msec\nrounds: 444"
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
          "id": "ae631854f29de25b4439b95c5f4973753a361772",
          "message": "Add CodeQL security scanning workflow",
          "timestamp": "2026-08-09T12:45:02-07:00",
          "tree_id": "dec5c85566c3fe60ab35f8975c2b85c2d732570d",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/ae631854f29de25b4439b95c5f4973753a361772"
        },
        "date": 1786305045680,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 871.8267230426047,
            "unit": "iter/sec",
            "range": "stddev: 0.000024189138899308533",
            "extra": "mean: 1.1470169169741447 msec\nrounds: 542"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 877.7362423187883,
            "unit": "iter/sec",
            "range": "stddev: 0.000033585647617275295",
            "extra": "mean: 1.139294416461849 msec\nrounds: 814"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 898404.2430739191,
            "unit": "iter/sec",
            "range": "stddev: 3.706005305355851e-7",
            "extra": "mean: 1.1130846806538535 usec\nrounds: 138237"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 373138.9686166119,
            "unit": "iter/sec",
            "range": "stddev: 5.783517335900402e-7",
            "extra": "mean: 2.6799666722225073 usec\nrounds: 66011"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3067.6038685937374,
            "unit": "iter/sec",
            "range": "stddev: 0.0007314519797716352",
            "extra": "mean: 325.98733175363475 usec\nrounds: 2110"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31548.67326042048,
            "unit": "iter/sec",
            "range": "stddev: 0.000004534774840992642",
            "extra": "mean: 31.69705400114414 usec\nrounds: 8685"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.5812082875027,
            "unit": "iter/sec",
            "range": "stddev: 0.00037820208992813575",
            "extra": "mean: 179.17267166666662 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.095535761781566,
            "unit": "iter/sec",
            "range": "stddev: 0.0020606542172487024",
            "extra": "mean: 196.25021720000007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16706960.554963127,
            "unit": "iter/sec",
            "range": "stddev: 7.842679578398678e-9",
            "extra": "mean: 59.8552918533665 nsec\nrounds: 164963"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 389.5311587669827,
            "unit": "iter/sec",
            "range": "stddev: 0.00006319763500812829",
            "extra": "mean: 2.567188728022139 msec\nrounds: 364"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fea4e81c23a410221c072d41cb27b3ac039a3ce3",
          "message": "Bump h2 from 4.3.0 to 4.4.1 (#34)\n\nBumps [h2](https://github.com/python-hyper/h2) from 4.3.0 to 4.4.1.\n- [Changelog](https://github.com/python-hyper/h2/blob/master/CHANGELOG.rst)\n- [Commits](https://github.com/python-hyper/h2/compare/v4.3.0...v4.4.1)\n\n---\nupdated-dependencies:\n- dependency-name: h2\n  dependency-version: 4.4.1\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T20:51:51Z",
          "tree_id": "83622e4818fd807cfde8a5785a93be81e4e09306",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/fea4e81c23a410221c072d41cb27b3ac039a3ce3"
        },
        "date": 1786309316005,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 1304.2758021612212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000478887052129277",
            "extra": "mean: 766.7090030674282 usec\nrounds: 652"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 1317.9208068273954,
            "unit": "iter/sec",
            "range": "stddev: 0.00004572672747416117",
            "extra": "mean: 758.7709328356991 usec\nrounds: 1072"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 1164444.6152616707,
            "unit": "iter/sec",
            "range": "stddev: 3.3584441270670456e-7",
            "extra": "mean: 858.7784999763882 nsec\nrounds: 149544"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 477237.8532667763,
            "unit": "iter/sec",
            "range": "stddev: 4.220252837220884e-7",
            "extra": "mean: 2.095391204102578 usec\nrounds: 79378"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3148.474860437042,
            "unit": "iter/sec",
            "range": "stddev: 0.0005443667495698833",
            "extra": "mean: 317.6140970873718 usec\nrounds: 1854"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 46653.89265253034,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027429111200677094",
            "extra": "mean: 21.43443865333633 usec\nrounds: 10188"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 6.334778143589878,
            "unit": "iter/sec",
            "range": "stddev: 0.0037027228120995017",
            "extra": "mean: 157.8587248571434 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.270759179162991,
            "unit": "iter/sec",
            "range": "stddev: 0.07657824351921974",
            "extra": "mean: 234.15040699999992 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 24492983.830619857,
            "unit": "iter/sec",
            "range": "stddev: 5.198114838633115e-9",
            "extra": "mean: 40.828018624249935 nsec\nrounds: 197006"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 608.1454163900377,
            "unit": "iter/sec",
            "range": "stddev: 0.00006229570276467286",
            "extra": "mean: 1.6443435616698687 msec\nrounds: 527"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8e6d57ee1dd9ddf764832702b3a800b79116069e",
          "message": "Bump softprops/action-gh-release from 2.6.2 to 3.0.2 (#26)\n\nBumps [softprops/action-gh-release](https://github.com/softprops/action-gh-release) from 2.6.2 to 3.0.2.\n- [Release notes](https://github.com/softprops/action-gh-release/releases)\n- [Changelog](https://github.com/softprops/action-gh-release/blob/master/CHANGELOG.md)\n- [Commits](https://github.com/softprops/action-gh-release/compare/3bb12739c298aeb8a4eeaf626c5b8d85266b0e65...3d0d9888cb7fd7b750713d6e236d1fcb99157228)\n\n---\nupdated-dependencies:\n- dependency-name: softprops/action-gh-release\n  dependency-version: 3.0.2\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T20:52:09Z",
          "tree_id": "207bc2d248139086750002d06c400d6d39bcc5ec",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/8e6d57ee1dd9ddf764832702b3a800b79116069e"
        },
        "date": 1786309325434,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 1109.2766029093943,
            "unit": "iter/sec",
            "range": "stddev: 0.00002983247715573883",
            "extra": "mean: 901.4884090922089 usec\nrounds: 616"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 1132.7257636207098,
            "unit": "iter/sec",
            "range": "stddev: 0.000025133202076058196",
            "extra": "mean: 882.8262162975286 usec\nrounds: 994"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 938821.9289451988,
            "unit": "iter/sec",
            "range": "stddev: 2.313713639596107e-7",
            "extra": "mean: 1.0651647231158492 usec\nrounds: 150671"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 382164.8135831994,
            "unit": "iter/sec",
            "range": "stddev: 3.6048512869180455e-7",
            "extra": "mean: 2.616672086118924 usec\nrounds: 71348"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 5020.222181674759,
            "unit": "iter/sec",
            "range": "stddev: 0.00009175118751808255",
            "extra": "mean: 199.19437104801554 usec\nrounds: 2404"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 38161.127720452736,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019231039327817014",
            "extra": "mean: 26.20467632207951 usec\nrounds: 8907"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.594761953354159,
            "unit": "iter/sec",
            "range": "stddev: 0.0039830812859901355",
            "extra": "mean: 178.73861450002929 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.240340426516397,
            "unit": "iter/sec",
            "range": "stddev: 0.0003722352333779434",
            "extra": "mean: 190.82729719999634 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 17351031.932444785,
            "unit": "iter/sec",
            "range": "stddev: 5.089720479006122e-9",
            "extra": "mean: 57.63345972121086 nsec\nrounds: 169953"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 502.3144837499199,
            "unit": "iter/sec",
            "range": "stddev: 0.00003077165809580463",
            "extra": "mean: 1.9907847222216586 msec\nrounds: 450"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9f3aa2e01fb167cb82e54066c9a09a156b8aba88",
          "message": "Bump codecov/codecov-action from 4.6.0 to 7.0.0 (#29)\n\nBumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 4.6.0 to 7.0.0.\n- [Release notes](https://github.com/codecov/codecov-action/releases)\n- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/codecov/codecov-action/compare/b9fd7d16f6d7d1b5d2bec1a2887e65ceed900238...fb8b3582c8e4def4969c97caa2f19720cb33a72f)\n\n---\nupdated-dependencies:\n- dependency-name: codecov/codecov-action\n  dependency-version: 7.0.0\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T20:52:31Z",
          "tree_id": "a550989da4f753cc8522fe2505e8df39bd2a8b70",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/9f3aa2e01fb167cb82e54066c9a09a156b8aba88"
        },
        "date": 1786309330269,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 956.268119707867,
            "unit": "iter/sec",
            "range": "stddev: 0.000022414644462409212",
            "extra": "mean: 1.0457318187136604 msec\nrounds: 513"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 959.6646431776063,
            "unit": "iter/sec",
            "range": "stddev: 0.000038758436279622364",
            "extra": "mean: 1.0420306792681626 msec\nrounds: 820"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 960915.8250471782,
            "unit": "iter/sec",
            "range": "stddev: 3.3491836563881534e-7",
            "extra": "mean: 1.0406738799945385 usec\nrounds: 117745"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 376909.0703964208,
            "unit": "iter/sec",
            "range": "stddev: 5.475584736844012e-7",
            "extra": "mean: 2.653159816366935 usec\nrounds: 64255"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 4287.288043860182,
            "unit": "iter/sec",
            "range": "stddev: 0.00010513653288891614",
            "extra": "mean: 233.24768239729036 usec\nrounds: 2119"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 32838.30472347138,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026639593598905534",
            "extra": "mean: 30.452241929689016 usec\nrounds: 8333"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.859442812750771,
            "unit": "iter/sec",
            "range": "stddev: 0.0033532889111639102",
            "extra": "mean: 170.66469149999955 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.482654652126397,
            "unit": "iter/sec",
            "range": "stddev: 0.0005102556846252044",
            "extra": "mean: 182.39339579999978 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16232369.542867886,
            "unit": "iter/sec",
            "range": "stddev: 9.447609658642431e-9",
            "extra": "mean: 61.605300283431255 nsec\nrounds: 159516"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 499.15751303601127,
            "unit": "iter/sec",
            "range": "stddev: 0.000048728922609279595",
            "extra": "mean: 2.0033756357141237 msec\nrounds: 420"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d8803cac4b45c056e570af76a67552548e279aa4",
          "message": "Bump aiohttp from 3.14.1 to 3.14.3 (#32)\n\nBumps [aiohttp](https://github.com/aio-libs/aiohttp) from 3.14.1 to 3.14.3.\n- [Changelog](https://github.com/aio-libs/aiohttp/blob/master/CHANGES.rst)\n- [Commits](https://github.com/aio-libs/aiohttp/compare/v3.14.1...v3.14.3)\n\n---\nupdated-dependencies:\n- dependency-name: aiohttp\n  dependency-version: 3.14.3\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T20:52:40Z",
          "tree_id": "70bb1ccedee7b13bf7d21e690e23169ff3a77fa5",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/d8803cac4b45c056e570af76a67552548e279aa4"
        },
        "date": 1786309498175,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 871.6857756716829,
            "unit": "iter/sec",
            "range": "stddev: 0.000042012062642455625",
            "extra": "mean: 1.1472023840579981 msec\nrounds: 552"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 882.6620953839272,
            "unit": "iter/sec",
            "range": "stddev: 0.000024008454912609123",
            "extra": "mean: 1.132936381011167 msec\nrounds: 811"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 923758.9254719596,
            "unit": "iter/sec",
            "range": "stddev: 3.948303417270104e-7",
            "extra": "mean: 1.0825335186765184 usec\nrounds: 134880"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 373662.71368601796,
            "unit": "iter/sec",
            "range": "stddev: 7.585446518543435e-7",
            "extra": "mean: 2.6762102917238937 usec\nrounds: 70892"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3495.569710001279,
            "unit": "iter/sec",
            "range": "stddev: 0.00016960591308819696",
            "extra": "mean: 286.076400404452 usec\nrounds: 1978"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31785.677500929738,
            "unit": "iter/sec",
            "range": "stddev: 0.000004018366298318791",
            "extra": "mean: 31.46071056596952 usec\nrounds: 8641"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.5302790974616505,
            "unit": "iter/sec",
            "range": "stddev: 0.0024104945682686028",
            "extra": "mean: 180.8227003333324 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.082060291798778,
            "unit": "iter/sec",
            "range": "stddev: 0.0012777270487250487",
            "extra": "mean: 196.77058960000124 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16692569.438881597,
            "unit": "iter/sec",
            "range": "stddev: 8.095917754077555e-9",
            "extra": "mean: 59.90689472111611 nsec\nrounds: 160488"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 388.84378067659384,
            "unit": "iter/sec",
            "range": "stddev: 0.00003740806326428286",
            "extra": "mean: 2.571726872575885 msec\nrounds: 361"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "856e0d8424e0d0833707cd9f961345dd070bfab1",
          "message": "Bump actions/checkout from 4.3.1 to 7.0.1 (#28)\n\nBumps [actions/checkout](https://github.com/actions/checkout) from 4.3.1 to 7.0.1.\n- [Release notes](https://github.com/actions/checkout/releases)\n- [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/actions/checkout/compare/34e114876b0b11c390a56381ad16ebd13914f8d5...3d3c42e5aac5ba805825da76410c181273ba90b1)\n\n---\nupdated-dependencies:\n- dependency-name: actions/checkout\n  dependency-version: 7.0.1\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T20:53:03Z",
          "tree_id": "3736d06d3de22fb266070523da25579679f54756",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/856e0d8424e0d0833707cd9f961345dd070bfab1"
        },
        "date": 1786309635883,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 1728.2548516953416,
            "unit": "iter/sec",
            "range": "stddev: 0.000014466007094900793",
            "extra": "mean: 578.6183669723504 usec\nrounds: 981"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 1750.0067391486825,
            "unit": "iter/sec",
            "range": "stddev: 0.00001644144425140898",
            "extra": "mean: 571.4263708987003 usec\nrounds: 1402"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 2053375.4334051707,
            "unit": "iter/sec",
            "range": "stddev: 5.591851719433335e-8",
            "extra": "mean: 487.0030018532323 nsec\nrounds: 190549"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 674685.0941335799,
            "unit": "iter/sec",
            "range": "stddev: 2.524921748372001e-7",
            "extra": "mean: 1.4821729554944214 usec\nrounds: 102934"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 4077.3842390980744,
            "unit": "iter/sec",
            "range": "stddev: 0.0005460561186278002",
            "extra": "mean: 245.2552767558649 usec\nrounds: 2392"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 64723.606393093796,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011692264906992508",
            "extra": "mean: 15.450313351307676 usec\nrounds: 12995"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 9.467947192638071,
            "unit": "iter/sec",
            "range": "stddev: 0.0007065664764766368",
            "extra": "mean: 105.61951600000086 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 2.373941678994942,
            "unit": "iter/sec",
            "range": "stddev: 0.41049835197202506",
            "extra": "mean: 421.24033999999995 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 31926669.479357455,
            "unit": "iter/sec",
            "range": "stddev: 4.357164131641713e-9",
            "extra": "mean: 31.321776317650706 nsec\nrounds: 198492"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 942.281866685144,
            "unit": "iter/sec",
            "range": "stddev: 0.000014496113926014352",
            "extra": "mean: 1.061253575342485 msec\nrounds: 803"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7a0f57eefef4bf60ad64a2262c26172f2b5ac208",
          "message": "Bump setuptools from 82.0.1 to 83.0.0 (#22)\n\nBumps [setuptools](https://github.com/pypa/setuptools) from 82.0.1 to 83.0.0.\n- [Release notes](https://github.com/pypa/setuptools/releases)\n- [Changelog](https://github.com/pypa/setuptools/blob/main/NEWS.rst)\n- [Commits](https://github.com/pypa/setuptools/compare/v82.0.1...v83.0.0)\n\n---\nupdated-dependencies:\n- dependency-name: setuptools\n  dependency-version: 83.0.0\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T20:53:33Z",
          "tree_id": "7a4313cf99c8ca6bd05a00d950481b9d81181321",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/7a0f57eefef4bf60ad64a2262c26172f2b5ac208"
        },
        "date": 1786309698390,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 870.9184912937084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000235360882404334",
            "extra": "mean: 1.1482130761910303 msec\nrounds: 525"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 875.0473871067484,
            "unit": "iter/sec",
            "range": "stddev: 0.000025063203750266584",
            "extra": "mean: 1.142795252844985 msec\nrounds: 791"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 914240.6904923581,
            "unit": "iter/sec",
            "range": "stddev: 3.659351261541493e-7",
            "extra": "mean: 1.0938038641240708 usec\nrounds: 134157"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 360519.48581286287,
            "unit": "iter/sec",
            "range": "stddev: 9.486025804748127e-7",
            "extra": "mean: 2.773775175411951 usec\nrounds: 74378"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3115.7839494709424,
            "unit": "iter/sec",
            "range": "stddev: 0.0004119685861390254",
            "extra": "mean: 320.94651497572517 usec\nrounds: 2070"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31535.55010808747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000044455166050570824",
            "extra": "mean: 31.710244361443507 usec\nrounds: 8291"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.522785977118213,
            "unit": "iter/sec",
            "range": "stddev: 0.0024516306918380354",
            "extra": "mean: 181.0680341666616 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.851584066261463,
            "unit": "iter/sec",
            "range": "stddev: 0.0028385745629314106",
            "extra": "mean: 206.11824640000123 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16818500.159477536,
            "unit": "iter/sec",
            "range": "stddev: 8.276800725475788e-9",
            "extra": "mean: 59.45833400824874 nsec\nrounds: 160463"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 386.72723746042095,
            "unit": "iter/sec",
            "range": "stddev: 0.00003984600693514456",
            "extra": "mean: 2.5858018343027713 msec\nrounds: 344"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a8bddd3033ec698822175c5ad5fe0f6fe3e80c95",
          "message": "Bump ossf/scorecard-action from 2.4.0 to 2.4.4 (#25)\n\nBumps [ossf/scorecard-action](https://github.com/ossf/scorecard-action) from 2.4.0 to 2.4.4.\n- [Release notes](https://github.com/ossf/scorecard-action/releases)\n- [Changelog](https://github.com/ossf/scorecard-action/blob/main/RELEASE.md)\n- [Commits](https://github.com/ossf/scorecard-action/compare/62b2cac7ed8198b15735ed49ab1e5cf35480ba46...2d1146689b8cda280b9bc96326124645441f03bc)\n\n---\nupdated-dependencies:\n- dependency-name: ossf/scorecard-action\n  dependency-version: 2.4.4\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T20:54:08Z",
          "tree_id": "6fce4ab524a0a696d6632330461f55b7bd3504a8",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/a8bddd3033ec698822175c5ad5fe0f6fe3e80c95"
        },
        "date": 1786309801671,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 866.324046112871,
            "unit": "iter/sec",
            "range": "stddev: 0.000024525866917241847",
            "extra": "mean: 1.1543024858734128 msec\nrounds: 531"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 871.7748210427287,
            "unit": "iter/sec",
            "range": "stddev: 0.00003561591961176869",
            "extra": "mean: 1.1470852057919054 msec\nrounds: 656"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 927885.9509714537,
            "unit": "iter/sec",
            "range": "stddev: 4.0658540708511395e-7",
            "extra": "mean: 1.077718655997589 usec\nrounds: 137099"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 377215.5194807081,
            "unit": "iter/sec",
            "range": "stddev: 6.740133166618224e-7",
            "extra": "mean: 2.6510043949852466 usec\nrounds: 70537"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3480.194494899116,
            "unit": "iter/sec",
            "range": "stddev: 0.00015604593440145815",
            "extra": "mean: 287.3402625817866 usec\nrounds: 2007"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31098.209633260765,
            "unit": "iter/sec",
            "range": "stddev: 0.000004357329496566199",
            "extra": "mean: 32.156192005679344 usec\nrounds: 8406"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.368989494111024,
            "unit": "iter/sec",
            "range": "stddev: 0.00038929405354622463",
            "extra": "mean: 186.25478800002307 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 4.869410474809129,
            "unit": "iter/sec",
            "range": "stddev: 0.000757435436752951",
            "extra": "mean: 205.3636688000097 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16794158.524792414,
            "unit": "iter/sec",
            "range": "stddev: 8.361443310251177e-9",
            "extra": "mean: 59.544513559506285 nsec\nrounds: 95058"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 391.16160877202947,
            "unit": "iter/sec",
            "range": "stddev: 0.00006515423234487329",
            "extra": "mean: 2.55648810510646 msec\nrounds: 333"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9b7bda869cb8e6f5b66594c1e5c11d3230a2c1bb",
          "message": "Bump cryptography from 49.0.0 to 50.0.0 (#33)\n\nBumps [cryptography](https://github.com/pyca/cryptography) from 49.0.0 to 50.0.0.\n- [Changelog](https://github.com/pyca/cryptography/blob/main/CHANGELOG.rst)\n- [Commits](https://github.com/pyca/cryptography/compare/49.0.0...50.0.0)\n\n---\nupdated-dependencies:\n- dependency-name: cryptography\n  dependency-version: 50.0.0\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T13:59:18-07:00",
          "tree_id": "0846875d414f296d1d25a23184747fe6420752b6",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/9b7bda869cb8e6f5b66594c1e5c11d3230a2c1bb"
        },
        "date": 1786309819770,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 871.721041842283,
            "unit": "iter/sec",
            "range": "stddev: 0.00002814975930709863",
            "extra": "mean: 1.1471559730697956 msec\nrounds: 557"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 876.1310494935652,
            "unit": "iter/sec",
            "range": "stddev: 0.00007125772658927885",
            "extra": "mean: 1.1413817608427819 msec\nrounds: 807"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 898224.4623476892,
            "unit": "iter/sec",
            "range": "stddev: 4.728522756685613e-7",
            "extra": "mean: 1.113307465915925 usec\nrounds: 129803"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 369773.4271369238,
            "unit": "iter/sec",
            "range": "stddev: 5.905209493424563e-7",
            "extra": "mean: 2.704358741358959 usec\nrounds: 75160"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3483.5828031464944,
            "unit": "iter/sec",
            "range": "stddev: 0.0001445438200557702",
            "extra": "mean: 287.06078095711257 usec\nrounds: 2027"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31809.537765376692,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061661404710970565",
            "extra": "mean: 31.437111956039075 usec\nrounds: 9468"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.547288099263933,
            "unit": "iter/sec",
            "range": "stddev: 0.00029017133912744025",
            "extra": "mean: 180.2682648000001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.0760402452876745,
            "unit": "iter/sec",
            "range": "stddev: 0.0003973241607693773",
            "extra": "mean: 197.00395419999808 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16520445.628352292,
            "unit": "iter/sec",
            "range": "stddev: 9.98373612908713e-9",
            "extra": "mean: 60.53105482117298 nsec\nrounds: 161525"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 386.7411863449659,
            "unit": "iter/sec",
            "range": "stddev: 0.00007935736207332826",
            "extra": "mean: 2.5857085702478524 msec\nrounds: 363"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "764c1fdc68a87307c59ebf0b6fd967a29882d223",
          "message": "Bump github/codeql-action/upload-sarif from 3.37.1 to 4.37.3 (#27)\n\nBumps [github/codeql-action/upload-sarif](https://github.com/github/codeql-action) from 3.37.1 to 4.37.3.\n- [Release notes](https://github.com/github/codeql-action/releases)\n- [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/github/codeql-action/compare/b7351df727350dca84cb9d725d57dcf5bc82ba26...e4fba868fa4b1b91e1fdab776edc8cfbe6e9fb81)\n\n---\nupdated-dependencies:\n- dependency-name: github/codeql-action/upload-sarif\n  dependency-version: 4.37.3\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: Rudrendu Paul <38769913+RudrenduPaul@users.noreply.github.com>",
          "timestamp": "2026-08-09T14:02:44-07:00",
          "tree_id": "6e16fc7681698649674eb43bf09f2534fe04d758",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/764c1fdc68a87307c59ebf0b6fd967a29882d223"
        },
        "date": 1786309828315,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 885.1729755004743,
            "unit": "iter/sec",
            "range": "stddev: 0.000020761593699688864",
            "extra": "mean: 1.1297226956512119 msec\nrounds: 529"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 890.9539107770564,
            "unit": "iter/sec",
            "range": "stddev: 0.000022419145559514112",
            "extra": "mean: 1.1223925142523226 msec\nrounds: 842"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 923983.0850920748,
            "unit": "iter/sec",
            "range": "stddev: 3.831468804121732e-7",
            "extra": "mean: 1.0822708944940806 usec\nrounds: 136352"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 374622.32741866214,
            "unit": "iter/sec",
            "range": "stddev: 5.988907978888992e-7",
            "extra": "mean: 2.6693550458951747 usec\nrounds: 80691"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3370.758602539462,
            "unit": "iter/sec",
            "range": "stddev: 0.0001840751245208699",
            "extra": "mean: 296.66912345684443 usec\nrounds: 1944"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 31799.19185007686,
            "unit": "iter/sec",
            "range": "stddev: 0.000004077306996826012",
            "extra": "mean: 31.447340068096192 usec\nrounds: 9404"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.600113512060978,
            "unit": "iter/sec",
            "range": "stddev: 0.0005154233375534282",
            "extra": "mean: 178.56780899999572 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.121203901463017,
            "unit": "iter/sec",
            "range": "stddev: 0.0009210107235936379",
            "extra": "mean: 195.26658559998396 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16689236.878628822,
            "unit": "iter/sec",
            "range": "stddev: 8.202482582310577e-9",
            "extra": "mean: 59.91885712165406 nsec\nrounds: 156202"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 396.57633860112395,
            "unit": "iter/sec",
            "range": "stddev: 0.00006302128066503598",
            "extra": "mean: 2.5215826126374092 msec\nrounds: 364"
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
          "id": "932b77161a98ef5da33049e8ea88ac47f8fbf306",
          "message": "Document known chromadb advisory with no upstream fix yet",
          "timestamp": "2026-08-09T14:45:15-07:00",
          "tree_id": "a2af28a7f20d3a7ab4efe6975fbc090ceab744e2",
          "url": "https://github.com/RudrenduPaul/agent-observability/commit/932b77161a98ef5da33049e8ea88ac47f8fbf306"
        },
        "date": 1786311953452,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_fidelity.py::test_fidelity_exchange_count",
            "value": 867.062486664247,
            "unit": "iter/sec",
            "range": "stddev: 0.00010154352046905724",
            "extra": "mean: 1.1533194151291086 msec\nrounds: 542"
          },
          {
            "name": "benchmarks/test_fidelity.py::test_replay_speed",
            "value": 893.9871914984649,
            "unit": "iter/sec",
            "range": "stddev: 0.00002108364694721112",
            "extra": "mean: 1.1185842588234858 msec\nrounds: 850"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_serialization_speed",
            "value": 908410.8154409275,
            "unit": "iter/sec",
            "range": "stddev: 3.303752154582955e-7",
            "extra": "mean: 1.1008235294013058 usec\nrounds: 135796"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_span_from_dict_speed",
            "value": 367920.2992621831,
            "unit": "iter/sec",
            "range": "stddev: 6.28075428542493e-7",
            "extra": "mean: 2.7179799592612084 usec\nrounds: 71654"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_write_latency",
            "value": 3487.669885502356,
            "unit": "iter/sec",
            "range": "stddev: 0.0002028809166562932",
            "extra": "mean: 286.72438413877074 usec\nrounds: 1614"
          },
          {
            "name": "benchmarks/test_ingestion.py::test_fixture_read_cursor_speed",
            "value": 30960.23309015605,
            "unit": "iter/sec",
            "range": "stddev: 0.000006181979134061517",
            "extra": "mean: 32.299498427159925 usec\nrounds: 9219"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_baseline",
            "value": 5.607809341077231,
            "unit": "iter/sec",
            "range": "stddev: 0.00037958983190397144",
            "extra": "mean: 178.32275299999858 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_instrumented",
            "value": 5.134915984137602,
            "unit": "iter/sec",
            "range": "stddev: 0.0008778500880071746",
            "extra": "mean: 194.7451531999988 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_overhead.py::test_overhead_pct_within_budget",
            "value": 16630340.430014273,
            "unit": "iter/sec",
            "range": "stddev: 1.0645656749152097e-8",
            "extra": "mean: 60.13106010717675 nsec\nrounds: 159465"
          },
          {
            "name": "benchmarks/test_replay_vs_live.py::test_replay_10step_agent_run",
            "value": 398.6764353733378,
            "unit": "iter/sec",
            "range": "stddev: 0.000036656834288470376",
            "extra": "mean: 2.5082997420290387 msec\nrounds: 345"
          }
        ]
      }
    ]
  }
}