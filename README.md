# agent-trace

Replay any failed LangGraph or OpenAI Agents SDK run offline, in under 1 second, with zero LLM API calls.

<!-- DEMO GIF: record a 12-step LangGraph run failing at step 7. Show replay executing against the local SQLite cache — no network activity. 6-8 seconds, terminal only. -->

[![PyPI](https://img.shields.io/pypi/v/agent-trace)](https://pypi.org/project/agent-trace/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/RudrenduPaul/agent-trace/actions/workflows/ci.yml/badge.svg)](https://github.com/RudrenduPaul/agent-trace/actions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/RudrenduPaul/agent-trace/badge)](https://securityscorecards.dev/viewer/?uri=github.com/RudrenduPaul/agent-trace)

---

## The problem

A LangGraph run fails after step 8. Your trace in LangSmith or Langfuse shows *what* broke. But to reproduce it you have to re-run the entire agent — 8 more LLM calls, 30 more seconds, another $0.15 in API cost. If the failure was caused by a specific tool response or a transient model output, you can't reproduce it at all. You're debugging against a moving target.

**agent-trace solves this at the HTTP transport layer.** It records every request and response verbatim to a local SQLite file. Replay serves those exact bytes back in sequence, in under 1ms per exchange — same code path, same span tree, same failure. No API calls.

---

## Install

```bash
pip install agent-trace
# or
uv add agent-trace
```

LangGraph support:

```bash
pip install agent-trace[langgraph]
```

OpenAI Agents SDK support:

```bash
pip install agent-trace[openai-agents]
```

---

## Quick start

```python
from agent_trace import tracer
import httpx

@tracer.instrument(record=True)
def fetch_data(query: str) -> dict:
    with tracer.span("http-call") as span:
        resp = httpx.get("https://httpbin.org/get", params={"q": query})
        span.set_attribute("http.status_code", resp.status_code)
        return resp.json()

result = fetch_data("hello")
# Trace and fixture saved to ~/.agent-trace/runs/run_<id>/
```

Replay offline — no API calls, no tokens:

```python
from agent_trace import replay

with replay("run_<id>") as ctx:
    result = fetch_data("hello")  # same call — served from fixture, zero network
    print(result)  # identical to the original run
```

> To store the input for later retrieval in replay, call `ctx.fixture.set_metadata('input', query)` inside the recording context.

---

> **Sync clients only (v0.1):** agent-trace currently intercepts `httpx.Client` and `requests.Session` (synchronous). `httpx.AsyncClient` — used by default in the OpenAI Python SDK v1.x and Anthropic SDK — is not yet intercepted. Async support is on the roadmap for v0.3. For now, use the synchronous `openai.OpenAI()` client (not `openai.AsyncOpenAI()`) when recording.

---

## Performance benchmarks

Measured on Apple M-series, Python 3.14, SQLite WAL mode, NVMe SSD.
Full scripts in [`benchmarks/`](benchmarks/) — run with `uv run pytest benchmarks/ -v --benchmark-only`.

| Metric | Measured value | Notes |
|--------|---------------|-------|
| Recording overhead per LLM exchange | **0.09 ms P50 · 0.12 ms P99** | SQLite WAL write. Verified by `benchmarks/test_replay_vs_live.py` |
| Overhead as % of GPT-4o p50 (800 ms) | **0.011%** | Unmeasurable in any production workload |
| Replay — 10-step agent run (20 exchanges) | **0.93 ms mean** | Served from SQLite, zero network I/O |
| Replay speedup vs live GPT-4o run | **~15,000x** | 0.93 ms vs 8,500 ms simulated live |
| Fixture write throughput | **7,700 writes/sec** | SQLite WAL, single writer |
| Fixture read throughput (replay cursor) | **46,000 reads/sec** | Per-URL cursor, concurrent-reader safe |
| Span serialization | **0.47 µs** | `Span.to_dict()`, pure CPU |
| Replay fidelity | **100%** | Response bytes byte-for-byte identical to recorded |

> **What 15,000x means in practice:** A 10-step GPT-4o agent run that takes 8–30 seconds live takes under 1 ms to replay from a fixture. In CI, every test run costs $0 in API fees and completes in milliseconds regardless of how many LLM calls the agent makes.

---

## How agent-trace compares

Most observability tools for LLM agents are **observe-only** — they show you a trace of what happened, but reproducing a failure still requires re-running the full agent against live APIs. The table below is based on published benchmarks, official documentation, and GitHub issue threads.

### Capability matrix

| | **agent-trace** | LangSmith | Langfuse | Helicone | OpenLLMetry |
|---|:---:|:---:|:---:|:---:|:---:|
| **Offline replay from local fixture** | ✅ | Partial ¹ | ❌ | ❌ | ❌ |
| **Works with any HTTP client** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **CI replay without API keys** | ✅ | Partial ¹ | ❌ | ❌ | ❌ |
| **Deterministic span timing in replay** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Captures raw HTTP request/response bytes** | ✅ | ❌ | ❌ | ✅ | ❌ |
| Span-level tracing | ✅ | ✅ | ✅ | ✅ | ✅ |
| OTLP export (Jaeger, Grafana Tempo) | ✅ | ❌ | ✅ | ❌ | ✅ |
| Open-source core | ✅ | ❌ | ✅ | ❌ | ✅ |
| Local-only, no server required | ✅ | ❌ | Self-host | ❌ | Self-host |
| Under active development | ✅ | ✅ | ✅ | ❌ ² | ✅ |

¹ LangSmith has `LANGSMITH_TEST_CACHE` / VCR cassettes (`langsmith[vcr]`) for Python + LangChain only. It captures HTTP to `api.openai.com` but not arbitrary HTTP clients, does not record full wire-level bytes, and requires a LangSmith account. See [LangSmith pytest docs](https://docs.langchain.com/langsmith/pytest).

² Helicone was acquired by Mintlify in March 2026 and is no longer under active development.

### Overhead per LLM call (published benchmarks)

| Tool | Per-call overhead | Mechanism | Source |
|------|------------------|-----------|--------|
| **agent-trace** | **0.09 ms P50 · 0.12 ms P99** | In-process SQLite WAL write | `benchmarks/test_replay_vs_live.py` |
| Langfuse SDK | 0.10–0.15 ms (queue insert) | Async in-memory queue; network I/O in background | [Langfuse SDK Performance Test](https://langfuse.com/guides/cookbook/langfuse_sdk_performance_test) |
| LangSmith | < 4 ms (async batch mode) | Background thread + PriorityQueue to cloud | [LangSmith production guide](https://docs.smith.langchain.com/) |
| OpenLLMetry | ~1–5 ms | OTel SDK span creation + async OTLP export | Traceloop documentation |
| Helicone | 10–30 ms (cloud) · 8 ms P50 (self-hosted) | Proxy hop — every LLM call routes through Helicone servers | [Helicone latency docs](https://docs.helicone.ai/references/latency-affect) |

**Note on Langfuse:** The 0.10–0.15 ms figure is the in-process queue insert only. The LangChain callback wrapper adds ~88 ms due to synchronous wrapping overhead. The old v2 synchronous SDK added 155–1,205 ms per call; this was eliminated in later versions. Source: [Langfuse SDK benchmark page](https://langfuse.com/guides/cookbook/langfuse_sdk_performance_test).

### Replay cost

| Scenario | agent-trace | LangSmith VCR | Langfuse Playground | Phoenix Span Replay |
|----------|-----------|-----------------|--------------------|---------------------|
| Cost per CI replay iteration | **$0** | $0 after recording | 1 live LLM call | 1 live LLM call |
| Reproduce intermittent failure | **Always** | Yes (if captured) | No | No |
| Requires cloud account | **No** | Yes | Yes or self-host | Yes or self-host |
| Works for non-LangChain agents | **Yes** | No ¹ | N/A | N/A |
| Captures raw response bytes | **Yes** | No (structured only) | No (structured only) | No (structured only) |

### What competitors do well

**Choose LangSmith** if your team is on LangChain and needs dataset management, prompt versioning, and human feedback loops. Its VCR cassette system covers the CI-replay use case for LangChain-based agents with a LangSmith account.

**Choose Langfuse** if you want a fully open-source, self-hostable observability stack with strong Postgres-backed storage. Best when you need dashboards and evals, not offline replay.

**Choose OpenLLMetry** if your team already runs on OpenTelemetry and wants standard `gen_ai.*` spans from LLM calls without adding a new observability system.

**agent-trace is not a replacement for dashboards and eval pipelines.** It solves the specific upstream problem: reproducing a specific failed run without any LLM API cost, for any agent built on any Python HTTP library.

---

## How it works

- **Transport interception, not API wrapping.** agent-trace patches `httpx.Client.__init__` and `requests.Session.get_adapter` at trace start. Every AI SDK — OpenAI, Anthropic, LangChain — creates its own HTTP client internally. Patching at the transport layer captures those calls with no SDK-specific glue.
- **SQLite fixture, not JSON files.** Each run writes to `~/.agent-trace/runs/<run_id>/fixture.db`. WAL mode lets multiple test workers open the same fixture concurrently. Large response bodies stay on disk until replayed — memory stays flat regardless of response size.
- **Per-(method, URL) cursor.** If your agent calls `POST /v1/chat/completions` three times, the fixture stores all three responses in sequence. Replay serves them in the same order via a per-URL offset cursor. No URL collision, no response mixing.
- **Clock abstraction.** All span timestamps come from `agent_trace.core.clock.get_time()`, not `time.time()`. During replay, `FixtureClock` replaces `WallClock`. Span durations in replayed traces reflect original execution times, not replay times.
- **"Deterministic" means inputs, not outputs.** During replay, each agent node receives the same inputs and tool responses it received during recording. The LLM is bypassed entirely — recorded bytes are returned directly.

---

## Integration matrix

| Integration | Status | Notes |
|-------------|--------|-------|
| LangGraph | Shipped v0.1 | `pip install agent-trace[langgraph]` |
| OpenAI Agents SDK | Shipped v0.1 | `pip install agent-trace[openai-agents]` |
| Anthropic Claude SDK | Planned v0.3 | HTTP interception works today; typed callback pending |
| CrewAI | Planned v0.3 | — |
| AutoGen | Planned v0.4 | — |

---

## Use in CI

Record once. Commit the fixture. Replay in every CI run at zero API cost:

```python
# tests/test_agent.py
import pytest
from pathlib import Path
from agent_trace import replay

FIXTURE_PATH = Path("fixtures/my_agent_run.db")

@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="Run: python scripts/record_fixture.py to generate the fixture"
)
def test_agent_answer():
    with replay(FIXTURE_PATH) as ctx:
        from my_module import my_agent
        result = my_agent("what is 2+2?")
    assert "4" in result
```

Set `AGENT_TRACE_NETWORK_GUARD=1` in CI. Any HTTP call that is not in the fixture raises `NetworkGuardError` immediately — catching regressions before they hit production.

---

## Self-host traces

agent-trace emits OTLP spans. Run Jaeger locally to browse trace trees:

```bash
docker compose up -d
```

The `docker-compose.yml` starts Jaeger with OTLP ingest on port 4317. Open [http://localhost:16686](http://localhost:16686) to browse traces.

```python
from agent_trace.exporters.otlp import OTLPExporter

exporter = OTLPExporter(endpoint="http://localhost:4317")
exporter.export(trace)
```

---

## Community

- [GitHub Issues](https://github.com/RudrenduPaul/agent-trace/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/RudrenduPaul/agent-trace/discussions) — questions and ideas
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup and PR guide
- Discord — link coming soon

---

## Security note

Fixture files at `~/.agent-trace/runs/` contain full HTTP request and response bodies, including API keys, prompt contents, and user data. Add this to your `.gitignore`:

```
~/.agent-trace/
.agent-trace/
*.fixture.db
```

Never commit a `fixture.db` generated against a production API key. Use a separate key for recording, or scrub the fixture before committing (see `docs/concepts.md` for the SQLite schema).
