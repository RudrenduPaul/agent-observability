# Agent Observability

Replay any failed LangGraph or OpenAI Agents SDK run offline, in under 1 second, with zero LLM API calls.

```
# Record a 12-step LangGraph run that fails at step 7
$ uv run --extra langgraph python demos/record_replay_demo.py

=== RECORD MODE ===
Running 12-step pipeline  (will fail at step 7)

  ✓ step_01  completed
  ...
  ✓ step_06  completed
  ✗ step_07  upstream dependency returned null — cannot continue

Recorded: 8 spans captured  →  fixture.db

=== REPLAY MODE ===
(No network calls — all state served from local fixture)

  ✓ step_01  completed
  ...
  ✓ step_06  completed
  ✗ step_07  upstream dependency returned null — cannot continue

Replay complete — same failure reproduced offline.
```

[![PyPI](https://img.shields.io/pypi/v/agent-trace)](https://pypi.org/project/agent-trace/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/RudrenduPaul/agent-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/RudrenduPaul/agent-observability/actions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/RudrenduPaul/agent-observability/badge)](https://securityscorecards.dev/viewer/?uri=github.com/RudrenduPaul/agent-observability)

---

## Why reproducing agent failures is expensive

A LangGraph run fails after step 8. Your trace in LangSmith or Langfuse shows *what* broke. But to reproduce it you have to re-run the entire agent: 8 more LLM calls, 30 more seconds, another $0.15 in API cost. If the failure was caused by a specific tool response or a transient model output, you can't reproduce it at all. You're debugging against a moving target.

**Agent Observability solves this at the HTTP transport layer.** It records every request and response verbatim to a local SQLite file. Replay serves those exact bytes back in sequence, in under 1ms per exchange: same code path, same span tree, same failure. No API calls.

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

Replay offline, no API calls, no tokens:

```python
from agent_trace import replay

with replay("run_<id>") as ctx:
    result = fetch_data("hello")  # same call — served from fixture, zero network
    print(result)  # identical to the original run
```

> To store the input for later retrieval in replay, call `ctx.fixture.set_metadata('input', query)` inside the recording context.

---

## How Agent Observability compares

Most observability tools for LLM agents are **observe-only**: they show you a trace of what happened, but reproducing a failure still requires re-running the full agent against live APIs. The table below is based on published benchmarks, official documentation, and GitHub issue threads.

### Capability matrix

| | **Agent Observability** | LangSmith | Langfuse | Helicone | OpenLLMetry |
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

² Helicone's active maintenance status is uncertain as of mid-2026; verify at helicone.ai before taking a dependency.

### Overhead per LLM call (published benchmarks)

| Tool | Per-call overhead | Mechanism | Source |
|------|------------------|-----------|--------|
| **Agent Observability** | **0.15 ms P50** per exchange | In-process SQLite WAL write, measured 2026-06-19 | `benchmarks/test_ingestion.py` |
| Langfuse SDK | 0.10–0.15 ms (queue insert) | Async in-memory queue; network I/O in background | [Langfuse SDK Performance Test](https://langfuse.com/guides/cookbook/langfuse_sdk_performance_test) |
| LangSmith | < 4 ms (async batch mode) | Background thread + PriorityQueue to cloud | [LangSmith production guide](https://docs.smith.langchain.com/) |
| OpenLLMetry | ~1–5 ms | OTel SDK span creation + async OTLP export | Traceloop documentation |
| Helicone | 10–30 ms (cloud) · 8 ms P50 (self-hosted) | Proxy hop; every LLM call routes through Helicone servers | [Helicone latency docs](https://docs.helicone.ai/references/latency-affect) |

**Note on Langfuse:** The 0.10–0.15 ms figure is the in-process queue insert only. The LangChain callback wrapper adds ~88 ms due to synchronous wrapping overhead. Source: [Langfuse SDK benchmark page](https://langfuse.com/guides/cookbook/langfuse_sdk_performance_test).

### Replay cost

| Scenario | Agent Observability | LangSmith VCR | Langfuse Playground | Phoenix Span Replay |
|----------|:-------------------:|:-------------:|:-------------------:|:-------------------:|
| Cost per CI replay iteration | **$0** | $0 after recording | 1 live LLM call | 1 live LLM call |
| Reproduce intermittent failure | **Always** | Yes (if captured) | No | No |
| Requires cloud account | **No** | Yes | Yes or self-host | Yes or self-host |
| Works for non-LangChain agents | **Yes** | No ¹ | N/A | N/A |
| Captures raw response bytes | **Yes** | No (structured only) | No (structured only) | No (structured only) |

### What competitors do well

**Choose LangSmith** if your team is on LangChain and needs dataset management, prompt versioning, and human feedback loops. Its VCR cassette system covers the CI-replay use case for LangChain-based agents with a LangSmith account.

**Choose Langfuse** if you want a fully open-source, self-hostable observability stack with strong Postgres-backed storage. Best when you need dashboards and evals, not offline replay.

**Choose OpenLLMetry** if your team already runs on OpenTelemetry and wants standard `gen_ai.*` spans from LLM calls without adding a new observability system.

**Agent Observability is not a replacement for dashboards and eval pipelines.** It solves the specific upstream problem: reproducing a specific failed run without any LLM API cost, for any agent built on any Python HTTP library.

---

## Self-host traces

Agent Observability emits OTLP spans. Run a local observability stack to browse trace trees:

```bash
docker compose up -d
```

The `docker-compose.yml` starts three services (all optional, stop any you don't need):
- **Jaeger** (port 16686): OTLP span ingestion and trace UI
- **Grafana** (port 3000): dashboards and alerts
- **Tempo** (port 3200): long-term trace storage backend

Open [http://localhost:16686](http://localhost:16686) for Jaeger's trace browser.
Open [http://localhost:3000](http://localhost:3000) for Grafana dashboards.

```python
from agent_trace.exporters.otlp import OTLPExporter

# 4317 = OTLP gRPC ingestion endpoint (send traces here).
# View traces at localhost:16686 (Jaeger) or localhost:3000 (Grafana).
exporter = OTLPExporter(endpoint="http://localhost:4317")
exporter.export(trace)
```

---

## Community

- [GitHub Issues](https://github.com/RudrenduPaul/agent-observability/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/RudrenduPaul/agent-observability/discussions) — questions and ideas
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup and PR guide
