# Agent Observability

**Deterministic record/replay for LLM agents.** Capture a failing agent run once, reproduce it offline in under 2 ms with zero API calls, on any Python HTTP client.

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

## The problem

A LangGraph run fails after step 8. Your trace in LangSmith or Langfuse shows *what* broke. But to reproduce it you have to re-run the entire agent: 8 more LLM calls, 30 more seconds, another $0.15 in API cost. If the failure was caused by a specific tool response or a transient model output, you can't reproduce it at all. You're debugging against a moving target.

**Agent Observability solves this at the HTTP transport layer.** It records every request and response verbatim to a local SQLite file. Replay serves those exact bytes back in sequence, in under 1 ms per exchange: same code path, same span tree, same failure. No API calls.

```
Recording overhead:   3.5%  (56.9 ms → 58.9 ms on a 10-step workflow)
Replay latency:       1.48 ms mean  (vs 8,000 ms live on GPT-4o × 10 steps)
Replay fidelity:      100%  (response bytes byte-for-byte identical to recorded)
CI cost per replay:   $0
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
    result = fetch_data("hello")  # served from fixture, zero network
    print(result)                 # identical to the original run
```

> To store the input for later retrieval in replay, call `ctx.fixture.set_metadata('input', query)` inside the recording context.

> **Sync clients only (v0.1):** Agent Observability currently intercepts `httpx.Client` and `requests.Session`. `httpx.AsyncClient` — used by default in the OpenAI Python SDK v1.x and Anthropic SDK — is not yet intercepted. Async support is planned for v0.3. Use the synchronous `openai.OpenAI()` client when recording.

---

## Use in CI: replay at zero cost

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

Set `AGENT_TRACE_NETWORK_GUARD=1` in CI. Any HTTP call not in the fixture raises `NetworkGuardError` immediately — catching regressions before they hit production.

```bash
AGENT_TRACE_NETWORK_GUARD=1 uv run pytest tests/
```

---

## How Agent Observability compares

Most observability tools for LLM agents are **observe-only** — they show you a trace of what happened, but reproducing a failure still requires re-running the full agent against live APIs.

| Capability | Agent Observability | LangSmith | Langfuse | Helicone | OpenLLMetry |
|---|---|---|---|---|---|
| Offline replay from local fixture | **Yes** | Partial ¹ | No | No | No |
| Works with any HTTP client | **Yes** | No | No | No | No |
| CI replay without API keys | **Yes** | Partial ¹ | No | No | No |
| Deterministic span timing in replay | **Yes** | No | No | No | No |
| Captures raw HTTP request/response bytes | **Yes** | No | No | Yes | No |
| Span-level tracing | Yes | Yes | Yes | Yes | Yes |
| OTLP export (Jaeger, Grafana Tempo) | Yes | No | Yes | No | Yes |
| Open-source core | Yes | No | Yes | No | Yes |
| Local-only, no server required | Yes | No | Self-host | No | Self-host |

¹ LangSmith has `LANGSMITH_TEST_CACHE` / VCR cassettes (`langsmith[vcr]`) for Python + LangChain only. It captures HTTP to `api.openai.com` but not arbitrary HTTP clients, does not record full wire-level bytes, and requires a LangSmith account.

**Choose LangSmith** if your team is on LangChain and needs dataset management, prompt versioning, and human feedback loops.

**Choose Langfuse** if you want a fully open-source, self-hostable observability stack with strong Postgres-backed storage.

**Choose OpenLLMetry** if your team already runs on OpenTelemetry and wants standard `gen_ai.*` spans without adding a new observability system.

**Agent Observability** is not a replacement for dashboards and eval pipelines. It solves the specific upstream problem: reproducing a specific failed run without any LLM API cost, for any agent built on any Python HTTP client.

---

## Try it with Docker

Agent Observability emits OTLP spans. Run a local observability stack to browse trace trees:

```bash
git clone https://github.com/RudrenduPaul/agent-observability
cd agent-observability
docker compose up -d
```

Starts three services (all optional):

- **Jaeger** (`http://localhost:16686`) — OTLP span ingestion and trace UI
- **Grafana** (`http://localhost:3000`) — dashboards and alerts
- **Tempo** (port 3200) — long-term trace storage backend

Then point your exporter at the collector:

```python
from agent_trace.exporters.otlp import OTLPExporter

# 4317 = OTLP gRPC ingestion endpoint
exporter = OTLPExporter(endpoint="http://localhost:4317")
exporter.export(trace)
```

---

## Security

- **Supply chain:** SLSA Level 2 via GitHub Actions provenance. All releases signed with Sigstore. SBOM attached to every GitHub Release.
- **Vulnerability scanning:** Dependabot keeps all GitHub Actions and Python dependencies current. Secret scanning auto-enables when the repo goes public.
- **Fixture safety:** Fixture files at `~/.agent-trace/runs/` contain full HTTP request and response bodies, including API keys and prompt contents. Add `.agent-trace/` and `*.db` to your `.gitignore`. Never commit a fixture generated against a production API key.
- **Disclosure:** [SECURITY.md](SECURITY.md) — report vulnerabilities to `agenttrace.oss.security@gmail.com` with a 48-hour response SLA.

---

## Contributing

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR
- Good first issues are labeled in [GitHub Issues](https://github.com/RudrenduPaul/agent-observability/issues)
- Replay engine (`src/agent_trace/_replay/`) requires 90% test coverage — correctness-critical
- Interceptor (`src/agent_trace/interceptor/`) requires 90% test coverage
- GitHub Discussions for design questions and ideas

Apache 2.0. Contributions welcome.

---

*Built by Rudrendu Paul*
