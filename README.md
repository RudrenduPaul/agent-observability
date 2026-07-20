# Agent Observability

Your LangGraph agent fails after step 8. LangSmith shows you what broke.
To reproduce it: 8 more LLM calls. 30 more seconds. $0.15 more in API cost.
If the failure was caused by a transient model output, you can't reproduce it at all.

**Agent Observability fixes this.** Record once. Replay offline in 0.93 ms. Zero API calls. Zero cost.

```
Recording overhead:   0.011%   (0.090 ms added per LLM call)
Replay latency:       0.93 ms  mean (vs ~8,500 ms live on GPT-4o × 10 steps)
Replay fidelity:      100%     (response bytes byte-for-byte identical)
CI cost per replay:   $0
```

![Terminal recording of agent-trace recording a live HTTP call, then replaying the same run offline with zero network requests](https://raw.githubusercontent.com/RudrenduPaul/agent-observability/main/docs/assets/dev-to-demos/demo-1-record-replay.gif)

[![PyPI](https://img.shields.io/pypi/v/agent-observability-trace-cli)](https://pypi.org/project/agent-observability-trace-cli/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/RudrenduPaul/agent-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/RudrenduPaul/agent-observability/actions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/RudrenduPaul/agent-observability/badge)](https://securityscorecards.dev/viewer/?uri=github.com/RudrenduPaul/agent-observability)

---

## Install

```bash
pip install agent-observability-trace-cli
# or
uv add agent-observability-trace-cli
```

LangGraph support:

```bash
pip install agent-observability-trace-cli[langgraph]
```

OpenAI Agents SDK support:

```bash
pip install agent-observability-trace-cli[openai-agents]
```

![Terminal recording of installing agent-observability-trace-cli into a fresh virtual environment, then running agent-trace version and recording a first HTTP call with agent-trace list showing the resulting run](https://raw.githubusercontent.com/RudrenduPaul/agent-observability/main/docs/demo.gif)

## Supported frameworks

LangGraph · OpenAI Agents SDK · CrewAI · AutoGen · LlamaIndex · Haystack · Agno · PydanticAI · Google GenAI
Plus: any `httpx.Client`, `httpx.AsyncClient`, or `requests.Session` — no framework required.

## 30-second CLI quickstart

```bash
# Record a live run (your script just needs `import agent_trace` somewhere)
agent-trace run --name my_agent -- python my_agent.py

# List recorded runs
agent-trace list

# Replay offline — zero network, zero cost
agent-trace replay run_<id>

# Show the trace for a run
agent-trace show run_<id>
```

Want programmatic control instead of the CLI? Use the Python API:

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

> **Sync and async clients:** Agent Observability intercepts `httpx.Client`, `httpx.AsyncClient`, and `requests.Session` — including the async client used by default in the OpenAI Python SDK v1.x and Anthropic SDK. The patch is installed at request-dispatch time, so it also covers clients constructed before recording/replay starts (e.g. a module-level `openai.AsyncOpenAI()` instance).

![Terminal recording of replaying a previously recorded run with zero network calls, then running agent-trace show to print the replayed span tree](https://raw.githubusercontent.com/RudrenduPaul/agent-observability/main/docs/usage.gif)

## The problem

A LangGraph run fails after step 8. Your trace in LangSmith or Langfuse shows *what* broke. But to reproduce it you have to re-run the entire agent: 8 more LLM calls, 30 more seconds, another $0.15 in API cost. If the failure was caused by a specific tool response or a transient model output, you can't reproduce it at all. You're debugging against a moving target.

**Agent Observability solves this at the HTTP transport layer.** It records every request and response verbatim to a local SQLite file. Replay serves those exact bytes back in sequence, in under 1 ms per exchange: same code path, same span tree, same failure. No API calls.

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

## What does this save you?

```
10-step agent × $0.15 per run × 10 debug sessions per week = $15/week in API costs
With Agent Observability CI replay: $0/week

At scale (10 engineers, each debugging 3 failures/week):
Before: ~$45/week, ~5 hours/week waiting for live re-runs
After: $0/week, 0.93 ms per replay
```

---

## Why not just use LangSmith, Langfuse, or Helicone?

Short answer: they show you what happened. They can't reproduce it offline.
LangSmith's VCR cassettes are Python + LangChain only, don't capture full wire bytes,
and require a LangSmith account. Agent Observability works on any Python HTTP client,
needs no account, and replays in 0.93 ms with 100% fidelity.

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

## Real failures record/replay catches

- Transient model output at step 6 causes a downstream tool to fail — unreproducible with a re-run, trivial to replay
- Rate-limit response at step 3 triggers a silent fallback path — only visible in the recorded fixture bytes, not in a live re-run
- Tool schema serialization error before HTTP dispatch — caught by `LangGraphTracer.on_llm_error` even though it never reaches the interceptor (see Known Limitations)
- Non-deterministic tool ordering in a parallel branch — replay pins the exact sequence that produced the failure, so you're debugging the actual run instead of a fresh one
- gRPC unary-stream response from Gemini that only fails on a specific chunk boundary — recorded once, replayed byte-for-byte instead of re-triggering a live streaming call each time

---

## Known limitations

Agent Observability's capture model is HTTP-interceptor-based (plus
instrumented framework callbacks for the integrations under
`src/agent_trace/integrations/`) and process-local. That model has real
edges — stated explicitly here so they're clear before you hit one, not
after:

- **Process-local only.** Recording/replay happens inside the Python
  process you import `agent_trace` into (`httpx.Client(transport=
  RecordingTransport(...))`, `session.mount(..., RecordingAdapter(...))`,
  or `ReplayEngine.replay()`'s monkeypatches — see
  `src/agent_trace/interceptor/`). It cannot observe or replay calls made
  by a third-party **hosted** service you don't run or deploy yourself
  (e.g. a vendor's own hosted chat assistant) — only your own process's
  outbound calls.

- **gRPC coverage is partial.** `src/agent_trace/interceptor/grpc_hook.py`
  patches `grpc.secure_channel`/`grpc.insecure_channel` (and the `grpc.aio`
  equivalents) to capture Gemini/Vertex AI traffic that bypasses `httpx`
  entirely — unary-unary calls (e.g. `GenerateContent`) and sync
  unary-stream calls (e.g. `StreamGenerateContent`) are fully recorded and
  replayed. Client-streaming and bidirectional-streaming gRPC calls, and
  any `grpc.aio` streaming call, are **not** captured — those go straight
  to the live network unintercepted, both during recording and (if
  attempted) replay.

- **Capture starts once a request object exists.** `RecordingTransport.
  handle_request`/`AsyncRecordingTransport.handle_async_request`
  (`httpx_hook.py`) and `RecordingAdapter.send`
  (`requests_patch.py`) only run once a fully-constructed
  `httpx.Request`/`PreparedRequest` reaches them. Any exception raised
  *before* that — while an SDK is serializing a tool schema, building
  headers, or otherwise assembling the call, or even earlier, during plain
  Python object construction (e.g. `TypeError` from `abc.ABCMeta` when
  instantiating an abstract class incorrectly) — happens entirely upstream
  of the interceptor's capture surface and produces zero fixture rows.
  A wired-in framework integration's own error callback (e.g.
  `LangGraphTracer.on_llm_error`) *does* still capture such pre-HTTP
  exceptions when they propagate through that framework's own
  `try`/`except` — so "invisible to the interceptor" is not the same as
  "invisible everywhere": it depends on whether a framework integration is
  wired in for the exception to pass through.

- **No visibility into a framework's own print/display code.** Exceptions
  raised inside local logging/printing/display machinery — e.g. `rich`
  Console output, IPython/Jupyter display hooks, triggered by a framework's
  own `verbose=True` logging — have zero HTTP traffic and zero framework
  callback surface. No existing or planned capture mechanism (HTTP
  interceptor, MCP stdio-transport hook, or any framework integration)
  observes this category of failure.

---

## Security

- **Supply chain:** Releases are built and published via GitHub Actions (`release.yml`). Sigstore signing and SBOM generation are wired into that workflow but not yet producing signed, SBOM-attached release assets end to end — treat that as in progress, not shipped, until a release actually carries signed artifacts.
- **Vulnerability scanning:** `dependabot.yml` opens weekly pip and monthly GitHub Actions version-bump PRs. Dependabot security-advisory alerts, secret scanning, and secret scanning push protection are all enabled on this repo.
- **Fixture safety:** Fixture files at `~/.agent-trace/runs/` contain full HTTP request and response bodies, including API keys and prompt contents. Add `.agent-trace/` and `*.db` to your `.gitignore`. Never commit a fixture generated against a production API key.
- **Disclosure:** [SECURITY.md](SECURITY.md) — report vulnerabilities to `agent.obs.oss.security@gmail.com` with a 48-hour response SLA.

---

## FAQ

**What is Agent Observability, and how is it different from a typical LLM tracing tool?**

It's a Python library and CLI (`agent-trace`) that records every HTTP request and response your agent makes, verbatim, to a local SQLite fixture, then replays those exact bytes later with no network call. Most tracing tools, including LangSmith, Langfuse, Helicone, and OpenLLMetry, show you what happened during a run. Agent Observability additionally lets you reproduce that exact run offline, deterministically, without touching the live API. See "Why not just use LangSmith, Langfuse, or Helicone?" above for the full capability breakdown against those four tools.

**How does deterministic record/replay actually work?**

Recording patches `httpx.Client`, `httpx.AsyncClient`, and `requests.Session` at the transport layer (`src/agent_trace/interceptor/`) to capture every outbound request and response as raw bytes into `fixture.db`. Replay installs a `FixtureClock` (`src/agent_trace/core/clock.py`) and serves those same bytes back in the original sequence, so the code path, span tree, and timestamps all match the original recording. The benchmark numbers quoted above (0.011% recording overhead, 0.93ms mean replay latency, 100% fidelity) come from `benchmarks/test_overhead.py`, `benchmarks/test_replay_vs_live.py`, and `benchmarks/test_fidelity.py` in this repo, runnable yourself with `uv run pytest benchmarks/`.

**How do I install it, and what platforms does it support?**

`pip install agent-observability-trace-cli`, or `uv add agent-observability-trace-cli`. It requires Python 3.10 or newer and depends only on `httpx` and `rich`, no compiled extensions, so it installs anywhere those wheels do. CI (`.github/workflows/ci.yml`) passes on Ubuntu, macOS, and Windows, across Python 3.10 through 3.13. An npm wrapper, [`agent-observability-trace-cli`](npm/) (source under [`npm/`](npm/) in this repo), is also published for teams that reach for `npx`/`npm`, but it still shells out to the Python CLI under the hood, so the Python package must be installed too.

**How does this compare to LangSmith specifically?**

LangSmith's `LANGSMITH_TEST_CACHE` (VCR-style cassettes, via `langsmith[vcr]`) is the closest built-in equivalent. It's Python plus LangChain only, captures HTTP calls to `api.openai.com` rather than any HTTP client, doesn't record full wire-level bytes, and requires a LangSmith account. Agent Observability works with any Python HTTP client, plus dedicated interceptors for gRPC, aiohttp, botocore, and WebSocket traffic, records full request and response bytes locally, and needs no account or hosted service. Pick LangSmith if you're already on LangChain and want dataset management, prompt versioning, and human feedback loops alongside tracing. Pick Agent Observability if the goal is reproducing one specific failed run at zero API cost, regardless of which SDK made the call.

**What happens if replay can't find a matching fixture entry?**

With `AGENT_TRACE_NETWORK_GUARD=1` set, any request missing from the fixture raises `NetworkGuardError` immediately instead of silently falling through to a live call. The most common cause is an HTTP client constructed before the recording or replay context was entered, since the patch only applies to clients created inside the `start_trace`/`replay` block. See "Known limitations" above for the full list of edges, including partial gRPC streaming coverage and pre-HTTP exceptions that never reach the interceptor.

**Does it capture agents built on non-Python frameworks?**

No. Capture is a Python HTTP-transport interceptor plus instrumented callbacks for the integrations under `src/agent_trace/integrations/` (LangGraph, CrewAI, AutoGen, LlamaIndex, Haystack, Agno, PydanticAI, Google GenAI, and others). It only sees traffic from your own Python process. Agents built in other languages, or calls made by a third-party hosted service you don't run yourself, are outside its capture surface.

**Are fixture files safe to commit to version control?**

Not by default. `fixture.db` contains full HTTP request and response bodies, which means API keys and prompt contents whenever they appear in headers or payloads. Add `.agent-trace/` and `*.db` to `.gitignore`, and never commit a fixture recorded against a production API key. Strip or redact secrets first if you want to keep a fixture as a committed CI test asset.

**Is this free to use commercially?**

Yes. The project is Apache 2.0 licensed (see [LICENSE](LICENSE)), which permits commercial use, modification, and redistribution, including inside closed-source products, subject to the license's own attribution and notice terms. There is no separate paid tier or commercial license.

---

## Contributing

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR
- Good first issues are labeled in [GitHub Issues](https://github.com/RudrenduPaul/agent-observability/issues)
- Replay engine (`src/agent_trace/_replay/`) requires 80% test coverage — correctness-critical
- Interceptor (`src/agent_trace/interceptor/`) requires 80% test coverage
- GitHub Discussions for design questions and ideas

Apache 2.0. Contributions welcome.

---

*Built by Rudrendu Paul*

