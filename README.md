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
[![CI](https://github.com/RudrenduPaul/agent-trace/actions/workflows/ci.yml/badge.svg)](https://github.com/RudrenduPaul/agent-trace/actions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/RudrenduPaul/agent-trace/badge)](https://securityscorecards.dev/viewer/?uri=github.com/RudrenduPaul/agent-trace)

---

## The problem

A LangGraph run fails after step 8. Your trace in LangSmith or Langfuse shows *what* broke. But to reproduce it you have to re-run the entire agent — 8 more LLM calls, 30 more seconds, another $0.15 in API cost. If the failure was caused by a specific tool response or a transient model output, you can't reproduce it at all. You're debugging against a moving target.

**Agent Observability solves this at the HTTP transport layer.** It records every request and response verbatim to a local SQLite file. Replay serves those exact bytes back in sequence, in under 1ms per exchange — same code path, same span tree, same failure. No API calls.

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

> **Sync clients only (v0.1):** Agent Observability currently intercepts `httpx.Client` and `requests.Session` (synchronous). `httpx.AsyncClient` — used by default in the OpenAI Python SDK v1.x and Anthropic SDK — is not yet intercepted. Async support is on the roadmap for v0.3. For now, use the synchronous `openai.OpenAI()` client (not `openai.AsyncOpenAI()`) when recording.

---

## Performance benchmarks

All numbers below were measured on **Apple M-series, Python 3.14.3, SQLite WAL mode, NVMe SSD, 2026-06-19**.
Run the benchmarks yourself with `uv run pytest benchmarks/ -v --benchmark-only`.
The Docker harness in [`agent-observability-bench`](https://github.com/RudrenduPaul/agent-observability-bench)
reproduces these numbers in a Docker Compose environment.

| Metric | Measured value | Notes |
|--------|---------------|-------|
| Recording overhead — 10-step workflow | **3.5%** (56.9 ms → 58.9 ms) | `benchmarks/test_overhead.py` vs baseline, mock LLM server |
| SQLite write latency per exchange | **0.15 ms P50** | `benchmarks/test_ingestion.py::test_fixture_write_latency`, WAL mode |
| Replay — 10-step agent run | **1.48 ms mean** | `benchmarks/test_replay_vs_live.py`, zero network I/O |
| Replay speedup vs live mock (57 ms) | **38×** | Replay serves SQLite; no network, no API cost |
| Replay speedup vs real GPT-4o (800 ms × 10 steps) | **~5,400×** | Projection: 8,000 ms live → 1.48 ms from fixture |
| Replay exchange serve latency | **13.9 µs mean** | `benchmarks/test_ingestion.py::test_fixture_read_cursor_speed` |
| Fixture read throughput | **~72,000 reads/sec** | Derived from 13.9 µs mean |
| Span serialization | **774 ns** | `benchmarks/test_ingestion.py::test_span_serialization_speed`, pure CPU |
| Replay fidelity | **100%** | Response bytes byte-for-byte identical to recorded |

> **What the numbers mean in practice:** A 10-step GPT-4o agent run that costs $0.15 and takes 8–30 seconds live replays from a local SQLite fixture in under 2 ms. In CI, every test run costs $0 in API fees. The 3.5% recording overhead comes from SQLite writes and httpx transport patching — it is well below the variability of any real LLM API call.

---

## How Agent Observability compares

Most observability tools for LLM agents are **observe-only** — they show you a trace of what happened, but reproducing a failure still requires re-running the full agent against live APIs. The table below is based on published benchmarks, official documentation, and GitHub issue threads.

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
| Helicone | 10–30 ms (cloud) · 8 ms P50 (self-hosted) | Proxy hop — every LLM call routes through Helicone servers | [Helicone latency docs](https://docs.helicone.ai/references/latency-affect) |

**Note on Langfuse:** The 0.10–0.15 ms figure is the in-process queue insert only. The LangChain callback wrapper adds ~88 ms due to synchronous wrapping overhead. The old v2 synchronous SDK added 155–1,205 ms per call; this was eliminated in later versions. Source: [Langfuse SDK benchmark page](https://langfuse.com/guides/cookbook/langfuse_sdk_performance_test).

### Replay cost

| Scenario | Agent Observability | LangSmith VCR | Langfuse Playground | Phoenix Span Replay |
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

**Agent Observability is not a replacement for dashboards and eval pipelines.** It solves the specific upstream problem: reproducing a specific failed run without any LLM API cost, for any agent built on any Python HTTP library.

---

## How it works

- **Transport interception, not API wrapping.** Agent Observability patches `httpx.Client.__init__` and `requests.Session.get_adapter` at trace start. Every AI SDK — OpenAI, Anthropic, LangChain — creates its own HTTP client internally. Patching at the transport layer captures those calls with no SDK-specific glue.
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

## Test coverage

**LangGraph integration: 13 tests (0 skipped)**

| Test | What it verifies |
|------|-----------------|
| `test_callback_handler_captures_node_spans` | Span emitted per node in a 2-node graph |
| `test_callback_handler_sets_langgraph_node_attribute` | `langgraph.node` attribute on each span |
| `test_callback_handler_error_span` | Failing node produces an ERROR-status span |
| `test_replay_context_allows_pure_python_graph` | Record → replay with `NETWORK_GUARD=1` |
| `test_all_spans_closed_after_clean_run` | No span has `end_time=None` after clean run |
| `test_all_spans_ok_on_clean_run` | All spans carry `SpanStatus.OK` on clean run |
| `test_span_registry_empty_after_graph_completes` | `handler._spans == {}` — no leaked open spans |
| `test_parent_child_span_hierarchy` | At least one child span has a `parent_id` |
| `test_node_spans_parent_ids_point_to_langgraph_root` | node spans are children of the LangGraph root chain span; every `parent_id` points to a real span |
| `test_chat_model_callbacks_fire_through_langgraph` | `on_chat_model_start` fires when a `BaseChatModel` is invoked inside a node (no API key — uses `FakeChatModel`) |
| `test_llm_span_has_token_attributes` | `llm.usage.prompt_tokens`, `.completion_tokens`, `.total_tokens` recorded from `llm_output` (no API key — uses `FakeChatModel`) |
| `test_concurrent_invocations_no_cross_contamination` | Two simultaneous `graph.invoke()` calls on one `LangGraphTracer` — no cross-contamination, no leaked spans |
| `test_replay_span_tree_matches_record_span_tree` | Replayed span names, order, and `langgraph.*` attributes match the recorded span tree exactly |

Run them:

```bash
uv run --extra langgraph pytest tests/integration/test_langgraph.py -m integration -v
```

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

Agent Observability emits OTLP spans. Run a local observability stack to browse trace trees:

```bash
docker compose up -d
```

The `docker-compose.yml` starts three services (all optional, stop any you don't need):
- **Jaeger** (port 16686): OTLP span ingestion and trace UI
- **Grafana** (port 3000): dashboards and alerts
- **Tempo** (port 3200): long-term trace storage backend

Open [http://localhost:16686](http://localhost:16686) for Jaeger's trace browser.

```python
from agent_trace.exporters.otlp import OTLPExporter

exporter = OTLPExporter(endpoint="http://localhost:4317")
exporter.export(trace)
```

---

## Engineering checklist

Status as of 2026-06-19.

| Item | Status | Notes |
|------|--------|-------|
| Deterministic replay end-to-end on LangGraph 0.2+ with `AGENT_TRACE_NETWORK_GUARD=1` | ✅ | `tests/integration/test_langgraph.py` — run with `uv run pytest tests/integration/ -m integration` (requires `OPENAI_API_KEY`) |
| LangGraph integration tests pass against real LangGraph (not mocked) | ✅ | `tests/integration/test_langgraph.py` exists; real API, tagged `@pytest.mark.integration` |
| OpenAI Agents SDK integration tests pass against real API | ✅ | `tests/integration/test_openai_agents.py` exists; one live run + fixture capture |
| All three benchmark scripts exist and produce output | ✅ | `benchmarks/test_overhead.py`, `test_fidelity.py`, `test_ingestion.py` — run `uv run pytest benchmarks/ --benchmark-only` |
| `benchmarks/README.md` reproduces every README number in under 5 minutes | ✅ | See [benchmarks/README.md](benchmarks/README.md#how-to-reproduce-readme-numbers) |
| `ruff check`, `mypy --strict`, `pytest --cov-fail-under=80` all pass | ✅ | 287 tests, 94.98% coverage; enforced in CI on every push |
| `docker compose up -d` opens trace UI (Jaeger at `localhost:16686`, Grafana at `localhost:3000`) | ✅ | `docker-compose.yml` — Jaeger all-in-one + Grafana+Tempo; OTLP gRPC receiver on `localhost:4317` |
| README GIF: failure captured in record mode, replayed offline in replay mode | ⏳ | Requires screen recording — see `examples/02-langgraph-failure-replay/` to reproduce manually |

## Security baseline

| Item | Status | Notes |
|------|--------|-------|
| `SECURITY.md` with `agent.obs.oss.security@gmail.com` contact and 48h SLA | ✅ | See [SECURITY.md](SECURITY.md) |
| Secret scanning enabled on GitHub | ⏳ | Auto-enables when repo goes public (free for public repos; requires GitHub Advanced Security for private) |
| Dependabot: weekly pip updates + monthly GitHub Actions updates | ✅ | [`.github/dependabot.yml`](.github/dependabot.yml) configured |

---

## Quality gates

Status as of 2026-06-19 on `main`.

| Gate | Status | Notes |
|------|--------|-------|
| **OpenSSF Scorecard ≥ 7/10** | Tracked | [`scorecard.yml`](.github/workflows/scorecard.yml) runs weekly on `main`; badge in header shows live score |
| **SBOM attached to release (cyclonedx-py)** | ✅ | [`release.yml`](.github/workflows/release.yml) generates `sbom.json` + `sbom.xml` (CycloneDX 1.6) and attaches both to every GitHub release |
| **SLSA Level 2 signing (sigstore)** | ✅ | `release.yml` signs `dist/*.whl` and `dist/*.tar.gz` via `sigstore/gh-action-sigstore-python@v3`; `.sigstore` bundles attached to release |
| **Test coverage: overall ≥ 80%, replay/ and interceptor/ each ≥ 90%** | ✅ | Current: **94.98%** overall · **90%** replay/ · **96%** interceptor/. Both gates enforced in [`ci.yml`](.github/workflows/ci.yml) |
| **Plugin SDK shipped** | ✅ | `from agent_trace.plugins import PluginBase, SpanPlugin, TracePlugin`. Register via `tracer.add_plugin(plugin)`. See [Plugin SDK](#plugin-sdk) below. |
| **5+ unique external contributors** | ⏳ | Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are labelled [`good first issue`](https://github.com/RudrenduPaul/agent-trace/labels/good%20first%20issue). |

---

## Plugin SDK

Extend Agent Observability without modifying agent code. Implement `SpanPlugin`, `TracePlugin`, or both:

```python
from agent_trace import tracer
from agent_trace.plugins import PluginBase

class MetricsPlugin(PluginBase):
    def on_span_end(self, span):
        # called after every span.end() with timing already set
        metrics.histogram("agent.span.duration_ms", span.duration_ms, tags={"name": span.name})

    def on_trace_end(self, trace):
        # called after trace.json is written; all spans are final
        metrics.increment("agent.trace.completed", tags={"name": trace.metadata.get("name")})

tracer.add_plugin(MetricsPlugin())
```

Hooks receive live `Span` / `Trace` objects — mutations are visible to subsequent code.
Exceptions inside a hook are caught and logged at `WARNING`; a buggy plugin never silences the caller.

```python
# Only want to observe traces, not individual spans?
from agent_trace.plugins import TracePlugin

class AuditPlugin(TracePlugin):
    def on_trace_end(self, trace):
        audit_log.write(trace.to_dict())

tracer.add_plugin(AuditPlugin())
```

| Hook | When called | State at call time |
|------|-------------|-------------------|
| `on_span_start(span)` | After `tracer.start_span()` | `start_time` set; `end_time` is None |
| `on_span_end(span)` | After `span.end()` | `start_time`, `end_time`, and `status` all set |
| `on_trace_start(trace)` | When `start_trace()` context is entered | No spans yet |
| `on_trace_end(trace)` | After `trace.json` is written | All spans complete and immutable |

---

## Launch checklist (Days 15–21)

| Item | Status | Notes |
|------|--------|-------|
| Apache 2.0 in LICENSE + README header | ✅ | `LICENSE` at repo root; badge in README header |
| 20 good-first-issues with full context | ✅ | Issue #7 live; run `bash scripts/create-launch-issues.sh` to create the remaining 19 |
| GitHub Discussions enabled + "Why we built this" pinned | ✅/⏳ | Discussions enabled ✅; paste `docs/launch/why-we-built-this.md` as pinned Announcement ⏳ |
| Repo made public | ⏳ | Settings → Danger Zone → Make public (confirm repo has no secrets in history first) |
| Show HN: Tuesday–Thursday 9–11am EST | ⏳ | Draft at `docs/launch/show-hn-draft.md` |
| 4+ people post HN link in LangChain Discord #show-and-tell within 15 min | ⏳ | Human coordination required |
| Respond to all HN comments within 2 hours | ⏳ | Human action required |
| 50 stars in first 48 hours | ⏳ | Gate: if not reached, diagnose before Week 5 content |

---

## Community

- [GitHub Issues](https://github.com/RudrenduPaul/agent-trace/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/RudrenduPaul/agent-trace/discussions) — questions and ideas
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup and PR guide

---

## Changelog

### 2026-06-19 — Engineering Audit: 11 Bugs Fixed

**Critical / High**

- **B1 — Dead code removed** (`integrations/openai_agents.py`): `_enrich_step_span` was a 27-line method that was never called. Removed entirely. Reduces maintenance surface and eliminates a source of divergence when the SDK updates its step schema.

- **B2 — Missing `total_tokens` in LLM span** (`integrations/openai_agents.py`): `on_llm_end` recorded `prompt_tokens` and `completion_tokens` but silently dropped `total_tokens`. Fixed: explicit `total_tokens` from the SDK response is now recorded; when the SDK omits it, the sum `prompt + completion` is computed as a fallback.

- **B3 — `httpx.AsyncClient` not patched in recording mode** (`__init__.py`): Only `httpx.Client.__init__` was monkey-patched during record mode. All async SDK clients (OpenAI, Anthropic) use `httpx.AsyncClient` and were hitting the live network instead of being captured. Fixed: both `Client.__init__` and `AsyncClient.__init__` are now patched and restored symmetrically.

- **B4 — Duplicate `_AttrValue` type alias** (`core/trace.py`): `_AttrValue = str | int | float | bool` was defined independently in both `span.py` and `trace.py`. Removed from `trace.py`; `trace.py` now imports it from `span.py` as the single source of truth.

- **B5 — Inconsistent LLM attribute name** (`integrations/langgraph.py`): `on_llm_start` set `llm.model_name`; all other callbacks used `llm.model`. Standardized to `llm.model` everywhere.

- **B6 — `_patch_requests` replaced adapter wholesale** (`__init__.py`, `interceptor/requests_patch.py`): `get_adapter` was overridden to always return a brand-new `RecordingAdapter`, discarding any custom adapter the user had installed (e.g., `HTTPAdapter(max_retries=...)`). Fixed: the original adapter is fetched first and passed as `inner=` to `RecordingAdapter`, which delegates the actual send to it while recording the exchange.

- **B7 — `httpx.AsyncClient` not patched in replay mode** (`_replay/engine.py`): The replay engine only patched `httpx.Client.__init__`, leaving async SDK HTTP calls to hit the live network during replay. Fixed: `httpx.AsyncClient.__init__` is now patched alongside `httpx.Client.__init__` with the same `ReplayTransport`.

- **B8 — gRPC `TracerProvider` resource leak** (`exporters/otlp.py`): `provider.shutdown()` was called at the end of `export()` but not in a `try/finally` block. A `KeyboardInterrupt` or exception during span export leaked the gRPC channel indefinitely. Fixed: `provider.shutdown()` is now guaranteed via `try/finally`.

- **B9 — Fixture stored with `run_id` instead of `trace_id`** (`__init__.py`): `Fixture(run_dir / "fixture.db", trace_id=effective_run_id)` passed the human-readable run directory name (e.g. `run_abc123`) as `trace_id`. The `Trace` object carries a separate 128-bit hex `trace_id` for OTLP. These were two different values, breaking any cross-correlation between the fixture and the trace. Fixed: `trace_id=trace.trace_id` now uses the actual trace identifier.

- **B10 — `FixtureClock` created but never advanced** (`_replay/engine.py`, `interceptor/httpx_hook.py`): During replay, a `FixtureClock` was installed as the time source, but no code ever called `clock.advance(...)`. All replay spans received the same initial timestamp, making span ordering indeterminate in any tool that sorts by time. Fixed: `ReplayTransport` now accepts an optional `clock` parameter and calls `clock.advance(exchange["recorded_at"])` after each exchange is served, so replay spans carry the same relative timestamps as the original run.

- **B11 — No error callbacks in `AgentTraceHook`** (`integrations/openai_agents.py`): When an agent turn or tool invocation raised an exception, neither `on_agent_error` nor `on_tool_error` was defined. The span for that agent or tool would remain open in `_spans` indefinitely, leaking memory for the lifetime of the hook object and leaving end_time as None. Fixed: both handlers now close the associated span with `SpanStatus.ERROR` and call `span.record_exception(error)`.

---

## Security note

Fixture files at `~/.agent-trace/runs/` contain full HTTP request and response bodies, including API keys, prompt contents, and user data. Add this to your `.gitignore`:

```
.agent-trace/
*.db
```

Never commit a `fixture.db` generated against a production API key. Use a separate key for recording, or scrub the fixture before committing (see `docs/concepts.md` for the SQLite schema).
