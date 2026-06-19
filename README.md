# agent-trace

Replay any failed LangGraph or OpenAI Agents SDK run offline, in under 3 seconds, with zero LLM API calls.

<!-- DEMO GIF: record a 12-step LangGraph run failing at step 7. Show replay executing against the local SQLite cache — no network activity. 6–8 seconds, terminal only. -->

[![PyPI](https://img.shields.io/pypi/v/agent-trace)](https://pypi.org/project/agent-trace/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/RudrenduPaul/agent-trace/badge)](https://securityscorecards.dev/viewer/?uri=github.com/RudrenduPaul/agent-trace)

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

## First trace

```python
from agent_trace import tracer
from agent_trace.exporters.stdout import StdoutExporter

# Decorate your agent function. record=True captures every outbound HTTP call.
@tracer.instrument(record=True)
def my_agent(query: str) -> str:
    with tracer.span("llm-call") as span:
        span.set_attribute("llm.model", "gpt-4o")
        # ... your LLM call here ...
        return "answer"

result = my_agent("what is 2+2?")
# Trace saved to ~/.agent-trace/runs/run_<id>/
# Fixture saved to ~/.agent-trace/runs/run_<id>/fixture.db
```

Replay it offline — no API calls, no tokens spent:

```python
from agent_trace import replay
from agent_trace.exporters.stdout import StdoutExporter

run_id = "run_abc123def456"  # printed after the recording step

with replay(run_id) as ctx:
    result = my_agent(ctx.get_metadata("input"))
    # Every httpx/requests call is served from fixture.db.
    # Zero bytes leave your machine.

# Print the span tree
import json
from pathlib import Path
trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
print(trace_path.read_text())
```

---

## Why this exists

Every LLM-backed agent team hits the same wall: a run fails after 9 steps, the Langfuse trace shows what broke, but reproducing the failure costs another 9 LLM calls and 45 seconds of wall time. If the failure is intermittent — a tool response that changes between runs, a model output that varies — you can't reproduce it at all. You're debugging against a moving target.

agent-trace intercepts HTTP at the transport layer. Every request your agent makes — OpenAI chat completions, Anthropic messages, tool API calls, vector DB queries — is recorded verbatim to a local SQLite file. On replay, the same bytes come back in the same order, from disk, in under 100 ms. The agent's code path is identical to the original run. The span tree matches. The failure reproduces. Now you can debug it.

---

## How it works

- **Transport interception, not API wrapping.** agent-trace patches `httpx.Client.__init__` and `requests.Session.get_adapter` at the moment your trace starts. Every AI SDK — OpenAI, Anthropic, LangChain — creates its own HTTP client internally. Patching at the transport layer means agent-trace captures those calls without any SDK-specific glue.
- **SQLite fixture, not JSON files.** Each recorded run writes to `~/.agent-trace/runs/<run_id>/fixture.db`. WAL mode lets multiple test workers open the same fixture concurrently. Large response bodies stay on disk until replayed — memory use stays flat regardless of response size.
- **Per-(method, URL) cursor.** If your agent calls `POST /v1/chat/completions` three times, the fixture stores all three responses in sequence. Replay serves them back in the same order using a per-URL offset cursor. No URL collision, no response mixing.
- **Clock abstraction.** All span timestamps come from `agent_trace.core.clock.get_time()`, not `time.time()`. During replay, `FixtureClock` is installed in place of `WallClock`. Span durations in replayed traces reflect the original execution times, not the time replay took.
- **"Deterministic" means inputs, not outputs.** During replay, each agent node receives the same inputs and the same tool responses it received during recording. The LLM itself is bypassed entirely — the recorded bytes are returned directly. There is no LLM output involved. If your agent's node logic is deterministic given the same inputs, the replay is deterministic.

---

## Benchmarks

| Metric | Value | Script |
|--------|-------|--------|
| Recording overhead per HTTP exchange | < 2 ms | `benchmarks/test_record_overhead.py` |
| Replay fidelity (response bytes identical to recorded) | 100% | `benchmarks/test_replay_fidelity.py` |
| P99 fixture write latency (SQLite WAL, local SSD) | < 5 ms | `benchmarks/test_fixture_write.py` |

Run all benchmarks:

```bash
uv run pytest benchmarks/ -v --benchmark-only
```

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

## Self-host

agent-trace emits OTLP spans. Run a local Jaeger instance to visualize trace trees:

```bash
docker compose up -d
```

The `docker-compose.yml` starts Jaeger with OTLP ingest on port 4317. Open [http://localhost:16686](http://localhost:16686) to browse traces.

Export to the local collector:

```python
from agent_trace.exporters.otlp import OTLPExporter

exporter = OTLPExporter(endpoint="http://localhost:4317")
exporter.export(trace)
```

---

## Use in CI

Record once locally or in a setup step. Commit the fixture (or cache it). Replay in every CI run:

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

Set `AGENT_TRACE_NETWORK_GUARD=1` in your CI environment (pytest does this automatically via `pyproject.toml` when you have the `env` key set). Any un-fixtured HTTP call will raise `NetworkGuardError` immediately instead of silently hitting a live endpoint.

---

## Community

- [GitHub Issues](https://github.com/RudrenduPaul/agent-trace/issues) — bug reports and feature requests
- [GitHub Discussions](https://github.com/RudrenduPaul/agent-trace/discussions) — questions and ideas
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to set up a dev environment and submit a PR
- Discord — link coming soon

---

## Security note

Fixture files at `~/.agent-trace/runs/` contain full HTTP request and response bodies. Those bodies may include your API keys, prompt contents, and user data. Add this to your `.gitignore` before committing anything:

```
~/.agent-trace/
.agent-trace/
*.fixture.db
```

Never commit a `fixture.db` file that was generated against a production API key. Use a separate key for recording, or scrub the fixture before committing (see `docs/concepts.md` for the SQLite schema).
