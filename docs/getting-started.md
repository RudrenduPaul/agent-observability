# Getting Started

This guide walks from zero to a working record/replay in about 10 minutes.

---

## 1. Install

```bash
pip install agent-trace
```

or with uv:

```bash
uv add agent-trace
```

Optional extras for specific frameworks:

```bash
pip install agent-trace[langgraph]        # LangGraph callback integration
pip install agent-trace[openai-agents]    # OpenAI Agents SDK integration
pip install agent-trace[otlp]             # OTLP exporter for Jaeger / Grafana Tempo
pip install agent-trace[requests]         # requests adapter (httpx is bundled by default)
```

Verify the install:

```bash
agent-trace version
# agent-trace 0.1.0
```

---

## 2. Your first trace — decorator approach

The decorator approach is the fastest way to start. Wrap any function and every
HTTP call made inside it is captured:

```python
from agent_trace import tracer

@tracer.instrument(record=True)
def summarize(text: str) -> str:
    # Any httpx or requests calls inside here are recorded.
    import httpx
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Summarize: {text}"}],
        },
    )
    return response.json()["choices"][0]["message"]["content"]

result = summarize("agent-trace is a record/replay tool for AI agents.")
print(result)
# Run directory printed to stderr: ~/.agent-trace/runs/run_<id>/
```

After the call, two files are written to `~/.agent-trace/runs/run_<id>/`:

- `trace.json` — span tree with timings and attributes
- `fixture.db` — SQLite database containing every HTTP exchange

---

## 3. Your first trace — context manager approach

If you need more control — multiple agents in one trace, custom run IDs, or
conditional recording — use the context manager directly:

```python
from agent_trace import tracer

with tracer.start_trace("document-pipeline", record=True) as trace:
    with tracer.span("fetch") as span:
        span.set_attribute("url", "https://example.com/doc.pdf")
        # ... fetch the document ...

    with tracer.span("extract-entities") as span:
        span.set_attribute("entity_count", 42)
        # ... call your LLM ...

    with tracer.span("write-output") as span:
        # ... write results ...
        pass

# trace.run_id is the directory name you need for replay
print(f"Run ID: {trace.run_id}")
```

You can also open manual spans without a context manager for lower-level control:

```python
span = tracer.start_span("my-op")
try:
    # ... do work ...
    span.end()
except Exception as exc:
    span.record_exception(exc)
    raise
```

---

## 4. Viewing the trace — stdout exporter

The quickest way to see what happened is the built-in stdout exporter, which
prints a colored span tree (requires `rich`, which is bundled):

```python
import json
from pathlib import Path
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter

run_id = "run_abc123def456"  # from the output above
trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
trace = Trace.from_dict(json.loads(trace_path.read_text()))

StdoutExporter().export(trace)
```

Or use the CLI directly:

```bash
agent-trace show run_abc123def456
agent-trace list
```

---

## 5. Your first replay — step by step

1. **Find the run ID** from the previous recording step (it's also visible in
   `agent-trace list`).

2. **Replay**:

```python
from agent_trace import replay

run_id = "run_abc123def456"

with replay(run_id) as ctx:
    # The same function call — no API key needed, no network, no cost.
    result = summarize("agent-trace is a record/replay tool for AI agents.")
    print(result)
    # result is identical to the original because the same response bytes
    # are served from fixture.db.

print(f"Fixture had {ctx.fixture.exchange_count()} recorded exchanges")
```

3. **Check the span tree**:

```bash
agent-trace show run_abc123def456
```

You will see the same span tree as the original run, with timing values from
the original execution (the `FixtureClock` restores recorded timestamps).

---

## 6. Using in tests — pytest example

Record once, replay in every test run:

```python
# tests/test_summarize.py
import pytest
from pathlib import Path
from agent_trace import replay

FIXTURE_PATH = Path("fixtures/summarize_run.db")

@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="No fixture. Run: python scripts/record.py"
)
def test_summarize_short_text():
    with replay(FIXTURE_PATH) as ctx:
        from myapp.agents import summarize
        result = summarize("agent-trace is a record/replay tool.")

    assert "record" in result.lower() or "replay" in result.lower()
    assert ctx.fixture.exchange_count() == 1  # exactly one LLM call


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="No fixture. Run: python scripts/record.py"
)
def test_summarize_returns_string():
    with replay(FIXTURE_PATH) as ctx:
        from myapp.agents import summarize
        result = summarize("agent-trace is a record/replay tool.")

    assert isinstance(result, str)
    assert len(result) > 0
```

Set `AGENT_TRACE_NETWORK_GUARD=1` in your CI environment so any un-fixtured
request raises `NetworkGuardError` immediately. If you use pytest and your
`pyproject.toml` already contains:

```toml
[tool.pytest.ini_options]
env = ["AGENT_TRACE_NETWORK_GUARD=1"]
```

then the guard is always active in test runs.

---

## 7. LangGraph integration — full working example

Install the extra:

```bash
pip install agent-trace[langgraph]
```

```python
from agent_trace import tracer, replay
from agent_trace.integrations.langgraph import LangGraphTracer

from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    question: str
    answer: str

def research_node(state: State) -> State:
    # This makes a real HTTP call to the LLM during recording.
    # During replay, the call is served from fixture.db.
    import httpx
    resp = httpx.post("https://api.openai.com/v1/chat/completions", ...)
    return {"answer": resp.json()["choices"][0]["message"]["content"]}

def respond_node(state: State) -> State:
    return {"answer": f"Based on research: {state['answer']}"}

graph = (
    StateGraph(State)
    .add_node("research", research_node)
    .add_node("respond", respond_node)
    .add_edge("research", "respond")
    .add_edge("respond", END)
    .set_entry_point("research")
    .compile()
)

# Record
with tracer.start_trace("langgraph-demo", record=True) as trace:
    result = graph.invoke({"question": "What is agent-trace?"})
    print(f"Recorded run_id: {trace.run_id}")

# Replay — no API calls
with replay(trace.run_id) as ctx:
    result2 = graph.invoke({"question": "What is agent-trace?"})
    assert result == result2
```

For full details see [docs/integrations/langgraph.md](integrations/langgraph.md).

---

## 8. Environment variables reference

| Variable | Default | Effect |
|----------|---------|--------|
| `AGENT_TRACE_NETWORK_GUARD` | `0` | Set to `1` to raise `NetworkGuardError` when a replay transport receives a request with no matching fixture entry. Always set this in CI. |
| `AGENT_TRACE_TRACE_DIR` | `~/.agent-trace/runs` | Override the directory where run directories and fixture files are stored. Useful for CI caches or shared network paths. |

Example:

```bash
# CI environment
export AGENT_TRACE_NETWORK_GUARD=1
export AGENT_TRACE_TRACE_DIR=/tmp/agent-trace-ci

pytest tests/ -v
```

---

## 9. Troubleshooting

### `NetworkGuardError` during replay

**What it means:** The replay transport received an outbound HTTP request, looked
it up in `fixture.db`, found no matching entry, and raised `NetworkGuardError`
because `AGENT_TRACE_NETWORK_GUARD=1` is set.

**Common causes:**

1. The function under test makes a request that was not captured during recording
   because it was called outside the `with tracer.start_trace(..., record=True)`
   block.
2. The URL or method differs between the recording and replay runs (e.g., a
   timestamp or session ID in the URL changed).
3. The fixture was recorded against a different code path than the one being
   replayed.

**Fix:** Re-record with `record=True` and verify the URL pattern is stable.

---

### Fixture not found (`FileNotFoundError`)

```
FileNotFoundError: No fixture.db found at ~/.agent-trace/runs/run_xyz/fixture.db.
Did you record this run with record=True?
```

**Check:**

1. The `run_id` spelling — it must match exactly, including the `run_` prefix.
   Run `agent-trace list` to see all recorded runs.
2. That the recording step used `record=True`. Without that flag, no `fixture.db`
   is written (only `trace.json`).
3. That `AGENT_TRACE_TRACE_DIR` is set to the same value in both the recording
   and replay environments if you overrode the default.

---

### Missing exchange in replay (LLM call not intercepted)

**Symptom:** The replay makes a live HTTP call (or raises `NetworkGuardError` for
a URL you expected to be in the fixture).

**Most common cause:** The AI SDK you are using creates an `httpx.AsyncClient`
(async, not sync) or uses a transport that was instantiated before the recording
context was entered. The agent-trace recording patch only applies to clients
created *inside* the `start_trace` context.

**Fix:** Construct your SDK client inside the recording block, or use
`with tracer.start_trace(..., record=True)` as the outermost context manager
before any SDK objects are created.

If your SDK uses `requests` and you are seeing misses, verify you installed the
requests extra:

```bash
pip install agent-trace[requests]
```
