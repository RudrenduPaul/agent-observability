# OpenAI Agents SDK Integration

agent-trace intercepts HTTP at the transport layer so every call the OpenAI
Agents SDK makes — completions, tool calls, handoffs — is captured and can be
replayed offline at zero API cost.

---

## 1. Install

```bash
pip install agent-observability-trace-cli[openai-agents]
```

This installs `openai-agents>=0.0.3`. The Agents SDK is a separate package
from the `openai` Python library. Check `pyproject.toml` for the pinned range.

---

## 2. Quick start

```python
"""
OpenAI Agents SDK — record and replay.

Prerequisites:
    pip install agent-observability-trace-cli[openai-agents]
    export OPENAI_API_KEY=your-key  # needed only for the record step
"""
from __future__ import annotations
import os

from agents import Agent, Runner  # openai-agents package
from agent_trace import tracer, replay
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.core.trace import Trace
import json
from pathlib import Path


def build_agent() -> Agent:
    return Agent(
        name="Researcher",
        instructions="You answer questions concisely.",
        model="gpt-4o-mini",
    )


# --- Record (requires OPENAI_API_KEY) ---
with tracer.start_trace("openai-agents-demo", record=True) as trace:
    agent = build_agent()
    result = Runner.run_sync(agent, "What year did Python 3.10 release?")
    run_id = trace.run_id

print(f"Recorded run: {run_id}")
print(f"Final output: {result.final_output}")

# --- Replay (no API key needed) ---
with replay(run_id) as ctx:
    agent2 = build_agent()
    result2 = Runner.run_sync(agent2, "What year did Python 3.10 release?")

print(f"Replayed output: {result2.final_output}")
assert result.final_output == result2.final_output

# --- Show span tree ---
trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
trace_obj = Trace.from_dict(json.loads(trace_path.read_text()))
StdoutExporter().export(trace_obj)
```

The OpenAI Agents SDK uses `httpx` internally. agent-trace patches
`httpx.Client.__init__` before the SDK creates its client, so all
chat completion calls and tool calls are captured automatically.

---

## 3. What gets traced

| What | How |
|------|-----|
| Agent turn completions (each LLM call) | httpx transport interception |
| Tool call requests and responses | httpx transport interception |
| Handoffs between agents | httpx transport interception |
| Span boundaries | `tracer.span(...)` added manually or via agent hooks |

agent-trace v0.1 does not add automatic per-turn spans for the Agents SDK.
You get one flat recording of all HTTP exchanges. Per-turn span attribution
is planned as a named callback integration in v0.2.

---

## 4. Replaying a multi-agent conversation

If your run uses multiple agents in a handoff chain, all HTTP calls from all
agents in the chain are recorded in sequence_num order. Replay serves them
back in that same order.

```python
from agents import Agent, Runner
from agent_trace import tracer, replay

def build_pipeline() -> Agent:
    researcher = Agent(
        name="Researcher",
        instructions="Research the topic and pass to writer.",
        model="gpt-4o-mini",
    )
    writer = Agent(
        name="Writer",
        instructions="Write a short paragraph using the research.",
        model="gpt-4o-mini",
    )
    # Handoff: researcher transfers to writer after research
    researcher.handoffs = [writer]
    return researcher

# Record the full multi-agent conversation
with tracer.start_trace("multi-agent-run", record=True) as trace:
    agent = build_pipeline()
    result = Runner.run_sync(agent, "Explain what agent-trace does.")
    run_id = trace.run_id

print(f"Exchanges recorded: ", end="")
from agent_trace.replay.fixture import Fixture
from pathlib import Path
fix_path = Path.home() / ".agent-trace" / "runs" / run_id / "fixture.db"
with Fixture(fix_path) as f:
    print(f.exchange_count())

# Replay — both agent turns served from fixture
with replay(run_id) as ctx:
    agent2 = build_pipeline()
    result2 = Runner.run_sync(agent2, "Explain what agent-trace does.")

assert result.final_output == result2.final_output
```

The fixture stores exchanges in `sequence_num` order across all agents. Because
each `POST /v1/chat/completions` call is keyed by `(method, url)` in the replay
cursor, and that URL is the same for all LLM calls, they replay in the order
they were recorded — which is the correct order for a sequential handoff chain.

If your agents make parallel async calls (via `Runner.run()` instead of
`Runner.run_sync()`), see the async limitation in section 5.

---

## 5. Known limitations and SDK version requirements

- **SDK version:** `openai-agents>=0.0.3` is required. The package is under
  active development; breaking changes to the `Runner` API may occur. Check the
  pinned range in `pyproject.toml` before upgrading.

- **Async runner:** `Runner.run()` (async) uses `httpx.AsyncClient`. agent-trace
  v0.1 only patches `httpx.Client` (synchronous). Use `Runner.run_sync()` for
  fully fixtured replay. Async support is planned for v0.2.

- **Streaming tool calls:** The Agents SDK may use streaming for tool call
  responses. Streaming is recorded as a buffered single response. The replay
  returns it as a single response body, not as a stream. The SDK handles this
  gracefully in non-streaming mode; if you explicitly enable streaming in your
  runner config, behaviour during replay may differ from recording.

- **Token counts:** Token counts from `usage` fields in the recorded responses
  are replayed verbatim. Cost estimates based on replayed token counts will
  match the original run.

- **Parallel tool calls:** If the LLM issues multiple tool calls in one turn
  (parallel function calling), all tool response HTTP calls are recorded and
  replayed in the order they were made. The fixture cursor is keyed per URL,
  so if two different tool endpoints are called, their responses are tracked
  independently.

- **Replay cannot simulate a modified request:** `ReplayTransport`/
  `AsyncReplayTransport` only ever serve back the exact recorded
  `response_body` for a matching `(method, url)` — they never reconstruct
  what a request with different parameters would have returned. If you
  change `model_settings` (e.g. `reasoning_effort`, `verbosity`), swap
  models, or edit a tool schema and then replay the *old* fixture, you get
  the *old* response back, not a re-run against your change. This matters
  specifically for `openai-agents` because `model_settings` differences
  across model versions are a real failure mode (see the "model_settings"
  span attributes above) — validating a `model_settings` fix requires a
  fresh recording, not a replay of the pre-fix run. Record/replay is a tool
  for reproducing and debugging a *captured* run offline at zero API cost,
  not a substitute for re-running inference against a changed request.
