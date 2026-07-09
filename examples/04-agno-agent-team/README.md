# Example 04 — Agno Agent/Team

Demonstrates `agent_trace.integrations.agno` capturing an Agno
(`agno-agi/agno`) run. Runs with **no API key** by default — a tiny scripted
fake model stands in for a real provider so this example is deterministic
and reproducible in CI; pass `--live` to use a real OpenAI model instead.

## What it demonstrates

1. **Solo `Agent` run with a tool call** — one `agent:` span, two `llm:`
   spans (before/after the tool call), one `tool:` span, all correctly
   parented and closed.
2. **`Team` run that delegates to a member `Agent`** — the member's own run
   span (`agent:researcher`) is a *child* of the team's run span
   (`team:research-team`), and the `tool:delegate_task_to_member` span
   carries `agno.child_run_id` correlating the delegation with the member's
   own run. This is the per-team-member attribution [redacted] #5326
   asked for — without it, a multi-agent `Team` failure is indistinguishable
   raw HTTP traffic with no routing context.
3. **An in-process exception that never reaches the HTTP layer** — a model
   that raises entirely inside its own code (the exact shape of backlog
   issue #5298's `UnboundLocalError` inside `agno/models/base.py`). Agno's
   own streaming loop catches it and re-surfaces it as a `RunErrorEvent`,
   which `AgnoTracer` turns into an `ERROR` span with the exception message
   attached — something the framework-agnostic HTTP interceptor alone could
   never see, since no HTTP call was ever made.

## How to run

```bash
# From the repo root:
pip install agent-trace[agno]
python examples/04-agno-agent-team/example.py

# Or against a real OpenAI model:
export OPENAI_API_KEY=your-key
pip install agent-trace[agno] openai
python examples/04-agno-agent-team/example.py --live
```

## Usage patterns shown

Convenience wrapper (drains the event stream automatically, returns the
final `RunOutput`/`TeamRunOutput`):

```python
from agent_trace import Tracer
from agent_trace.integrations.agno import instrument_agent_arun

t = Tracer()
with t.start_trace("my_run", record=True) as trace:
    result = await instrument_agent_arun(agent, "hello", tracer=t, trace=trace)
```

Hook-based (for when you're already consuming the event stream yourself,
e.g. to stream partial output to a UI):

```python
from agent_trace.integrations.agno import AgnoTracer

hook = AgnoTracer(tracer=t, trace=trace)
async for event in agent.arun("hello", stream=True, stream_events=True):
    hook.process_event(event)
    # ... also do whatever you were already doing with `event` ...
```
