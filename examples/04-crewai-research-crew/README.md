# Example 04 — crewAI Research Crew

This example shows the `CrewAITracer` integration: a two-agent, two-task
sequential crew (researcher → writer) instrumented with agent-trace, recorded
once against the real OpenAI API, then replayed offline at zero API cost.

## Prerequisites

```bash
pip install agent-observability-trace-cli[crewai]
export OPENAI_API_KEY=your-key   # needed only for the record step
```

## The crew

Two agents, sequential process:

```
Researcher (gathers 3 facts) → Writer (summarizes in 2 sentences)
```

`CrewAITracer` subscribes to crewAI's global event bus
(`crewai.events.crewai_event_bus`) and emits an agent-trace span for the crew
kickoff, each agent execution, each task, each LLM call, and each tool call —
correctly nested, using crewAI's own event-scope bookkeeping (see
`src/agent_trace/integrations/crewai.py` for how span pairing works).

Both agents use `llm="gpt-4o-mini"`, which resolves to crewAI's native
`OpenAICompletion` class — talking to OpenAI over `httpx.Client`, the same
transport agent-trace's generic HTTP interceptor already patches. So
record/replay works exactly like the LangGraph examples: no crewAI-specific
fixture handling needed.

## Step 1 — Record

```bash
python examples/04-crewai-research-crew/example.py record --topic "the CAP theorem"
```

This runs the crew against the real OpenAI API, captures every HTTP exchange
into `~/.agent-trace/runs/<run_id>/fixture.db`, and prints the resulting span
tree (crew → task → agent → llm, nested).

## Step 2 — Replay

```bash
python examples/04-crewai-research-crew/example.py replay <run_id>
```

Re-runs the same crew with no `OPENAI_API_KEY` needed — every HTTP response is
served from the recorded fixture. `AGENT_TRACE_NETWORK_GUARD` blocks any
accidental live call during replay.

## What this demonstrates

- A crewAI-native integration (no manual `httpx`/`requests` interceptor
  wiring).
- Correct span nesting (crew → task → agent → llm) derived entirely from
  crewAI's own event-scope stack, not a hand-rolled run-id.
- Errors (e.g. an invalid API key) are captured as `ERROR`-status spans with
  the real exception text attached — confirmed by running this example with a
  deliberately invalid key during development.
