# Example 02 — LangGraph Failure Replay

This example shows the core agent-trace workflow: record a multi-step
LangGraph run against real APIs, then replay it offline to debug failures
without spending more tokens.

## Prerequisites

```bash
pip install agent-observability-trace-cli[langgraph]
export OPENAI_API_KEY=your-key   # needed only for the record step
```

## The graph

A 3-node `StateGraph`:

```
[research] → [analyze] → [respond] → END
```

- `research` — asks the LLM to gather notes on the question
- `analyze` — extracts 3 key insights from the notes
- `respond` — writes a two-sentence final answer

Each node makes one HTTP call to `POST /v1/chat/completions`. The recording
captures all 3 calls. The replay serves all 3 from `fixture.db`.

## Step 1 — Record

```bash
python examples/02-langgraph-failure-replay/example.py record
```

Optional custom question:

```bash
python examples/02-langgraph-failure-replay/example.py record \
  --question "Why is Python's GIL being removed?"
```

Output includes:

```
Recording run for: 'What is the difference between LangGraph and LangChain?'
(This makes real API calls. Ensure OPENAI_API_KEY is set.)

Research notes: LangGraph is a library built on top of LangChain...
Analysis:       - LangGraph adds stateful, cyclic graph execution...
Response:       LangGraph extends LangChain with explicit state machines...

Run ID: run_a1b2c3d4e5f6

Trace: langgraph-failure-replay  [run_a1b2c3d4e5f6]
├── research  OK  (821.3 ms)
├── analyze   OK  (654.1 ms)
└── respond   OK  (442.7 ms)

Replay with:
  python examples/02-langgraph-failure-replay/example.py replay run_a1b2c3d4e5f6
```

The run ID is printed at the end. Use it in the replay step.

## Step 2 — Replay

```bash
python examples/02-langgraph-failure-replay/example.py replay run_a1b2c3d4e5f6
```

Output:

```
Replaying run: run_a1b2c3d4e5f6
(No API calls will be made — all responses served from fixture.db)

Fixture has 3 recorded HTTP exchange(s).
Question: 'What is the difference between LangGraph and LangChain?'

Research notes: LangGraph is a library built on top of LangChain...
Analysis:       - LangGraph adds stateful, cyclic graph execution...
Response:       LangGraph extends LangChain with explicit state machines...

Exchanges consumed: 3
```

The replayed response is byte-for-byte identical to the recorded response.
Replay typically completes in under 100 ms regardless of the original run time.

## Debugging a failure

If the original run failed (e.g., a `ValueError` in `analyze_node` when the
LLM returned malformed JSON), the failure reproduces in replay because the
same malformed bytes are served from the fixture:

```bash
# Replay the failing run
python examples/02-langgraph-failure-replay/example.py replay run_failed123

# Output:
# Replay reproduced the same failure: ValueError: ...
```

Now add a `pdb.set_trace()` or extra logging inside `analyze_node` and replay
again. No tokens are spent on each debug iteration.

## Files written after the record step

```
~/.agent-trace/runs/run_<id>/
  fixture.db    — SQLite database with 3 HTTP exchanges
  trace.json    — span tree with timings
```

## See also

- `docs/concepts.md` — how the fixture and replay engine work
- `docs/integrations/langgraph.md` — full LangGraph integration guide
- `agent-trace list` — see all recorded runs
