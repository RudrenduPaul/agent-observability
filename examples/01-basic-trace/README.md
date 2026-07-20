# Example 01 — Basic Trace

This is the simplest possible agent-trace example. It traces a pure Python
function — no LLM, no API calls, no credentials needed.

## What it demonstrates

- `tracer.start_trace("name")` — opens a trace and writes `trace.json` on exit
- `tracer.span("name")` — opens a child span; sets attributes on it
- `Span.set_attribute(key, value)` — attaches a key/value pair to a span
- `StdoutExporter().export(trace)` — prints a colored span tree to the terminal
- Loading a saved `trace.json` from disk with `Trace.from_dict()`

## How to run

```bash
# From the repo root:
uv run python examples/01-basic-trace/example.py

# Or with plain Python:
pip install agent-observability-trace-cli
python examples/01-basic-trace/example.py
```

## What the output looks like

```
Processing document...

--- Span tree ---
Trace: process_document  [run_a1b2c3d4e5f6]
├── extract-entities  OK  (51.2 ms)
├── summarize         OK  (82.4 ms)
└── score             OK  (30.8 ms)

Trace saved to: /Users/you/.agent-trace/runs/run_a1b2c3d4e5f6

--- Result ---
Entities: ['agent-trace', 'observability', 'library', 'agents', 'records']
Summary:  agent-trace is an observability library for AI agents.
Score:    0.500
```

The span tree is color-coded when `rich` is installed (bundled with agent-trace):
green for `OK`, red for `ERROR`, yellow for `UNSET`.

## What gets saved to disk

After the run, two files are written to `~/.agent-trace/runs/run_<id>/`:

- `trace.json` — the full span tree with start/end times, attributes, and status
- No `fixture.db` — this example does not use `record=True`, so no HTTP
  fixture is written

To also save a fixture (useful if you add real HTTP calls):

```python
with tracer.start_trace("process_document", record=True) as trace:
    ...
```
