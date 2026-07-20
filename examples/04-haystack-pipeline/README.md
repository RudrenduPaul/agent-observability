# Example 04 — Haystack Pipeline

Traces a Haystack 2.x `Pipeline.run()` with agent-trace. No LLM API calls,
no credentials — every component in this pipeline is pure Python.

## What it demonstrates

- Wiring `HaystackTracer` into Haystack's own native instrumentation surface
  via `haystack.tracing.enable_tracing(...)` — Haystack has no callback list
  to pass in (unlike LangGraph), so registration is global for the lifetime
  of the `with tracer.start_trace(...)` block.
- One `haystack.pipeline.run` span for the whole pipeline, and one
  `haystack.component.run` span per component, correctly parented.
- Component-level tags: `haystack.component.name`, `haystack.component.type`,
  and I/O socket spec, captured automatically by Haystack's own tracer
  regardless of content-tracing settings.
- Opt-in raw content capture: set `HAYSTACK_CONTENT_TRACING_ENABLED=1` to
  additionally capture each component's *actual received arguments* and
  *actual returned output* (`haystack.component.input` /
  `haystack.component.output`) — gated by Haystack itself, off by default
  since pipeline content can carry arbitrary user data.

This is the exact capability gap
[issue #4574](https://github.com/deepset-ai/haystack/issues/4574) exposes: a
`params` dict passed into one component not reaching the component it was
intended for is an in-process Python argument-propagation bug. It happens
before, and independent of, any HTTP request — agent-trace's `httpx`/
`requests` interceptor is structurally the wrong layer to catch it. Only a
framework-level hook capturing component inputs/outputs at the point of the
call (what this example wires up) can surface that class of bug.

## How to run

```bash
# From the repo root:
uv run python examples/04-haystack-pipeline/example.py

# Or with plain Python:
pip install "agent-observability-trace-cli[haystack]"
python examples/04-haystack-pipeline/example.py

# With raw component input/output content also captured on the spans:
HAYSTACK_CONTENT_TRACING_ENABLED=1 uv run python examples/04-haystack-pipeline/example.py
```

## What the output looks like

```
Running Haystack pipeline...

--- Span tree ---
Trace: haystack_pipeline_demo  run_46ab4f98f13c  ((0.3 ms total))
└── haystack.pipeline.run  OK  (0.3 ms)
    ├── haystack.component.run  OK  (0.0 ms)
    ├── haystack.component.run  OK  (0.0 ms)
    └── haystack.component.run  OK  (0.0 ms)

Trace saved to: /Users/you/.agent-trace/runs/run_46ab4f98f13c
Spans captured: 4  (errors: 0)

--- Result ---
Best chunk: {'text': 'agent makes and replays it offline without making API calls, ...', 'score': 2, 'keywords': ['agent', 'debugging']}
```

## What gets saved to disk

After the run, `trace.json` is written to `~/.agent-trace/runs/run_<id>/`
with the full span tree, including (when content tracing is enabled) the
real arguments each component received and returned.
