# Example 07 — `graph.invoke(stream_mode=...)` vs `graph.stream(...)` Delivery Timing (#4653)

Reproduces the exact distinction behind
[langgraph#4653](https://github.com/langchain-ai/langgraph/issues/4653)
(and its "same issue" commenters `JasonChen280`/`Kigstn`): passing
`stream_mode=` to `graph.invoke(...)` does **not** make it deliver output
progressively. LangGraph's own `Pregel.invoke()` fully drains its own
internal `self.stream(...)` generator in a loop before returning anything
to the caller — `stream_mode` only changes the *shape* of the final
returned value, not *when* the caller gets it. `graph.stream(...)` is the
only call that actually yields chunks as they happen.

No API key required — both nodes are plain Python stand-ins with a small,
deliberate `time.sleep()` so the timing difference is directly observable.

## The distinction

```python
# Looks like it should stream progressively — it does not.
result = graph.invoke(state, stream_mode="updates")
# `result` only exists once the ENTIRE run has finished. Nothing was
# observable to the caller before that single moment.

# This is the only one that actually yields progressively:
for chunk in graph.stream(state, stream_mode="updates"):
    ...  # each chunk arrives right after its node finishes
```

## What this demonstrates about agent-trace's capture

1. **`graph.invoke(...)` produces no `graph:stream` span at all.** A
   callback-only trace (`on_chain_start`/`on_chain_end`/...) looks
   structurally identical whether `invoke()` or `stream()` was used —
   nothing in `LangGraphTracer`'s existing callback hooks fires on "a value
   was yielded to the caller's own code," because that boundary isn't a
   LangChain callback event at all.
2. **`traced_stream()` (`src/agent_trace/integrations/langgraph.py`) makes
   the difference visible.** Wrapping `graph.stream(...)` in
   `traced_stream()` opens a dedicated `graph:stream` span carrying one
   `stream_yield` SpanEvent per chunk, timestamped on the same clock as
   every other span, at the exact moment the caller's own `for` loop
   receives it — not when the underlying node finished internally.
3. **The two traces are now genuinely different, not just the printed
   output.** `invoke-mode`'s trace has zero `graph:stream` spans;
   `stream-mode`'s trace has one, with per-chunk arrival timestamps a
   developer can read directly off `trace.json` to answer "was my code
   actually get called progressively, or did I just think it did?" without
   reading LangGraph's `Pregel` internals.

## Run

```bash
python examples/07-langgraph-invoke-vs-stream-timing/example.py
```

Expected output (abridged — exact timings vary slightly run to run):

```
--- graph.invoke(stream_mode='updates') ---
  [ 0.610s] caller received the ENTIRE result at once, after both nodes had already finished: [...]
  (nothing was observable to the caller before this single moment — invoke() blocks until the whole run completes)

--- graph.stream(stream_mode='updates'), wrapped in traced_stream() ---
  [ 0.306s] caller received a chunk progressively: {'step_one': {'steps': ['step_one_done']}}
  [ 0.610s] caller received a chunk progressively: {'step_two': {'steps': ['step_one_done', 'step_two_done']}}

--- What each trace's span tree shows ---
invoke-mode trace:  0 'graph:stream' span(s) (none — invoke() never yields progressively, so there's nothing for traced_stream() to wrap)
stream-mode trace:  1 'graph:stream' span(s), carrying 2 stream_yield event(s) — one per chunk, timestamped at the moment the caller's own for-loop actually received it

stream_yield event timestamps (relative to span start, seconds):
  index=0  +0.306s
  index=1  +0.610s
```

## See also

- `src/agent_trace/integrations/langgraph.py` — `traced_stream()`/`traced_astream()`
- `docs/integrations/langgraph.md` — full LangGraph integration guide
