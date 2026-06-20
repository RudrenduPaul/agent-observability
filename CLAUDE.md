# CLAUDE.md — agent-observability

## Git Workflow

When asked to commit, push, or "update GitHub" — just do it. No questions.

- `git add` relevant files → `git commit` → `git push origin main` in one shot
- "commit" means commit AND push — always both, never just one
- Never ask "should I push?" — just execute

Every commit message must end with:

```
Built by Rudrendu Paul, developed with Claude Code
```

Never use `Co-Authored-By:` lines.

---

## Project Overview

**agent-trace** is a Python-native agent observability library. The core technical angle is
deterministic record/replay: capture a real agent run once, replay it offline with zero API
calls to reproduce any failure exactly.

The mechanism: HTTP transport interception (httpx + requests layer). During a record pass,
every LLM request/response pair is serialized to a local SQLite fixture. During a replay
pass, the transport is swapped for a local cache reader. No network calls. No token spend.
No LLM nondeterminism.

What "deterministic" means here and what it does not:
- DOES mean: same fixture → same inputs to each agent node + same tool responses.
- DOES NOT mean: the LLM produces the same output. During replay we bypass the LLM entirely
  by serving the recorded response from cache.
- Never conflate these in documentation or code comments.

License: Apache 2.0 (core) + proprietary enterprise tier (hosted trace history, dashboards, SSO).
Primary integrations: LangGraph (P0), OpenAI Agents SDK (P0).

---

## Repo Layout

```
src/agent_trace/
├── core/         ← Span, Trace, clock.py (THE critical file — read before touching spans)
├── interceptor/  ← httpx hook + requests adapter (the transport interception layer)
├── _replay/      ← replay engine + SQLite fixture reader/writer
├── exporters/    ← OTLP, file, stdout
└── integrations/ ← LangGraph + OpenAI Agents SDK (lazy-imported, optional deps)
tests/
├── unit/         ← no network calls
└── integration/  ← real APIs only, tagged @pytest.mark.integration
benchmarks/       ← run before claiming any performance number
docs/lessons.md   ← read at session start before writing any code
```

---

## Engineering Standards (run before every commit)

```bash
uv run ruff check src/ tests/ benchmarks/
uv run ruff format --check src/ tests/ benchmarks/
uv run mypy src/ --strict
uv run pytest tests/unit/ --cov=src/ --cov-fail-under=80
```

Integration tests (real APIs, opt-in):
```bash
uv run pytest tests/integration/ -m integration
```

No mocking in integration tests. If a test needs a real LangGraph run or a real OpenAI
Agents SDK call, make it. If you cannot, write a unit test instead and say so explicitly.

Test coverage targets:
- Overall: 80% minimum (enforced in CI)
- src/agent_trace/_replay/: 90% minimum (correctness-critical)
- src/agent_trace/interceptor/: 90% minimum

---

## Plan Mode

Enter plan mode for any task that:
- Touches 2+ files
- Changes the Span or Trace data model
- Adds or removes anything from the public API (src/agent_trace/__init__.py)
- Modifies the replay engine or the clock abstraction
- Adds a new integration

Write a plan first. If something goes wrong mid-task, stop and re-plan.

---

## Anti-Sycophancy Checklist (always active)

1. Counter-evidence first. When evaluating any design, name the failure mode before the
   benefit. The replay engine fails silently if an external API call is not captured during
   record mode. Say that first.

2. No performance claims without a benchmark command. Never write "low overhead" or
   "sub-millisecond tracing" unless you have run `uv run pytest benchmarks/` this session
   and can show the output.

3. OSS stars are not revenue. Stars validate the problem. They do not validate willingness to
   pay for the hosted tier. Trace history data gravity requires months of user retention in a
   hosted environment, not a GitHub star.

4. Name the forkability risk. The Apache 2.0 core can be forked. Laminar can ship
   transport-interception replay in 4 to 8 weeks. Any moat discussion must acknowledge this.

5. Scope "deterministic" correctly. Same fixture → same span tree, same inputs to each node,
   same tool responses. Not same LLM output — we bypass the LLM. Never conflate these.

6. Integration maintenance burden is real. LangGraph and OpenAI Agents SDK change APIs.
   Before claiming "full support," check the pinned version in pyproject.toml and confirm
   the integration test is passing in CI.

---

## Key Invariants — Never Break These

- Span.start_time and Span.end_time use core/clock.py, never time.time() directly.
  Breaking this breaks replay determinism.
- Trace fixtures are JSON-serializable: no datetime objects, enums by value only, no sets.
- The replay engine makes zero network calls. AGENT_TRACE_NETWORK_GUARD=1 causes any
  network attempt during replay to raise immediately.
- All public API is in __init__.py. Users import from agent_trace, never from internals.
- mypy src/ --strict must pass. No Any escapes without a # type: ignore comment explaining why.
- Never commit a fixture.db generated against a production API key.

---

## Session Start Checklist

1. git status && git log --oneline -5
2. uv run pytest tests/unit/ -q
3. Read docs/lessons.md
4. If a bug is reported: write a failing test first, then fix it.

---

## Lessons File

After any correction or non-obvious decision, append to docs/lessons.md:

```
## YYYY-MM-DD — <short title>
Pattern: what went wrong or what was non-obvious
Rule: the rule that prevents recurrence
Anti-sycophancy check: was this flagged proactively or only after correction?
```
