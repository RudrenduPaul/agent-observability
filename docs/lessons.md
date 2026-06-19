# Lessons — agent-trace

Append to this file after any correction or non-obvious architectural decision.
Read this file at the start of every session before writing code.

## Format

```
## YYYY-MM-DD — <short title>
Pattern: what went wrong or was non-obvious
Rule: the rule that prevents recurrence
Anti-sycophancy check: was this flagged proactively or only after correction?
```

---

## 2026-06-19 — Clock abstraction is the replay invariant

Pattern: Calling time.time() directly in span creation code breaks replay because
the replay clock (FixtureClock) is only consulted via get_time() in core/clock.py.
Any direct time.time() call bypasses the clock swap and produces wall-clock timestamps
during replay, making span timing non-deterministic.

Rule: All timestamp generation in src/ must call get_time() from agent_trace.core.clock.
Grep for "time.time()" in src/ before every commit — it must return 0 results.
The only permitted exception is fixture.py's record_exchange(), which records wall time
for the fixture metadata (not for span timestamps).

Anti-sycophancy check: Flagged proactively in architecture design before any code was written.

---

## 2026-06-19 — Fixture portability requires JSON-primitive-only serialization

Pattern: Storing Python datetime objects in fixture files makes them unreadable
across Python versions and outside Python entirely. Enums stored as Python objects
(not their .value) fail to deserialize from JSON without custom decoders.

Rule: Span.to_dict() and Fixture exchange storage must produce dicts containing
ONLY: str, int, float, bool, None, list, dict. No datetime objects (use float
Unix timestamps). Enums by .value only. No sets (not JSON-serializable).

Anti-sycophancy check: Flagged proactively in CLAUDE.md before any serialization code was written.
