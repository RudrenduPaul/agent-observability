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

---

## 2026-06-19 — Path traversal in run_id inputs

Pattern: User-supplied run_id values (e.g. "../../etc/passwd") were passed directly to
Path / operations without sanitization. This allowed callers to read or write files
outside the intended trace directory.

Rule: Any user-controlled path component must be validated with
`resolved_candidate.relative_to(resolved_base)` before any filesystem access.
Both the base and candidate must be `.resolve()`d first to normalize symlinks and
`..` components. Applies to start_trace(), replay(), and _cli._run_dir().

Anti-sycophancy check: Not flagged until engineering audit in session 2. Should have
been caught at initial design — any file path derived from user input is a traversal risk.

---

## 2026-06-19 — FixtureClock must not start at 0.0

Pattern: FixtureClock() initialized _current to 0.0, giving replay spans the Unix
epoch as their timestamp. This made replayed traces appear to have been recorded on
1970-01-01 and broke any downstream tool that rendered absolute timestamps.

Rule: FixtureClock() defaults to time.time() at construction. Use FixtureClock(initial=0.0)
when a zero base is explicitly required (e.g. duration-only tests). Never silently assign
epoch-zero to a span timestamp.

Anti-sycophancy check: Flagged during engineering audit — the 0.0 default was never
questioned at design time because it made unit tests easy to write.

---

## 2026-06-19 — instrument() must detect async functions before wrapping

Pattern: @tracer.instrument() returned a sync wrapper for all functions. When called on
an async def, the wrapper returned a coroutine object instead of awaiting it, silently
discarding the result and producing no spans.

Rule: Use inspect.iscoroutinefunction(fn) before creating the wrapper. If True, define
an `async def async_wrapper` that awaits fn(*args, **kwargs). Never use
asyncio.iscoroutinefunction — it is deprecated in Python 3.16+.

Anti-sycophancy check: Not caught at design time. Surfaced in engineering audit because
async agents are the primary use case for newer AI SDKs.

---

## 2026-06-19 — Concurrent start_trace(record=True) overwrites saved transport

Pattern: A second call to start_trace(record=True) while the first was still active
overwrote self._original_httpx_init with the already-patched version. When the outer
trace exited it restored the patched method, leaving httpx permanently patched.

Rule: Track nesting depth with a counter (_transport_depth). Only install the recording
transport when depth transitions from 0 to 1. Only uninstall when depth returns to 0.
Any intermediate call just increments/decrements the counter.

Anti-sycophancy check: Flagged proactively during engineering audit as a known race
condition class. Reentrancy bugs in context managers are easy to miss without explicit
depth tracking.

---

## 2026-06-19 — LangGraph __bases__ mutation is forbidden in Python 3.14+

Pattern: _get_tracer_class() was using cls.__bases__ = (BaseCallbackHandler,) to inject
a base class at instantiation time. This mutates the class object in place, is not thread
safe, and raises TypeError in Python 3.14+ where class layout is locked after definition.

Rule: Build the concrete class once with BaseCallbackHandler as a genuine base (inside
a closure, using a threading.Lock for double-checked locking). Store it in a module-level
singleton. Subsequent calls return the cached class without any mutation.

Anti-sycophancy check: The __bases__ mutation was in the original implementation. The
Python 3.14 breakage was flagged proactively during the engineering audit before CI ran
against 3.14.

---

## 2026-06-19 — GitHub Actions @master pins are supply-chain risks

Pattern: trivy-action@master and similar actions without a pinned version or SHA mean
any malicious commit to the upstream action repo runs in CI with full secrets access.

Rule: Pin every third-party GitHub Action to a specific version tag (e.g. @v0.30.0),
never @master or @latest. For actions where the maintainer is not fully trusted, pin
to a commit SHA instead. Audit all .github/workflows/*.yml files before every major
release.

Anti-sycophancy check: Flagged in engineering audit. The original ci.yml template in
the planning doc also used @master — the error was in the source spec, not just the
implementation.
