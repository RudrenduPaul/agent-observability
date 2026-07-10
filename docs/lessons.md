# Lessons — agent-observability

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

## 2026-07-05 — setdefault() transport patching silently loses to caller-supplied transports

Pattern: `_patch_httpx` in `agent_trace/__init__.py` used `kwargs.setdefault("transport", RecordingTransport(fixture))` to inject the recorder into every `httpx.Client`/`AsyncClient` built during `start_trace(record=True)`. `setdefault` only applies when the `transport` key is absent. `langchain-openai` (a P0 integration target) always constructs its own `httpx.HTTPTransport(socket_options=..., limits=...)` for kernel-level TCP keepalive tuning and passes it explicitly on every platform (Linux, macOS, Windows) unless `LANGCHAIN_OPENAI_TCP_KEEPALIVE=0` is set. Confirmed by live reproduction: recording a `ChatOpenAI` call captured 0 exchanges with the bug present, 1 exchange after setting that env var — the LLM call itself succeeded normally either way, with no error or warning surfaced to the caller.

Rule: transport-patching code must always wrap whatever transport the caller supplies (`RecordingTransport(fixture, inner=kwargs.get("transport"))`), never `setdefault`. `RecordingTransport`/`AsyncRecordingTransport` already accept an `inner` param for exactly this. Any future patch point that injects a transport/adapter into a third-party HTTP client must assume the callee may pass its own transport explicitly and chain around it, not merely default around it. Known residual gap: `mounts=`-based transport injection (e.g. `langchain_openai`'s `openai_proxy` code path, which sets `mounts={"all://": transport}` instead of `transport=`) is not wrapped by this fix and remains unrecorded — not yet reproduced or fixed.

Anti-sycophancy check: Found only after being asked to build a diagnostic reproduction for an unrelated LangGraph issue — the recording silently returning 0 exchanges could easily have been misread as "nothing to record" rather than "recording is broken." Not caught by existing unit tests before this session; two regression tests now cover it (`test_explicit_transport_is_wrapped_not_bypassed`, `test_explicit_async_transport_is_wrapped_not_bypassed` in `tests/unit/test_tracer.py`).

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

---

## 2026-06-19 — trace_id and run_id must be independent UUIDs

Pattern: Tracer.start_trace() set both trace_id and run_id to the same value
("run_abc123def456"). Since "run_..." is not valid hex, the OTLP exporter fell
through to hash(span.trace_id) & ((1<<128)-1), producing a 64-bit value where
OTLP expects 128 bits of randomness. All OTLP trace IDs had the upper 64 bits
zeroed, making correlated traces appear non-random in Jaeger/Grafana Tempo.

Rule: trace_id = uuid.uuid4().hex (32 hex chars = 128 bits, OTLP-valid). run_id
is the human-readable directory label ("run_abc123"). Always generate them
independently in start_trace(). Never conflate trace identity with run identity.
The Trace dataclass already had separate fields; the bug was in the instantiation.

Anti-sycophancy check: Not caught in passes 1 or 2. Found in pass 3 by auditing
the OTLP exporter's _is_hex() fallback path and tracing back to where trace_id
is set. A regression test (test_start_trace_trace_id_is_hex_and_differs_from_run_id)
now enforces this invariant.

---

## 2026-06-19 — cmd_replay ignores AGENT_TRACE_TRACE_DIR

Pattern: _cli.cmd_replay called `replay(run_id)` without `trace_dir=_trace_dir()`.
`_require_run_dir()` and `_fixture_path()` both honour `AGENT_TRACE_TRACE_DIR` via
`_trace_dir()`, but the actual `replay()` call used the default `~/.agent-trace/runs`,
causing a FileNotFoundError for any user with a custom trace dir.

Rule: Every path resolution in _cli.py that is not an explicit absolute path must go
through `_trace_dir()`. Cross-check all CLI commands when adding env-var-controlled
directory overrides.

Anti-sycophancy check: Caught in bug audit pass 4. Two sibling functions in the same
file used `_trace_dir()` correctly; the inconsistency in cmd_replay was not caught in
earlier passes.

---

## 2026-06-19 — Silent exception swallow in step span lifecycle

Pattern: instrument_runner had `except Exception: step_span.end(SpanStatus.OK)` with
no logging. Any exception reaching that branch (currently impossible since
_enrich_step_span catches internally, but reachable if that changes) would be
completely invisible. Also, calling `end()` unconditionally in the except block would
double-set end_time if the try's `end()` had already set it.

Rule: Exception branches in span lifecycle code must always log at debug level and guard
`end()` with `if span.end_time is None`. Silent swallows make replay failures
undiagnosable.

Anti-sycophancy check: Caught in bug audit pass 4. The issue was masked by the internal
try/except in _enrich_step_span, so earlier passes did not flag it.

---

## 2026-06-19 — Spurious pydantic dependency in pyproject.toml

Pattern: `pydantic>=2.7` was listed as a core dependency in `[project] dependencies`
but was never imported anywhere in src/, tests/, benchmarks/, or examples/. Every
`pip install agent-trace` user received a ~4MB Rust-compiled package they didn't need.

Rule: After any dependency is added to pyproject.toml, grep for its import in the full
codebase before merging. A dependency that appears only in pyproject.toml and nowhere in
Python files is a strong signal it was added speculatively and never used. Audit core
dependencies separately from optional extras — spurious core deps affect all users, not
just those who opt in.

Anti-sycophancy check: Caught in bug audit pass 7 by reading pyproject.toml for the
first time. Six passes of source-file auditing never caught it because the bug was in
the packaging config, not the Python code.

---

## 2026-06-19 — replay() does not accept direct fixture.db file paths

Pattern: `replay()` always appended `/fixture.db` to its input path, so passing a
path that already ends in `.db` (e.g. `Path("fixtures/fixture.db")`) would look for
`fixture.db/fixture.db` and raise `FileNotFoundError`. The CI pipeline example
(`examples/03-ci-pipeline/`) used this pattern directly in both `example.py` and
`test_with_fixture.py`, making those examples fail silently without a recorded fixture
to test against.

Rule: When the resolved path has a `.db` suffix, treat it as the fixture file directly.
When it has no suffix, append `/fixture.db`. Gate: `fixture_path = p if p.suffix == ".db" else p / "fixture.db"`.
Test both forms (file path and directory path) in unit tests.

Anti-sycophancy check: Caught in bug audit pass 8 by reading example files for the
first time. Prior passes covered src/, tests/, benchmarks/ but not examples/. The fix
required touching the public API (`__init__.py`) and adding two regression tests.

---

## 2026-06-19 — Pre-commit mypy additional_dependencies stale after pydantic removal

Pattern: When pydantic was removed from `pyproject.toml` in pass 7, the matching
`pydantic>=2.7` entry in `.pre-commit-config.yaml` under the mypy hook's
`additional_dependencies` was missed. Anyone running `pre-commit run mypy` continued
to install pydantic unnecessarily. Config-file dependencies are not caught by import
scanning, so the fix in pyproject.toml did not propagate here.

Rule: When removing a dependency from pyproject.toml, grep for its name in
`.pre-commit-config.yaml` and `tox.ini` as well — those files carry parallel
dependency lists that static import analysis cannot see.

Anti-sycophancy check: Caught in bug audit pass 8 by reading .pre-commit-config.yaml
for the first time. Removing a package from one config file and missing it in a sibling
config file is the canonical "fix in one place, break in another" class of error.

---

## 2026-06-19 — uv.lock must be committed alongside pyproject.toml changes

Pattern: When pydantic was removed from pyproject.toml in commit 1bf3ec0, uv.lock
was updated locally (pydantic entries removed) but never staged or committed. The
lockfile and manifest diverged silently in git. Anyone running `uv sync` from a
fresh clone would install pydantic despite it not being a declared dependency.

Rule: Any change to pyproject.toml that adds or removes a dependency must include
a `uv lock` run and the resulting uv.lock in the same commit. Add `uv lock --check`
to CI so that lockfile/manifest divergence is caught automatically on every PR.

Anti-sycophancy check: Caught in bug audit pass 12 by inspecting `git diff --stat HEAD`.
The divergence was invisible to import scanning and ruff/mypy — only a git diff check
caught it. Six prior passes auditing pyproject.toml and .pre-commit-config.yaml both
missed uv.lock because it was not read during those passes.

---

## 2026-06-19 — LangGraph 1.x passes serialized=None to all callbacks

Pattern: LangGraph 0.x passed a `dict[str, Any]` as `serialized` to every callback
method (`on_chain_start`, `on_llm_start`, `on_chat_model_start`, `on_tool_start`).
LangGraph 1.0+ passes `None` for `serialized` and moves the name into `**kwargs['name']`.
Every `.get()` call on `None` crashed with `AttributeError: 'NoneType' object has no
attribute 'get'`, causing 0 spans to be captured silently (langchain_core catches and
logs callback errors without re-raising them by default).

Rule: Always guard `serialized` with `ser = serialized or {}`. Check `kwargs.get("name")`
FIRST when extracting names from callbacks — it is the authoritative source in LangGraph 1.x
and is also populated in 0.x. Never call methods directly on a typed-but-nullable parameter.

Anti-sycophancy check: Not caught in 18 passes of quality-gate runs because the crash
was swallowed by langchain_core's callback error handler. Zero spans is a silent failure.
Only caught when the integration was tested end-to-end with a real LangGraph invocation.
The version bump (0.6→1.2.6) triggered a breaking API change that 18 passes of unit tests
and mypy could not detect because the integration was not exercised under real conditions.

---

## 2026-06-20 — Python 3.14 threads do not inherit ContextVar by default

Pattern: `sys.flags.thread_inherit_context` is `0` in Python 3.14 (and earlier). When a new
`threading.Thread` is spawned, its ContextVar values are all set to their defaults — not
copied from the parent thread. `Tracer._active_trace_var` is a ContextVar, so any span
created inside a spawned thread lands in a detached (trace-less) span object instead of
`trace.spans`. This is a silent failure — `start_span()` returns a valid Span, just not
attached to any trace.

Rule: Any code that spawns threads and expects `Tracer.start_span()` to record into the
active trace must explicitly pass the context using `contextvars.copy_context().run(fn, ...)`.
Do NOT rely on ContextVar inheritance in threads. Document this in any public API that
calls `start_span()` from multiple threads (e.g. `LangGraphTracer` for concurrent graphs).

Anti-sycophancy check: Not caught until the concurrent integration test was written.
`LangGraphTracer` had a `threading.Lock` protecting `_spans` but no documentation or guard
against the detached-span silent failure in spawned threads.

---

## 2026-06-20 — ReplayAdapter.send() fallback raised NotImplementedError

Pattern: `ReplayAdapter` inherits from `BaseAdapter` (requests). When no fixture entry
matched the request and `AGENT_TRACE_NETWORK_GUARD=0` (the default), the fallback path
called `super().send()` which resolves to `BaseAdapter.send()` — an abstract method that
unconditionally raises `NotImplementedError`. Any live-network fallback during replay
silently crashed instead of making a real request.

Rule: Never call `super().send()` on `BaseAdapter` — it is abstract. The correct fallback
is to instantiate a concrete `HTTPAdapter()`, call `.send()`, and `.close()` in a
`try/finally`. Mirror how the httpx equivalent uses `httpx.HTTPTransport()` for its own
fallback. Add a test that patches `HTTPAdapter.send` to verify the fallback path reaches
it (rather than `BaseAdapter.send`).

Anti-sycophancy check: Not caught in any prior pass because the guard=0 fallback path had
no test. The guard=1 path (which raises NetworkGuardError before reaching the fallback) was
tested, giving false coverage confidence for the else branch.
