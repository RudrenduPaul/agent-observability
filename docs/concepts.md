# Concepts

This document explains the internals of agent-trace at the level needed to
understand failure modes, extend the library, or audit its correctness.

---

## 1. What deterministic replay means — and what it does not mean

"Deterministic replay" in agent-trace means that each agent node receives the
same HTTP responses it received during the original recording. The LangGraph
node that called `POST /v1/chat/completions` during recording receives the
exact same response bytes — same status code, same headers, same body — during
replay. From that node's perspective, nothing changed. If the node's logic is
a pure function of its input state and the HTTP response it receives, its output
will be identical to the original run.

What deterministic replay does **not** mean: it does not guarantee the same LLM
output across runs. During replay, no LLM is consulted. The bytes that the LLM
returned during recording are replayed verbatim. This is a feature, not a
limitation — the point of replay is to reproduce the exact failure, which
requires the exact same inputs at each step, including the exact same LLM
response. If the LLM were called again, a stochastic output could mask the bug.

What deterministic replay also does **not** cover: Python-level side effects
that are not mediated by HTTP. If your agent writes to a file, calls a subprocess,
reads a database via a native driver, or uses system time via `time.time()`
directly, those calls are not intercepted. Replay will execute them against
the real environment. See section 7 for the full list of what agent-trace does
not intercept.

---

## 2. The transport interception mechanism

AI SDKs — OpenAI's Python client, the Anthropic SDK, LangChain's HTTP calls —
all create their own HTTP client instances internally. You cannot inject a custom
transport into them at construction time without forking each SDK.

**httpx patch — recording:** rather than patching `httpx.Client.__init__` (an
earlier design, no longer used for recording), the recording path replaces
`httpx.Client._transport_for_url` / `httpx.AsyncClient._transport_for_url` —
the method httpx calls internally on every single request and redirect hop —
so it always wraps whatever transport httpx would have used (default or
caller-supplied, including per-URL `mounts=` transports) in a
`RecordingTransport`/`AsyncRecordingTransport`. This fixes two problems the
`__init__`-time patch had: a client constructed *before* recording activates
(e.g. an LLM client built once at module-import time, as `langgraph dev`
entry points typically do) is still captured, since `_transport_for_url` is
looked up fresh on every `send()`; and a client constructed with an explicit
`transport=` kwarg (e.g. langchain-openai's TCP-keepalive transport) no
longer silently defeats the patch the way `kwargs.setdefault("transport",
...)` could. Both `httpx.Client` (sync) and `httpx.AsyncClient` (async) are
covered — async support is not on a roadmap, it is implemented today
(`src/agent_trace/interceptor/httpx_hook.py`).

**httpx patch — replay:** the replay engine (`src/agent_trace/_replay/engine.py`)
still patches `httpx.Client.__init__`/`httpx.AsyncClient.__init__` directly
(injecting `ReplayTransport`/`AsyncReplayTransport` via
`kwargs.setdefault("transport", ...)`), a genuinely different mechanism from
recording's `_transport_for_url` patch, not just older docs describing the
same thing two ways. The practical consequence — an httpx client constructed
before the `replay(...)` block is entered won't be intercepted — is already
called out in the README's FAQ ("What happens if replay can't find a
matching fixture entry?").

**requests patch:** `requests.Session.get_adapter(url)` is the method that
selects which adapter handles a given URL scheme (`https://`, `http://`). It is
replaced with a function that always returns `RecordingAdapter` (or
`ReplayAdapter`). This means every `Session.send()` call goes through the
fixture regardless of which URL scheme is used.

**Beyond httpx and requests**, dedicated interceptors exist for traffic that
never touches either: gRPC (`interceptor/grpc_hook.py`, patches
`grpc.secure_channel`/`grpc.insecure_channel` and the `grpc.aio` equivalents —
unary-unary and sync unary-stream calls are fully covered, client-streaming
and `grpc.aio` streaming are not), `aiohttp.ClientSession`
(`interceptor/aiohttp_hook.py`), `botocore`'s `URLLib3Session.send`
(`interceptor/botocore_hook.py`, for AWS SDK/Bedrock traffic), WebSocket
connections (`interceptor/websocket_hook.py`), and MCP's stdio JSON-RPC
transport (`interceptor/stdio_hook.py`). See the README's "Known limitations"
section for the exact coverage boundary of each.

All patches are always restored in a `finally` block to prevent leakage into
sibling async tasks or test cases.

The interception layer sits below the SDK's retry logic, authentication
headers, and serialization code. Everything the SDK does — except the actual
network I/O — runs as normal.

---

## 3. The clock abstraction

All timestamp generation in agent-trace core code goes through a single
function:

```python
from agent_trace.core.clock import get_time
```

`get_time()` reads from a `ContextVar[Clock]`. In production, the default
value is `WallClock`, which delegates to `time.time()`. During replay, the
engine calls `set_clock(FixtureClock())` to replace the wall clock with a
replay clock that returns pre-recorded timestamps.

`FixtureClock` starts at `0.0` and advances only when `advance(timestamp)`
is called. The replay engine calls `advance()` before each span is created,
feeding it the `start_time` from the recorded trace. This means span start
times in a replayed trace match the original run's wall-clock times exactly
— not the times at which the replay happened.

**Why this matters:** If you call `time.time()` directly anywhere in the
agent-trace source code, you bypass the clock abstraction. During replay, that
call returns the actual current time, not the recorded time. The result is
non-deterministic span timing. The rule is enforced by a grep check:
`grep -r "time.time()" src/` must return zero results before any commit.
The single permitted exception is `fixture.py`'s `record_exchange()`, which
records the wall-clock moment an exchange was captured (for audit purposes),
not for use in span timestamps.

The clock is stored in a `ContextVar` rather than a module-level global so
that multiple async tasks running concurrently can each have an independent
clock without interfering with each other.

---

## 4. Fixture structure

Each recorded run produces a SQLite database at
`~/.agent-trace/runs/<run_id>/fixture.db`. The schema has four tables:

```sql
CREATE TABLE http_exchanges (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT NOT NULL,
    url              TEXT NOT NULL,
    method           TEXT NOT NULL,
    request_headers  TEXT NOT NULL DEFAULT '{}',   -- JSON
    request_body     TEXT NOT NULL DEFAULT '',
    response_status  INTEGER,
    response_headers TEXT NOT NULL DEFAULT '{}',   -- JSON
    response_body    TEXT NOT NULL DEFAULT '',
    recorded_at      REAL NOT NULL,               -- Unix timestamp, wall-clock
    sequence_num     INTEGER NOT NULL,            -- monotonically increasing
    duration_ms      REAL,
    error_type       TEXT,
    error_message    TEXT
);

CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- WebSocket frames for persistent duplex sessions (e.g. the OpenAI Agents
-- SDK's Realtime API). Unlike http_exchanges, a single connection_id can have
-- many rows in each direction, so replay is served per
-- (connection_id, direction) rather than per (method, url).
CREATE TABLE ws_frames (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT NOT NULL,
    connection_id    TEXT NOT NULL,
    url              TEXT NOT NULL,
    direction        TEXT NOT NULL,
    frame_type       TEXT NOT NULL DEFAULT 'text',
    payload          TEXT NOT NULL DEFAULT '',
    recorded_at      REAL NOT NULL,
    sequence_num     INTEGER NOT NULL
);

-- MCP stdio-transport JSON-RPC frames -- one row per frame flowing over a
-- child process's stdin (direction='to_server') or stdout
-- (direction='from_server'). Distinct from http_exchanges because MCP's
-- stdio transport has no HTTP request/response pairing.
CREATE TABLE mcp_frames (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id       TEXT NOT NULL,
    server_command TEXT NOT NULL,
    direction      TEXT NOT NULL,
    frame_type     TEXT NOT NULL,
    rpc_id         TEXT,
    method         TEXT,
    payload        TEXT NOT NULL,
    recorded_at    REAL NOT NULL,
    sequence_num   INTEGER NOT NULL
);
```

**Serialization rules.** All values stored in `fixture.db` and in `trace.json`
are JSON primitives: `str`, `int`, `float`, `bool`, `None`, `list`, or `dict`.
No `datetime` objects are stored anywhere — timestamps are plain `float` Unix
seconds. Enums are stored by their `.value` (a string). Python `set` is never
used because it is not JSON-serializable.

**Portability.** A `fixture.db` file created on Python 3.10 can be read on
Python 3.13, on any OS, and by any SQLite client — no Python import is needed
to read the raw bytes. The `response_body` field stores the decoded UTF-8 text
of the response (with `errors="replace"` for non-UTF-8 bytes). Binary response
bodies are not currently supported; add a `content_encoding` column and a
base64 field if you need binary.

**WAL mode.** The database is opened with `PRAGMA journal_mode=WAL`. This lets
multiple test workers open the same fixture file concurrently (read-only) without
blocking each other.

---

## 5. Replay sequence

Here is the exact sequence of operations when you use `with replay("run_id") as ctx:`:

1. `replay("run_id")` constructs a `ReplayContext` and resolves the fixture path
   to `~/.agent-trace/runs/run_id/fixture.db`.

2. `__enter__` calls `replay_context(fixture_path)` from `replay.engine`.

3. The engine opens the SQLite fixture and calls `fixture.reset_read_cursor()`
   to clear any per-(method, URL) offsets from previous replays.

4. `FixtureClock()` is constructed and `set_clock(clock)` installs it as the
   active clock for the current context.

5. `httpx.Client.__init__` is patched with a wrapper that injects
   `ReplayTransport(fixture)` as the default transport.

6. `requests.Session.get_adapter` is patched to always return
   `ReplayAdapter(fixture)`.

7. Control returns to user code inside the `with` block.

8. When user code (or the agent under test) calls `httpx.Client.send(request)`,
   `ReplayTransport.handle_request(request)` is called. It calls
   `fixture.next_exchange(url, method)`, advances the per-URL cursor by 1, and
   returns an `httpx.Response` constructed from the stored bytes — with no
   network I/O.

9. If `next_exchange` returns `None` (no recorded entry for this URL/method)
   and `AGENT_TRACE_NETWORK_GUARD=1`, a `NetworkGuardError` is raised
   immediately.

10. When the `with` block exits (success or exception), the `finally` block
    restores the original `httpx.Client.__init__` and `requests.Session.get_adapter`
    via `restore_clock(token)` and closes the fixture's SQLite connection.

---

## 6. The network guard

Setting `AGENT_TRACE_NETWORK_GUARD=1` activates a hard check inside both
`ReplayTransport` and `ReplayAdapter`. When they receive a request that has
no matching entry in the fixture, they raise `NetworkGuardError` instead of
falling through to the real network.

Use the network guard:

- **Always in CI.** If a test that should replay against a fixture silently hits
  a live endpoint, it is non-deterministic, costs tokens, and can fail or pass
  for the wrong reason.
- **In local development** when running test suites that are expected to be
  fully fixtured.

Do not use the network guard:

- During the initial recording step (`record=True`), which must reach the real
  endpoints.
- In development when you want a partially-fixtured run to fall through to live
  APIs for the un-fixtured URLs.

The guard is checked via `os.environ.get("AGENT_TRACE_NETWORK_GUARD", "0") == "1"`.
It is read on every request, so you can toggle it at runtime with
`os.environ["AGENT_TRACE_NETWORK_GUARD"] = "1"` before entering a replay block.

agent-trace's own `pyproject.toml` sets this in `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
env = ["AGENT_TRACE_NETWORK_GUARD=1"]
```

---

## 7. What agent-trace does NOT do

Understanding the boundaries prevents surprises.

**Does not replay arbitrary Python side effects.** If your agent calls
`open("output.txt", "w")`, that file write happens during replay exactly as it
did during recording. There is no filesystem capture or replay.

**Does not guarantee the same LLM output across two recording runs.** Two
separate recordings of the same prompt will produce two different fixtures
because LLM outputs are stochastic. A single fixture replayed multiple times
will produce the same LLM "output" because it is just returning recorded bytes.

**Does not intercept arbitrary non-HTTP external calls.** subprocess calls
(`subprocess.run`, `os.system`), database connections via native C drivers
(psycopg2, sqlite3 without the fixture wrapper), and any other I/O that does
not go through one of the dedicated interceptors is not intercepted. Traffic
that agent frameworks actually use *is* covered beyond plain HTTP: gRPC
(unary calls), aiohttp, botocore/AWS SDK, WebSocket, and MCP's stdio
JSON-RPC transport each have their own interceptor — see section 2.

**Both sync and async httpx are intercepted.** `httpx.Client` and
`httpx.AsyncClient` are both patched (the OpenAI and Anthropic SDKs' default
async clients included) — there is no async gap to work around.

**Does not store binary response bodies cleanly.** Response bodies are decoded
as UTF-8 text with `errors="replace"`. Binary responses (images, audio, PDF
downloads via the API) will have their non-UTF-8 bytes replaced with the
replacement character `�`. Do not use agent-trace for agents that process
binary HTTP responses without first adding binary fixture support.
