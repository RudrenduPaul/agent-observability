"""
SQLite-backed HTTP fixture for record/replay.

Recording path: the interceptor transports call record_exchange() after each
real HTTP round-trip.  The fixture appends a row with full request/response
data and a monotonically increasing sequence_num.

Replay path: the replay transports call next_exchange() which serves rows in
sequence_num order, using a per-(method:url) cursor so that the same URL
called multiple times is replayed in the same order it was recorded.

Why SQLite and not JSON files?
- Concurrent test workers can each open their own fixture file with WAL mode.
- Large response bodies don't balloon memory — they stay on disk until needed.
- sequence_num gives a total ordering across all URLs, which is necessary for
  multi-agent traces where two different hosts may be called interleaved.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = ["Fixture", "max_inter_chunk_gap_ms", "time_to_first_chunk_ms"]
# Note: Fixture's diff_response_shapes()/retry_groups()/
# exchanges_for_correlation_id()/correlation_ids() are public methods on the
# already-exported Fixture class, not separate module-level names.

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS http_exchanges (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT NOT NULL,
    url              TEXT NOT NULL,
    method           TEXT NOT NULL,
    request_headers  TEXT NOT NULL DEFAULT '{}',
    request_body     TEXT NOT NULL DEFAULT '',
    response_status  INTEGER,
    response_headers TEXT NOT NULL DEFAULT '{}',
    response_body    TEXT NOT NULL DEFAULT '',
    recorded_at      REAL NOT NULL,
    sequence_num     INTEGER NOT NULL,
    duration_ms      REAL,
    error_type       TEXT,
    error_message    TEXT
);
-- Composite index so next_exchange() lookups use the PK order efficiently
-- instead of scanning and sorting the full table on every call.
CREATE INDEX IF NOT EXISTS idx_exchanges_method_url_seq
    ON http_exchanges (method, url, sequence_num);
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- WebSocket frames for persistent duplex sessions (e.g. the OpenAI Agents
-- SDK's Realtime API).  Unlike http_exchanges, a single connection_id can
-- have many rows in each direction, so replay is served per
-- (connection_id, direction) rather than per (method, url).
CREATE TABLE IF NOT EXISTS ws_frames (
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
CREATE INDEX IF NOT EXISTS idx_ws_frames_conn_dir_seq
    ON ws_frames (connection_id, direction, sequence_num);
-- MCP stdio-transport JSON-RPC frames — one row per frame flowing over a
-- child process's stdin (direction='to_server') or stdout
-- (direction='from_server').  Distinct from http_exchanges because MCP's
-- stdio transport has no HTTP request/response pairing: notifications carry
-- no id, and a single session emits many frames per (server_command, tool).
CREATE TABLE IF NOT EXISTS mcp_frames (
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
CREATE INDEX IF NOT EXISTS idx_mcp_frames_seq
    ON mcp_frames (sequence_num);
"""

# Columns added after the original schema. CREATE TABLE IF NOT EXISTS above
# only applies to a brand-new database file — a fixture.db created by an
# older agent-trace version already has an http_exchanges table without
# these columns, and SQLite has no "ADD COLUMN IF NOT EXISTS". _migrate()
# adds them defensively, treating "column already exists" as a no-op.
#
# response_status's NOT NULL constraint (dropped above, for new databases)
# cannot be relaxed on an existing table via ALTER TABLE in SQLite — a
# fixture.db created before this change keeps response_status NOT NULL, so
# record_exchange()'s failed-before-response path (response_status=None)
# only works against a fixture.db created under the current schema. This is
# a deliberate, honest limitation rather than a full table-rebuild
# migration: pre-existing fixtures keep working exactly as before for their
# existing (always-succeeded) rows.
_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("duration_ms", "REAL"),
    ("error_type", "TEXT"),
    ("error_message", "TEXT"),
    # JSON-encoded list of per-chunk arrival offsets (seconds since the
    # request was dispatched), populated only when the exchange was recorded
    # via a streaming/pass-through transport (RecordingTransport(...,
    # stream=True) / AsyncRecordingTransport(..., stream=True)). NULL for
    # every exchange recorded the historical eager-buffering way — absence
    # means "not captured", not "arrived instantly".
    ("chunk_timestamps", "TEXT"),
    # Caller-supplied correlation identifier (e.g. propagated via
    # agent_trace.interceptor.httpx_hook.correlation_context()) tying this
    # exchange back to the concurrent batch input or graph node that
    # produced it. NULL when the caller didn't set one.
    ("correlation_id", "TEXT"),
    # Content-hash of (method, url, request_body) — always computed at
    # write time (see _inspect.content_hash), regardless of caller input.
    # Rows sharing the same attempt_group are attempts of one logical
    # request (e.g. an SDK's own automatic retry-on-5xx), letting a
    # developer see "this logical call took N retries before a 200"
    # without manually sequencing same-URL rows by eye.
    ("attempt_group", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any schema columns introduced after the original release to a
    pre-existing http_exchanges table. Safe to call on every open — each
    ALTER TABLE is caught individually so an already-migrated database
    (or a brand-new one where _SCHEMA already created the column) is a
    silent no-op rather than an error."""
    for column, sql_type in _MIGRATION_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE http_exchanges ADD COLUMN {column} {sql_type}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


def _content_hash(method: str, url: str, request_body: str) -> str:
    """Stable identity for "this is the same logical request" — rows
    sharing this hash are attempts of one logical call (see
    ``Fixture.retry_groups()``). Mirrors
    ``agent_trace._inspect.content_hash`` (duplicated rather than imported
    to keep this module import-cycle-free of the CLI-facing package).

    sha1 here is a content-identity fingerprint, not a security boundary —
    usedforsecurity=False documents that and avoids FIPS-mode failures.
    """
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(method.upper().encode("utf-8"))
    digest.update(b"\0")
    digest.update(url.encode("utf-8"))
    digest.update(b"\0")
    digest.update((request_body or "").encode("utf-8"))
    return digest.hexdigest()


def _row_to_exchange(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    return {
        "url": row["url"],
        "method": row["method"],
        "request_headers": json.loads(row["request_headers"]),
        "request_body": row["request_body"],
        "response_status": row["response_status"],
        "response_headers": json.loads(row["response_headers"]),
        "response_body": row["response_body"],
        "recorded_at": row["recorded_at"],
        "sequence_num": row["sequence_num"],
        # None on a fixture.db row recorded before these columns existed —
        # callers must treat absence as "unknown", not "zero"/"no error".
        "duration_ms": row["duration_ms"] if "duration_ms" in keys else None,
        "error_type": row["error_type"] if "error_type" in keys else None,
        "error_message": row["error_message"] if "error_message" in keys else None,
        "chunk_timestamps": (
            json.loads(row["chunk_timestamps"])
            if "chunk_timestamps" in keys and row["chunk_timestamps"]
            else None
        ),
        "correlation_id": row["correlation_id"] if "correlation_id" in keys else None,
        "attempt_group": row["attempt_group"] if "attempt_group" in keys else None,
    }


def time_to_first_chunk_ms(exchange: dict[str, Any]) -> float | None:
    """Milliseconds from request dispatch to the first streamed chunk
    arriving, or None if this exchange has no per-chunk timestamps recorded
    (not captured via a streaming transport, or a response with zero body
    chunks)."""
    timestamps = exchange.get("chunk_timestamps")
    if not timestamps:
        return None
    return float(timestamps[0]) * 1000


def max_inter_chunk_gap_ms(exchange: dict[str, Any]) -> float | None:
    """Largest gap (ms) between two consecutive streamed chunks, or None if
    this exchange has no per-chunk timestamps recorded. 0.0 for a single
    chunk (nothing to measure a gap against)."""
    timestamps = exchange.get("chunk_timestamps")
    if not timestamps:
        return None
    if len(timestamps) < 2:
        return 0.0
    gaps = [b - a for a, b in itertools.pairwise(timestamps)]
    return float(max(gaps)) * 1000


def _row_to_ws_frame(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "connection_id": row["connection_id"],
        "url": row["url"],
        "direction": row["direction"],
        "frame_type": row["frame_type"],
        "payload": row["payload"],
        "recorded_at": row["recorded_at"],
        "sequence_num": row["sequence_num"],
    }


def _row_to_mcp_frame(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "server_command": row["server_command"],
        "direction": row["direction"],
        "frame_type": row["frame_type"],
        "rpc_id": row["rpc_id"],
        "method": row["method"],
        "payload": row["payload"],
        "recorded_at": row["recorded_at"],
        "sequence_num": row["sequence_num"],
    }


class Fixture:
    """Thread-safe SQLite-backed HTTP fixture.

    Parameters
    ----------
    path:
        Filesystem path for the SQLite database.  The file is created if it
        does not exist.
    trace_id:
        Optional trace identifier stored in every recorded exchange.  Pass an
        empty string (the default) when the trace_id is not yet known.
    on_exchange_recorded:
        Optional callback invoked with the just-recorded exchange dict
        (same shape as an ``all_exchanges()``/``next_exchange()`` entry,
        plus its ``id``) immediately after each successful
        ``record_exchange()`` commit. Wire this to a remote fixture backend
        (see ``agent_trace.exporters.remote_fixture``) to durably persist
        each exchange as it's recorded — so a worker process killed or
        swept mid-run (e.g. on a managed platform, issue #7417) still has
        every exchange recorded up to that point recoverable from remote
        storage, instead of only the local, ephemeral ``fixture.db`` this
        process's filesystem may never be read again. Exceptions raised by
        the callback are caught and logged, never propagated — a remote
        upload failure must not break the local recording it's piggybacking
        on.
    """

    def __init__(
        self,
        path: Path,
        trace_id: str = "",
        on_exchange_recorded: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._path = path
        self._trace_id = trace_id
        self._on_exchange_recorded = on_exchange_recorded
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        _migrate(self._conn)
        self._conn.commit()
        # Per-(method:url) last-served row id for next_exchange().
        # Stores the `id` of the most recently served row (0 = none yet).
        # Using id > last_id avoids O(n^2) OFFSET scans — each lookup is
        # O(log n) via the composite index on (method, url, sequence_num).
        self._read_cursor: dict[str, int] = {}
        # Per-(connection_id:direction) last-served row id for
        # next_ws_frame(). Same id > last_id strategy as _read_cursor above.
        self._ws_read_cursor: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_exchange(
        self,
        url: str,
        method: str,
        request_headers: dict[str, str],
        request_body: str,
        response_status: int | None = None,
        response_headers: dict[str, str] | None = None,
        response_body: str | None = None,
        *,
        duration_ms: float | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        chunk_timestamps: list[float] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Persist one HTTP round-trip — or one failed-before-response
        attempt — to the fixture database.

        ``correlation_id``, when provided by the caller (e.g. via
        ``agent_trace.interceptor.httpx_hook.correlation_context()``), ties
        this exchange back to the concurrent batch input or graph node that
        produced it — see ``exchanges_for_correlation_id()`` below.
        ``attempt_group`` is *not* a parameter: it's always computed from
        ``(method, url, request_body)`` at write time (see
        ``agent_trace._inspect.content_hash``) so that repeated calls with
        an identical request signature — the shape an SDK's own automatic
        retry-on-5xx logic produces — are grouped as attempts of one logical
        request with zero caller wiring required. See ``retry_groups()``.

        Two mutually-exclusive shapes:

        - A genuine exchange: pass ``response_status``/``response_headers``/
          ``response_body`` (the historical call shape — every existing
          caller already passes these three).
        - A failed-before-response attempt (connection refused, DNS
          failure, TLS failure, a malformed URL raising before any
          ``httpx.Response``/``requests.Response`` exists, ...): pass
          ``error_type``/``error_message`` instead and leave
          ``response_status`` as None. ``request_headers``/``request_body``
          are still recorded since they're fully constructed *before* the
          network call is attempted, so they're always available even when
          the call itself never completes.

        Raises ``ValueError`` if neither a response nor an error was given —
        every row must be one shape or the other, never neither.

        Uses time.time() intentionally — we want the *wall-clock* moment the
        exchange was recorded, not the abstract clock.  This timestamp is for
        audit/debugging only; replay ordering is driven by sequence_num, not
        recorded_at. ``duration_ms``, when provided, is the caller-measured
        elapsed time for the underlying transport call (dispatch to response,
        or dispatch to the failure being raised) — likewise audit/debugging
        data, not used for replay ordering. ``chunk_timestamps``, when
        provided, is a list of per-chunk arrival offsets (seconds since
        dispatch) for a streamed response recorded via a pass-through
        transport — see ``time_to_first_chunk_ms``/``max_inter_chunk_gap_ms``
        below for reading it back.
        """
        if response_status is None and error_type is None:
            raise ValueError(
                "record_exchange() requires either response_status (a "
                "genuine HTTP response) or error_type (a failed-before-"
                "response attempt) — got neither."
            )
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(sequence_num), -1) + 1 FROM http_exchanges"
            )
            row = cur.fetchone()
            next_seq: int = int(row[0])
            attempt_group = _content_hash(method, url, request_body)

            self._conn.execute(
                """\
                INSERT INTO http_exchanges
                    (trace_id, url, method, request_headers, request_body,
                     response_status, response_headers, response_body,
                     recorded_at, sequence_num, duration_ms, error_type,
                     error_message, chunk_timestamps, correlation_id,
                     attempt_group)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._trace_id,
                    url,
                    method.upper(),
                    json.dumps(request_headers),
                    request_body,
                    response_status,
                    json.dumps(response_headers or {}),
                    response_body or "",
                    time.time(),  # wall-clock intentional — see docstring
                    next_seq,
                    duration_ms,
                    error_type,
                    error_message,
                    (
                        json.dumps(chunk_timestamps)
                        if chunk_timestamps is not None
                        else None
                    ),
                    correlation_id,
                    attempt_group,
                ),
            )
            self._conn.commit()
            id_row = self._conn.execute("SELECT last_insert_rowid()").fetchone()
            recorded_row_id = int(id_row[0])

        if self._on_exchange_recorded is not None:
            try:
                exchange = {
                    "id": recorded_row_id,
                    "url": url,
                    "method": method.upper(),
                    "request_headers": request_headers,
                    "request_body": request_body,
                    "response_status": response_status,
                    "response_headers": response_headers or {},
                    "response_body": response_body or "",
                    "recorded_at": time.time(),
                    "sequence_num": next_seq,
                    "duration_ms": duration_ms,
                    "error_type": error_type,
                    "error_message": error_message,
                    "chunk_timestamps": chunk_timestamps,
                    "correlation_id": correlation_id,
                    "attempt_group": attempt_group,
                }
                self._on_exchange_recorded(exchange)
            except Exception:
                logger.warning(
                    "agent-trace: on_exchange_recorded callback raised — "
                    "exchange was still recorded locally",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # MCP stdio-transport frame recording
    # ------------------------------------------------------------------

    def record_mcp_frame(
        self,
        server_command: str,
        direction: str,
        frame_type: str,
        rpc_id: str | None,
        method: str | None,
        payload: str,
    ) -> None:
        """Persist one MCP stdio JSON-RPC frame to the fixture database.

        Parameters
        ----------
        server_command:
            The command (and args) used to launch the MCP server subprocess,
            used to disambiguate frames when multiple servers are recorded
            into the same fixture (e.g. a ``MultiServerMCPClient`` session).
        direction:
            ``"to_server"`` (client stdin) or ``"from_server"`` (client stdout).
        frame_type:
            ``"request"``, ``"response"``, ``"notification"``, or ``"error"``.
        rpc_id:
            The JSON-RPC ``id`` as a string, or None for notifications.
        method:
            The JSON-RPC ``method`` name, or None for responses/errors.
        payload:
            The raw JSON text of the frame, exactly as sent/received on the wire.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(sequence_num), -1) + 1 FROM mcp_frames"
            )
            row = cur.fetchone()
            next_seq: int = int(row[0])

            self._conn.execute(
                """\
                INSERT INTO mcp_frames
                    (trace_id, server_command, direction, frame_type, rpc_id,
                     method, payload, recorded_at, sequence_num)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._trace_id,
                    server_command,
                    direction,
                    frame_type,
                    rpc_id,
                    method,
                    payload,
                    time.time(),  # wall-clock intentional — see record_exchange
                    next_seq,
                ),
            )
            self._conn.commit()

    def all_mcp_frames(self) -> list[dict[str, Any]]:
        """Return every recorded MCP frame in sequence_num order."""
        with self._lock:
            cur = self._conn.execute(
                """\
                SELECT server_command, direction, frame_type, rpc_id, method,
                       payload, recorded_at, sequence_num
                FROM mcp_frames
                ORDER BY sequence_num ASC
                """
            )
            rows = cur.fetchall()

        return [_row_to_mcp_frame(row) for row in rows]

    def mcp_frame_count(self) -> int:
        """Return total number of recorded MCP frames."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM mcp_frames")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def next_exchange(self, url: str, method: str) -> dict[str, Any] | None:
        """Return the next recorded exchange for *(method, url)* or None.

        Exchanges are served in the order they were recorded (ascending
        sequence_num).  Each (method:url) key maintains its own row-id cursor
        so that the same URL called multiple times replays responses in order.

        Uses ``id > last_served_id`` instead of OFFSET so each call is O(log n)
        via the composite index — not O(n) like OFFSET would be.
        """
        key = f"{method.upper()}:{url}"
        with self._lock:
            last_id = self._read_cursor.get(key, 0)
            cur = self._conn.execute(
                """\
                SELECT id, url, method, request_headers, request_body,
                       response_status, response_headers, response_body,
                       recorded_at, sequence_num, duration_ms, error_type,
                       error_message, chunk_timestamps, correlation_id,
                       attempt_group
                FROM http_exchanges
                WHERE method = ? AND url = ? AND id > ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (method.upper(), url, last_id),
            )
            row = cur.fetchone()
            if row is None:
                return None

            self._read_cursor[key] = int(row["id"])
            return _row_to_exchange(row)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def all_exchanges(self) -> list[dict[str, Any]]:
        """Return every recorded exchange in sequence_num order."""
        with self._lock:
            cur = self._conn.execute(
                """\
                SELECT url, method, request_headers, request_body,
                       response_status, response_headers, response_body,
                       recorded_at, sequence_num, duration_ms, error_type,
                       error_message, chunk_timestamps, correlation_id,
                       attempt_group
                FROM http_exchanges
                ORDER BY sequence_num ASC
                """
            )
            rows = cur.fetchall()

        return [_row_to_exchange(row) for row in rows]

    def diff_response_shapes(self, url: str) -> list[set[str]]:
        """Return the distinct sets of top-level JSON response keys seen
        across every recorded exchange to *url*, so a developer can spot
        "this endpoint sometimes returns shape A, sometimes shape B"
        without manually looping over ``all_exchanges()`` and diffing
        ``dict.keys()`` by hand (#3994).

        Only exchanges whose ``response_body`` is a JSON object are
        considered — a non-JSON or non-dict body is silently skipped rather
        than raised on. Returns an empty list if *url* was never called, or
        if every response for it failed to parse as a JSON object.
        """
        shapes: list[set[str]] = []
        seen: list[frozenset[str]] = []
        for exchange in self.all_exchanges():
            if exchange["url"] != url:
                continue
            try:
                body = json.loads(exchange["response_body"])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(body, dict):
                continue
            key_set = frozenset(body.keys())
            if key_set not in seen:
                seen.append(key_set)
                shapes.append(set(key_set))
        return shapes

    def retry_groups(self) -> dict[str, list[dict[str, Any]]]:
        """Return ``{attempt_group: [exchanges]}`` for every attempt_group
        with more than one recorded exchange — i.e. every logical request
        that was attempted more than once (typically an SDK's own automatic
        retry-on-5xx logic), each list ordered by sequence_num.

        ``attempt_group`` is a content-hash of ``(method, url,
        request_body)`` computed automatically at record time — this
        requires no caller wiring; identical repeated requests are grouped
        for free. See #5508.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for exchange in self.all_exchanges():
            group = exchange.get("attempt_group")
            if not group:
                continue
            groups.setdefault(group, []).append(exchange)
        return {group: rows for group, rows in groups.items() if len(rows) > 1}

    def exchanges_for_correlation_id(self, correlation_id: str) -> list[dict[str, Any]]:
        """Return every recorded exchange tagged with *correlation_id* (see
        ``agent_trace.interceptor.httpx_hook.correlation_context()``), in
        sequence_num order — the exchanges that belong to one concurrent
        batch input or graph node (#30924, #6037)."""
        return [
            e for e in self.all_exchanges() if e.get("correlation_id") == correlation_id
        ]

    def correlation_ids(self) -> list[str]:
        """Return the distinct, non-null correlation_ids recorded in this
        fixture, in first-seen (sequence_num) order."""
        seen: list[str] = []
        for exchange in self.all_exchanges():
            cid = exchange.get("correlation_id")
            if cid and cid not in seen:
                seen.append(cid)
        return seen

    def reset_read_cursor(self) -> None:
        """Reset all per-(method:url) read offsets to 0.

        Call this at the start of each replay so the same fixture can be
        replayed multiple times within one process lifetime.
        """
        with self._lock:
            self._read_cursor.clear()

    def exchange_count(self) -> int:
        """Return total number of recorded exchanges."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM http_exchanges")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def failed_exchange_count(self) -> int:
        """Return the number of recorded failed-before-response attempts —
        rows with no response_status (connection refused, DNS failure, a
        malformed URL, ...), distinct from a genuine HTTP 4xx/5xx response.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM http_exchanges WHERE response_status IS NULL"
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def earliest_timestamp(self) -> float | None:
        """Return the earliest recorded_at timestamp, or None if empty."""
        with self._lock:
            cur = self._conn.execute("SELECT MIN(recorded_at) FROM http_exchanges")
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else None

    # ------------------------------------------------------------------
    # WebSocket frames (persistent duplex sessions, e.g. Realtime API)
    # ------------------------------------------------------------------

    def record_ws_frame(
        self,
        connection_id: str,
        url: str,
        direction: str,
        payload: str,
        frame_type: str = "text",
    ) -> None:
        """Persist one WebSocket frame to the fixture database.

        Parameters
        ----------
        connection_id:
            Identifier for the logical WS connection this frame belongs to
            (a single fixture can hold frames from multiple connections/
            sessions recorded in the same trace).
        direction:
            ``"send"`` for a frame the client sent, ``"recv"`` for a frame
            the client received.
        frame_type:
            ``"text"`` or ``"binary"``.  Binary payloads are stored as
            UTF-8-decoded text (errors="replace"), matching how the
            recording interceptor prepares them before this call.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(sequence_num), -1) + 1 FROM ws_frames"
            )
            row = cur.fetchone()
            next_seq: int = int(row[0])

            self._conn.execute(
                """\
                INSERT INTO ws_frames
                    (trace_id, connection_id, url, direction, frame_type,
                     payload, recorded_at, sequence_num)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._trace_id,
                    connection_id,
                    url,
                    direction,
                    frame_type,
                    payload,
                    time.time(),  # wall-clock intentional — see record_exchange
                    next_seq,
                ),
            )
            self._conn.commit()

    def next_ws_frame(
        self, connection_id: str, direction: str
    ) -> dict[str, Any] | None:
        """Return the next recorded frame for *(connection_id, direction)* or None.

        Frames are served in the order they were recorded (ascending
        sequence_num), independently for each direction, so a replayed
        session sees its inbound ("recv") frames in original order without
        being coupled to how many frames were sent in between.
        """
        key = f"{connection_id}:{direction}"
        with self._lock:
            last_id = self._ws_read_cursor.get(key, 0)
            cur = self._conn.execute(
                """\
                SELECT id, connection_id, url, direction, frame_type,
                       payload, recorded_at, sequence_num
                FROM ws_frames
                WHERE connection_id = ? AND direction = ? AND id > ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (connection_id, direction, last_id),
            )
            row = cur.fetchone()
            if row is None:
                return None

            self._ws_read_cursor[key] = int(row["id"])
            return _row_to_ws_frame(row)

    def all_ws_frames(self, connection_id: str | None = None) -> list[dict[str, Any]]:
        """Return recorded WS frames in sequence_num order.

        Pass *connection_id* to filter to a single connection; omit it to
        return every frame across every connection recorded in this fixture.
        """
        with self._lock:
            if connection_id is None:
                cur = self._conn.execute(
                    """\
                    SELECT connection_id, url, direction, frame_type,
                           payload, recorded_at, sequence_num
                    FROM ws_frames
                    ORDER BY sequence_num ASC
                    """
                )
            else:
                cur = self._conn.execute(
                    """\
                    SELECT connection_id, url, direction, frame_type,
                           payload, recorded_at, sequence_num
                    FROM ws_frames
                    WHERE connection_id = ?
                    ORDER BY sequence_num ASC
                    """,
                    (connection_id,),
                )
            rows = cur.fetchall()

        return [_row_to_ws_frame(row) for row in rows]

    def reset_ws_read_cursor(self) -> None:
        """Reset all per-(connection_id:direction) read offsets to 0.

        Call this at the start of each replay so the same fixture can be
        replayed multiple times within one process lifetime.
        """
        with self._lock:
            self._ws_read_cursor.clear()

    def ws_frame_count(self, connection_id: str | None = None) -> int:
        """Return total number of recorded WS frames, optionally per-connection."""
        with self._lock:
            if connection_id is None:
                cur = self._conn.execute("SELECT COUNT(*) FROM ws_frames")
            else:
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM ws_frames WHERE connection_id = ?",
                    (connection_id,),
                )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value: str) -> None:
        """Upsert a key/value pair in the metadata table."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get_metadata(self, key: str) -> str | None:
        """Return the stored value for *key*, or None if absent."""
        with self._lock:
            cur = self._conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cur.fetchone()
            return str(row["value"]) if row else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Fixture:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
