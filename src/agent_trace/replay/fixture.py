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

import json
import sqlite3
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = ["Fixture"]

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS http_exchanges (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id         TEXT NOT NULL,
    url              TEXT NOT NULL,
    method           TEXT NOT NULL,
    request_headers  TEXT NOT NULL DEFAULT '{}',
    request_body     TEXT NOT NULL DEFAULT '',
    response_status  INTEGER NOT NULL,
    response_headers TEXT NOT NULL DEFAULT '{}',
    response_body    TEXT NOT NULL DEFAULT '',
    recorded_at      REAL NOT NULL,
    sequence_num     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


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
    """

    def __init__(self, path: Path, trace_id: str = "") -> None:
        self._path = path
        self._trace_id = trace_id
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Per-(method:url) read offset for next_exchange().
        self._read_cursor: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_exchange(
        self,
        url: str,
        method: str,
        request_headers: dict[str, str],
        request_body: str,
        response_status: int,
        response_headers: dict[str, str],
        response_body: str,
    ) -> None:
        """Persist one HTTP round-trip to the fixture database.

        Uses time.time() intentionally — we want the *wall-clock* moment the
        exchange was recorded, not the abstract clock.  This timestamp is for
        audit/debugging only; replay ordering is driven by sequence_num, not
        recorded_at.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(sequence_num), -1) + 1 FROM http_exchanges"
            )
            row = cur.fetchone()
            next_seq: int = int(row[0]) if row else 0

            self._conn.execute(
                """\
                INSERT INTO http_exchanges
                    (trace_id, url, method, request_headers, request_body,
                     response_status, response_headers, response_body,
                     recorded_at, sequence_num)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._trace_id,
                    url,
                    method.upper(),
                    json.dumps(request_headers),
                    request_body,
                    response_status,
                    json.dumps(response_headers),
                    response_body,
                    time.time(),  # wall-clock intentional — see docstring
                    next_seq,
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def next_exchange(self, url: str, method: str) -> dict[str, Any] | None:
        """Return the next recorded exchange for *(method, url)* or None.

        Exchanges are served in the order they were recorded (ascending
        sequence_num).  Each (method:url) key maintains its own offset so
        that ``GET /v1/chat/completions`` called three times replays as
        response-1, response-2, response-3 in order.
        """
        key = f"{method.upper()}:{url}"
        with self._lock:
            offset = self._read_cursor.get(key, 0)
            cur = self._conn.execute(
                """\
                SELECT url, method, request_headers, request_body,
                       response_status, response_headers, response_body,
                       recorded_at, sequence_num
                FROM http_exchanges
                WHERE method = ? AND url = ?
                ORDER BY sequence_num ASC
                LIMIT 1 OFFSET ?
                """,
                (method.upper(), url, offset),
            )
            row = cur.fetchone()
            if row is None:
                return None

            self._read_cursor[key] = offset + 1

            return {
                "url": row[0],
                "method": row[1],
                "request_headers": json.loads(row[2]),
                "request_body": row[3],
                "response_status": row[4],
                "response_headers": json.loads(row[5]),
                "response_body": row[6],
                "recorded_at": row[7],
                "sequence_num": row[8],
            }

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
                       recorded_at, sequence_num
                FROM http_exchanges
                ORDER BY sequence_num ASC
                """
            )
            rows = cur.fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "url": row[0],
                    "method": row[1],
                    "request_headers": json.loads(row[2]),
                    "request_body": row[3],
                    "response_status": row[4],
                    "response_headers": json.loads(row[5]),
                    "response_body": row[6],
                    "recorded_at": row[7],
                    "sequence_num": row[8],
                }
            )
        return result

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
            return str(row[0]) if row else None

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
