"""
Unit tests for agent_trace.replay.fixture — Fixture (SQLite-backed HTTP exchange store).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agent_trace._replay.fixture import Fixture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    fixture: Fixture,
    url: str = "https://api.example.com/v1/test",
    method: str = "POST",
    status: int = 200,
    body: str = '{"ok": true}',
) -> None:
    fixture.record_exchange(
        url=url,
        method=method,
        request_headers={"content-type": "application/json"},
        request_body='{"query": "hello"}',
        response_status=status,
        response_headers={"content-type": "application/json"},
        response_body=body,
    )


# ---------------------------------------------------------------------------
# Creation & basic storage
# ---------------------------------------------------------------------------


class TestFixtureCreation:
    def test_creates_sqlite_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        assert not db_path.exists()
        with Fixture(db_path):
            pass
        assert db_path.exists()

    def test_exchange_count_starts_at_zero(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            assert f.exchange_count() == 0

    def test_record_exchange_increments_count(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            assert f.exchange_count() == 1

    def test_multiple_records_counted(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            for _ in range(5):
                _record(f)
            assert f.exchange_count() == 5


# ---------------------------------------------------------------------------
# next_exchange()
# ---------------------------------------------------------------------------


class TestNextExchange:
    def test_next_exchange_returns_first_recorded(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f, url="https://api.example.com/a", body='{"n": 1}')
            ex = f.next_exchange("https://api.example.com/a", "POST")
            assert ex is not None
            assert ex["response_body"] == '{"n": 1}'

    def test_next_exchange_returns_in_order(self, tmp_path: Path) -> None:
        url = "https://api.example.com/stream"
        with Fixture(tmp_path / "f.db") as f:
            _record(f, url=url, body="first")
            _record(f, url=url, body="second")
            _record(f, url=url, body="third")

            e1 = f.next_exchange(url, "POST")
            e2 = f.next_exchange(url, "POST")
            e3 = f.next_exchange(url, "POST")

            assert e1 is not None and e1["response_body"] == "first"
            assert e2 is not None and e2["response_body"] == "second"
            assert e3 is not None and e3["response_body"] == "third"

    def test_next_exchange_returns_none_when_exhausted(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            f.next_exchange("https://api.example.com/v1/test", "POST")
            # Second call — no more exchanges
            result = f.next_exchange("https://api.example.com/v1/test", "POST")
            assert result is None

    def test_next_exchange_returns_none_for_unknown_url(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            result = f.next_exchange("https://not-recorded.example.com", "GET")
            assert result is None

    def test_next_exchange_method_is_case_insensitive(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(
                f, method="get", url="https://api.example.com/tool", body="tool-out"
            )
            # next_exchange with uppercase should match lowercase recording
            ex = f.next_exchange("https://api.example.com/tool", "GET")
            assert ex is not None
            assert ex["response_body"] == "tool-out"


# ---------------------------------------------------------------------------
# reset_read_cursor()
# ---------------------------------------------------------------------------


class TestResetReadCursor:
    def test_reset_allows_replay_from_beginning(self, tmp_path: Path) -> None:
        url = "https://api.example.com/reset-test"
        with Fixture(tmp_path / "f.db") as f:
            _record(f, url=url, body="replay-me")

            e1 = f.next_exchange(url, "POST")
            assert e1 is not None

            # Cursor exhausted
            assert f.next_exchange(url, "POST") is None

            # Reset and replay
            f.reset_read_cursor()
            e2 = f.next_exchange(url, "POST")
            assert e2 is not None
            assert e2["response_body"] == "replay-me"

    def test_reset_replays_multiple_exchanges_in_order(self, tmp_path: Path) -> None:
        url = "https://api.example.com/seq"
        with Fixture(tmp_path / "f.db") as f:
            for i in range(3):
                _record(f, url=url, body=f"response-{i}")

            for _ in range(3):
                f.next_exchange(url, "POST")

            f.reset_read_cursor()
            bodies = [f.next_exchange(url, "POST")["response_body"] for _ in range(3)]  # type: ignore[index]
            assert bodies == ["response-0", "response-1", "response-2"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_set_and_get_metadata(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.set_metadata("model", "gpt-4o")
            assert f.get_metadata("model") == "gpt-4o"

    def test_get_metadata_missing_key_returns_none(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            assert f.get_metadata("nonexistent") is None

    def test_set_metadata_overwrites_existing(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.set_metadata("key", "v1")
            f.set_metadata("key", "v2")
            assert f.get_metadata("key") == "v2"

    def test_multiple_metadata_keys(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.set_metadata("a", "1")
            f.set_metadata("b", "2")
            assert f.get_metadata("a") == "1"
            assert f.get_metadata("b") == "2"


# ---------------------------------------------------------------------------
# all_exchanges()
# ---------------------------------------------------------------------------


class TestAllExchanges:
    def test_all_exchanges_empty(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            assert f.all_exchanges() == []

    def test_all_exchanges_in_sequence_order(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            urls = [
                "https://api.example.com/first",
                "https://api.example.com/second",
                "https://api.example.com/third",
            ]
            for url in urls:
                _record(f, url=url, method="GET")
            exchanges = f.all_exchanges()
            assert len(exchanges) == 3
            assert [e["url"] for e in exchanges] == urls

    def test_all_exchanges_dict_has_required_keys(self, tmp_path: Path) -> None:
        required_keys = {
            "url",
            "method",
            "request_headers",
            "request_body",
            "response_status",
            "response_headers",
            "response_body",
            "sequence_num",
        }
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            ex = f.all_exchanges()[0]
            assert required_keys.issubset(set(ex.keys()))

    def test_all_exchanges_response_status_correct(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f, status=404, body="not found")
            ex = f.all_exchanges()[0]
            assert ex["response_status"] == 404
            assert ex["response_body"] == "not found"


# ---------------------------------------------------------------------------
# duration_ms — per-HTTP-exchange latency
# ---------------------------------------------------------------------------


class TestDurationMs:
    def test_duration_ms_stored_and_returned(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_exchange(
                url="https://api.example.com/timed",
                method="POST",
                request_headers={},
                request_body="{}",
                response_status=200,
                response_headers={},
                response_body="{}",
                duration_ms=123.45,
            )
            ex = f.all_exchanges()[0]
            assert ex["duration_ms"] == pytest.approx(123.45)

    def test_duration_ms_defaults_to_none_when_not_provided(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            ex = f.all_exchanges()[0]
            assert ex["duration_ms"] is None

    def test_duration_ms_returned_via_next_exchange_too(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_exchange(
                url="https://api.example.com/timed2",
                method="GET",
                request_headers={},
                request_body="",
                response_status=200,
                response_headers={},
                response_body="ok",
                duration_ms=42.0,
            )
            ex = f.next_exchange("https://api.example.com/timed2", "GET")
            assert ex is not None
            assert ex["duration_ms"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Failed-before-response exchanges — error_type/error_message, nullable
# response_status
# ---------------------------------------------------------------------------


class TestFailedExchanges:
    def test_record_failed_exchange_with_no_response_status(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_exchange(
                url="https://bad-host.invalid/x",
                method="POST",
                request_headers={"content-type": "application/json"},
                request_body='{"model": "gpt-4o"}',
                error_type="ConnectError",
                error_message="Connection refused",
            )
            ex = f.all_exchanges()[0]
            assert ex["response_status"] is None
            assert ex["error_type"] == "ConnectError"
            assert ex["error_message"] == "Connection refused"

    def test_failed_exchange_preserves_request_data(self, tmp_path: Path) -> None:
        """Request headers/body are always available even though the call
        never got a response — they were constructed before the failure."""
        with Fixture(tmp_path / "f.db") as f:
            f.record_exchange(
                url="https://bad-host.invalid/x",
                method="POST",
                request_headers={"authorization": "Bearer sk-test"},
                request_body='{"prompt": "hello"}',
                error_type="ConnectTimeout",
                error_message="timed out after 30s",
            )
            ex = f.all_exchanges()[0]
            assert ex["request_headers"] == {"authorization": "Bearer sk-test"}
            assert ex["request_body"] == '{"prompt": "hello"}'

    def test_failed_exchange_response_headers_and_body_default_empty(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_exchange(
                url="https://bad-host.invalid/x",
                method="GET",
                request_headers={},
                request_body="",
                error_type="DNSError",
                error_message="could not resolve host",
            )
            ex = f.all_exchanges()[0]
            assert ex["response_headers"] == {}
            assert ex["response_body"] == ""

    def test_record_exchange_raises_without_status_or_error(
        self, tmp_path: Path
    ) -> None:
        """Neither a response nor an error is a malformed call — must raise,
        not silently write a garbage row."""
        with Fixture(tmp_path / "f.db") as f:
            with pytest.raises(ValueError):
                f.record_exchange(
                    url="https://api.example.com/x",
                    method="GET",
                    request_headers={},
                    request_body="",
                )

    def test_failed_exchange_counted_in_exchange_count(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            f.record_exchange(
                url="https://bad-host.invalid/x",
                method="GET",
                request_headers={},
                request_body="",
                error_type="ConnectError",
                error_message="refused",
            )
            assert f.exchange_count() == 2

    def test_failed_exchange_count_helper(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            _record(f)
            f.record_exchange(
                url="https://bad-host.invalid/x",
                method="GET",
                request_headers={},
                request_body="",
                error_type="ConnectError",
                error_message="refused",
            )
            assert f.failed_exchange_count() == 1
            assert f.exchange_count() == 3

    def test_failed_exchange_count_zero_when_none_recorded(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            assert f.failed_exchange_count() == 0

    def test_genuine_error_response_not_counted_as_failed_exchange(
        self, tmp_path: Path
    ) -> None:
        """A real HTTP 500 has a response_status — it's a provider error,
        not a failed-before-response attempt. Must not be double-counted."""
        with Fixture(tmp_path / "f.db") as f:
            _record(f, status=500, body='{"error": "server error"}')
            assert f.failed_exchange_count() == 0
            assert f.exchange_count() == 1


# ---------------------------------------------------------------------------
# Schema migration — pre-existing databases gain the new columns
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_opening_pre_migration_database_adds_new_columns(
        self, tmp_path: Path
    ) -> None:
        """A fixture.db created under the pre-migration schema (no
        duration_ms/error_type/error_message columns, response_status
        NOT NULL) must still open cleanly, and gains the new columns."""
        import sqlite3

        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """\
            CREATE TABLE http_exchanges (
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
            CREATE TABLE metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO http_exchanges "
            "(trace_id, url, method, request_headers, request_body, "
            " response_status, response_headers, response_body, "
            " recorded_at, sequence_num) "
            "VALUES ('t', 'https://x/y', 'GET', '{}', '', 200, '{}', 'ok', 1.0, 0)"
        )
        conn.commit()
        conn.close()

        # Opening via Fixture must not raise, and the old row must still be
        # readable with the new columns defaulting to None.
        with Fixture(db_path) as f:
            exchanges = f.all_exchanges()
            assert len(exchanges) == 1
            assert exchanges[0]["response_status"] == 200
            assert exchanges[0]["duration_ms"] is None
            assert exchanges[0]["error_type"] is None

            # And new rows recorded after migration work normally.
            f.record_exchange(
                url="https://x/new",
                method="GET",
                request_headers={},
                request_body="",
                response_status=200,
                response_headers={},
                response_body="ok",
                duration_ms=10.0,
            )
            assert f.exchange_count() == 2

    def test_migration_is_idempotent_across_multiple_opens(
        self, tmp_path: Path
    ) -> None:
        """Opening the same fixture.db twice (e.g. two Fixture instances in
        sequence) must not raise on the second migration attempt."""
        db_path = tmp_path / "f.db"
        with Fixture(db_path) as f:
            _record(f)
        # Second open — columns already exist, ALTER TABLE must no-op.
        with Fixture(db_path) as f:
            assert f.exchange_count() == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_exchange(self, tmp_path: Path) -> None:
        """10 threads calling record_exchange() concurrently — count must be 10."""
        with Fixture(tmp_path / "f.db") as f:
            errors: list[Exception] = []

            def worker(i: int) -> None:
                try:
                    _record(f, url=f"https://api.example.com/thread/{i}")
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert f.exchange_count() == 10


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_enter_returns_fixture(self, tmp_path: Path) -> None:
        f = Fixture(tmp_path / "f.db")
        with f as ctx:
            assert ctx is f

    def test_exit_closes_connection(self, tmp_path: Path) -> None:
        """After __exit__, SQLite connection should be closed.
        Further queries should raise an OperationalError."""

        db_path = tmp_path / "f.db"
        f = Fixture(db_path)
        with f:
            _record(f)

        # Connection is closed after exit — subsequent access to _conn should fail
        with pytest.raises(Exception):
            f._conn.execute("SELECT 1")

    def test_context_manager_with_exception(self, tmp_path: Path) -> None:
        """__exit__ closes connection even when an exception is raised."""

        db_path = tmp_path / "f.db"
        f = Fixture(db_path)
        try:
            with f:
                _record(f)
                raise ValueError("test exception")
        except ValueError:
            pass

        with pytest.raises(Exception):
            f._conn.execute("SELECT 1")
