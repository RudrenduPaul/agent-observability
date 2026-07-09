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
# WebSocket frames (ws_frames)
# ---------------------------------------------------------------------------


class TestWsFrames:
    def test_ws_frame_count_starts_at_zero(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            assert f.ws_frame_count() == 0

    def test_record_ws_frame_increments_count(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-1", "wss://x", "send", "hello")
            assert f.ws_frame_count() == 1

    def test_next_ws_frame_returns_first_recorded(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-1", "wss://x", "recv", "first")
            frame = f.next_ws_frame("conn-1", "recv")
            assert frame is not None
            assert frame["payload"] == "first"

    def test_next_ws_frame_returns_in_order(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-1", "wss://x", "recv", "one")
            f.record_ws_frame("conn-1", "wss://x", "recv", "two")
            f.record_ws_frame("conn-1", "wss://x", "recv", "three")

            payloads = [
                f.next_ws_frame("conn-1", "recv")["payload"]
                for _ in range(3)  # type: ignore[index]
            ]
            assert payloads == ["one", "two", "three"]

    def test_next_ws_frame_returns_none_when_exhausted(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-1", "wss://x", "recv", "only")
            f.next_ws_frame("conn-1", "recv")
            assert f.next_ws_frame("conn-1", "recv") is None

    def test_next_ws_frame_returns_none_for_unknown_connection(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            assert f.next_ws_frame("never-recorded", "recv") is None

    def test_send_and_recv_directions_have_independent_cursors(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-1", "wss://x", "send", "out-1")
            f.record_ws_frame("conn-1", "wss://x", "recv", "in-1")
            f.record_ws_frame("conn-1", "wss://x", "send", "out-2")

            assert f.next_ws_frame("conn-1", "send")["payload"] == "out-1"  # type: ignore[index]
            assert f.next_ws_frame("conn-1", "recv")["payload"] == "in-1"  # type: ignore[index]
            assert f.next_ws_frame("conn-1", "send")["payload"] == "out-2"  # type: ignore[index]

    def test_different_connections_do_not_interfere(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-A", "wss://x", "recv", "a-frame")
            f.record_ws_frame("conn-B", "wss://x", "recv", "b-frame")

            assert f.next_ws_frame("conn-A", "recv")["payload"] == "a-frame"  # type: ignore[index]
            assert f.next_ws_frame("conn-B", "recv")["payload"] == "b-frame"  # type: ignore[index]
            assert f.next_ws_frame("conn-A", "recv") is None

    def test_reset_ws_read_cursor_allows_replay_from_beginning(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-1", "wss://x", "recv", "replay-me")
            f.next_ws_frame("conn-1", "recv")
            assert f.next_ws_frame("conn-1", "recv") is None

            f.reset_ws_read_cursor()
            frame = f.next_ws_frame("conn-1", "recv")
            assert frame is not None
            assert frame["payload"] == "replay-me"

    def test_all_ws_frames_empty(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            assert f.all_ws_frames() == []

    def test_all_ws_frames_in_sequence_order_across_connections(
        self, tmp_path: Path
    ) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-A", "wss://x", "send", "a-out")
            f.record_ws_frame("conn-B", "wss://x", "send", "b-out")
            f.record_ws_frame("conn-A", "wss://x", "recv", "a-in")

            frames = f.all_ws_frames()
            assert [fr["payload"] for fr in frames] == ["a-out", "b-out", "a-in"]

    def test_all_ws_frames_filters_by_connection_id(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-A", "wss://x", "send", "a-out")
            f.record_ws_frame("conn-B", "wss://x", "send", "b-out")

            frames = f.all_ws_frames(connection_id="conn-A")
            assert len(frames) == 1
            assert frames[0]["payload"] == "a-out"

    def test_ws_frame_count_filters_by_connection_id(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-A", "wss://x", "send", "a-out")
            f.record_ws_frame("conn-A", "wss://x", "recv", "a-in")
            f.record_ws_frame("conn-B", "wss://x", "send", "b-out")

            assert f.ws_frame_count(connection_id="conn-A") == 2
            assert f.ws_frame_count(connection_id="conn-B") == 1
            assert f.ws_frame_count() == 3

    def test_default_frame_type_is_text(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame("conn-1", "wss://x", "send", "plain")
            assert f.all_ws_frames()[0]["frame_type"] == "text"

    def test_binary_frame_type_is_preserved(self, tmp_path: Path) -> None:
        with Fixture(tmp_path / "f.db") as f:
            f.record_ws_frame(
                "conn-1", "wss://x", "recv", "audio-bytes", frame_type="binary"
            )
            assert f.all_ws_frames()[0]["frame_type"] == "binary"

    def test_ws_frames_do_not_affect_http_exchange_count(self, tmp_path: Path) -> None:
        """ws_frames and http_exchanges are independent tables/sequences."""
        with Fixture(tmp_path / "f.db") as f:
            _record(f)
            f.record_ws_frame("conn-1", "wss://x", "send", "hello")
            assert f.exchange_count() == 1
            assert f.ws_frame_count() == 1


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
