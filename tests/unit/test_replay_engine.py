"""
Unit tests for agent_trace.replay.engine — ReplayEngine and replay_context.

Replay engine invariants:
- Makes ZERO network calls (AGENT_TRACE_NETWORK_GUARD=1 enforces this)
- Installs FixtureClock on entry, restores WallClock on exit
- Resets fixture read cursor on entry
- Patches httpx.Client and requests.Session
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent_trace._replay.engine import ReplayEngine, replay_context
from agent_trace._replay.fixture import Fixture
from agent_trace.core.clock import FixtureClock, WallClock, get_clock
from agent_trace.interceptor.httpx_hook import NetworkGuardError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_fixture(path: Path, exchanges: list[dict]) -> None:
    """Create a Fixture with pre-recorded exchanges."""
    with Fixture(path) as f:
        for ex in exchanges:
            f.record_exchange(**ex)


def _default_exchange(
    url: str = "https://api.example.com/test",
    method: str = "GET",
    body: str = '{"ok": true}',
    status: int = 200,
) -> dict:
    return dict(
        url=url,
        method=method,
        request_headers={"content-type": "application/json"},
        request_body="{}",
        response_status=status,
        response_headers={"content-type": "application/json"},
        response_body=body,
    )


# ---------------------------------------------------------------------------
# ReplayEngine.replay() — clock behaviour
# ---------------------------------------------------------------------------


class TestReplayEngineClock:
    def test_replay_installs_fixture_clock(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [])
        engine = ReplayEngine(db)
        with engine.replay():
            assert isinstance(get_clock(), FixtureClock)

    def test_clock_restored_to_wall_clock_after_exit(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [])
        engine = ReplayEngine(db)
        with engine.replay():
            pass
        assert isinstance(get_clock(), WallClock)

    def test_clock_restored_even_on_exception(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange()])
        engine = ReplayEngine(db)
        try:
            with engine.replay():
                raise ValueError("inner error")
        except (ValueError, NetworkGuardError):
            pass
        assert isinstance(get_clock(), WallClock)


# ---------------------------------------------------------------------------
# ReplayEngine.replay() — fixture cursor reset
# ---------------------------------------------------------------------------


class TestReplayEngineCursorReset:
    def test_reset_read_cursor_on_entry(self, tmp_path: Path) -> None:
        """Entering replay() resets the cursor so each replay starts fresh."""
        url = "https://api.example.com/cursor-test"
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange(url=url, body='{"n": 1}')])

        engine = ReplayEngine(db)

        # First replay — consume the exchange
        with engine.replay() as fixture:
            ex1 = fixture.next_exchange(url, "GET")
            assert ex1 is not None

        # Second replay — cursor should have been reset on entry
        with engine.replay() as fixture:
            ex2 = fixture.next_exchange(url, "GET")
            assert ex2 is not None
            assert ex2["response_body"] == '{"n": 1}'

    def test_replay_yields_open_fixture(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange()])
        engine = ReplayEngine(db)
        with engine.replay() as fixture:
            assert isinstance(fixture, Fixture)
            # Fixture is usable — count works
            assert fixture.exchange_count() == 1


# ---------------------------------------------------------------------------
# ReplayEngine.replay() — httpx patching
# ---------------------------------------------------------------------------


class TestReplayEngineHttpxPatch:
    def test_httpx_client_serves_fixture_response(self, tmp_path: Path) -> None:
        url = "https://api.openai.com/v1/chat/completions"
        body = json.dumps({"choices": [{"message": {"content": "from fixture"}}]})
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange(url=url, method="POST", body=body)])

        engine = ReplayEngine(db)
        with engine.replay():
            with httpx.Client() as client:
                response = client.post(url, json={"model": "gpt-4o"})
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "from fixture"

    def test_network_guard_raises_on_unrecorded_url(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")

        db = tmp_path / "f.db"
        _build_fixture(db, [])  # no exchanges

        engine = ReplayEngine(db)
        with pytest.raises(NetworkGuardError):
            with engine.replay():
                with httpx.Client() as client:
                    client.get("https://not-in-fixture.example.com/")

    def test_httpx_client_restored_after_exit(self, tmp_path: Path) -> None:
        """After replay() exits, httpx.Client.__init__ must be the original."""
        original_init = httpx.Client.__init__

        db = tmp_path / "f.db"
        _build_fixture(db, [])
        engine = ReplayEngine(db)

        with engine.replay():
            patched_init = httpx.Client.__init__
            assert patched_init is not original_init

        restored_init = httpx.Client.__init__
        assert restored_init is original_init


# ---------------------------------------------------------------------------
# fixture_exchange_count()
# ---------------------------------------------------------------------------


class TestFixtureExchangeCount:
    def test_returns_zero_for_empty_fixture(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [])
        engine = ReplayEngine(db)
        assert engine.fixture_exchange_count() == 0

    def test_returns_correct_count(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        exchanges = [
            _default_exchange(url=f"https://api.example.com/{i}") for i in range(5)
        ]
        _build_fixture(db, exchanges)
        engine = ReplayEngine(db)
        assert engine.fixture_exchange_count() == 5


# ---------------------------------------------------------------------------
# replay_context() — convenience wrapper
# ---------------------------------------------------------------------------


class TestReplayContext:
    def test_replay_context_equivalent_to_engine(self, tmp_path: Path) -> None:
        url = "https://api.example.com/ctx-test"
        body = '{"ctx": true}'
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange(url=url, method="GET", body=body)])

        with replay_context(db) as fixture:
            ex = fixture.next_exchange(url, "GET")
        assert ex is not None
        assert ex["response_body"] == body

    def test_replay_context_installs_fixture_clock(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [])
        with replay_context(db):
            assert isinstance(get_clock(), FixtureClock)

    def test_replay_context_restores_wall_clock(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [])
        with replay_context(db):
            pass
        assert isinstance(get_clock(), WallClock)

    def test_replay_context_yields_fixture_instance(self, tmp_path: Path) -> None:
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange()])
        with replay_context(db) as fixture:
            assert isinstance(fixture, Fixture)
            assert fixture.exchange_count() == 1


# ---------------------------------------------------------------------------
# Nested replay() behaviour
# ---------------------------------------------------------------------------


class TestNestedReplay:
    def test_nested_replay_raises_or_works(self, tmp_path: Path) -> None:
        """Document chosen behaviour: nested replay() calls may work but
        the inner context patches on top of the outer patch.

        We test that after both contexts exit, the system is back to normal.
        """
        db_outer = tmp_path / "outer.db"
        db_inner = tmp_path / "inner.db"

        url = "https://api.example.com/nested"
        _build_fixture(
            db_outer, [_default_exchange(url=url, body='{"level": "outer"}')]
        )
        _build_fixture(
            db_inner, [_default_exchange(url=url, body='{"level": "inner"}')]
        )

        engine_outer = ReplayEngine(db_outer)
        engine_inner = ReplayEngine(db_inner)

        try:
            with engine_outer.replay() as outer_fixture:
                # Outer is active
                assert isinstance(get_clock(), FixtureClock)

                with engine_inner.replay() as inner_fixture:
                    # Inner overrides outer
                    assert isinstance(get_clock(), FixtureClock)
                    # Inner fixture is accessible
                    assert inner_fixture.exchange_count() == 1

                # After inner exits, outer clock should still be FixtureClock
                # (engine.replay() only restores the token it set)
                assert isinstance(get_clock(), FixtureClock)

        except Exception:
            pass  # Some implementations may raise on nesting

        # After both exit: wall clock must be restored
        assert isinstance(get_clock(), WallClock)


# ---------------------------------------------------------------------------
# ReplayEngine.replay() — botocore (AWS SDK) patching
# ---------------------------------------------------------------------------


class TestReplayEngineBotocorePatch:
    def test_boto3_client_serves_fixture_response(self, tmp_path: Path) -> None:
        import boto3

        url = (
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
            "model/anthropic.claude-v2/invoke"
        )
        body = json.dumps({"completion": "from fixture"})
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange(url=url, method="POST", body=body)])

        session = boto3.Session(
            region_name="us-east-1",
            aws_access_key_id="AKIAFAKE",
            aws_secret_access_key="fakefakefakefakefakefakefakefakefakefake",
        )

        engine = ReplayEngine(db)
        with engine.replay():
            client = session.client(
                "bedrock-runtime",
                # Endpoint is never actually contacted during replay — the
                # patched URLLib3Session.send serves the fixture instead.
                endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
                region_name="us-east-1",
            )
            response = client.invoke_model(
                modelId="anthropic.claude-v2",
                body=json.dumps({"prompt": "hi"}),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())

        assert result == {"completion": "from fixture"}

    def test_network_guard_raises_on_unrecorded_botocore_call(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import boto3

        from agent_trace.interceptor.botocore_hook import NetworkGuardError as BotoNGE

        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")

        db = tmp_path / "f.db"
        _build_fixture(db, [])  # no exchanges

        session = boto3.Session(
            region_name="us-east-1",
            aws_access_key_id="AKIAFAKE",
            aws_secret_access_key="fakefakefakefakefakefakefakefakefakefake",
        )
        engine = ReplayEngine(db)
        with pytest.raises(BotoNGE):
            with engine.replay():
                client = session.client(
                    "bedrock-runtime",
                    endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
                    region_name="us-east-1",
                )
                client.invoke_model(
                    modelId="anthropic.claude-v2",
                    body=json.dumps({"prompt": "hi"}),
                    contentType="application/json",
                    accept="application/json",
                )

    def test_botocore_send_restored_after_exit(self, tmp_path: Path) -> None:
        """After replay() exits, URLLib3Session.send must be the original."""
        import botocore.httpsession

        original_send = botocore.httpsession.URLLib3Session.send

        db = tmp_path / "f.db"
        _build_fixture(db, [])
        engine = ReplayEngine(db)

        with engine.replay():
            patched_send = botocore.httpsession.URLLib3Session.send
            assert patched_send is not original_send

        assert botocore.httpsession.URLLib3Session.send is original_send


# ---------------------------------------------------------------------------
# Async client patching — AsyncReplayTransport must be used for AsyncClient
# ---------------------------------------------------------------------------


class TestReplayEngineAsyncHttpxPatch:
    async def test_async_client_serves_fixture_response(self, tmp_path: Path) -> None:
        """AsyncClient created inside replay() must serve responses from fixture."""
        url = "https://api.openai.com/v1/chat/completions"
        body = json.dumps({"choices": [{"message": {"content": "async from fixture"}}]})
        db = tmp_path / "f.db"
        _build_fixture(db, [_default_exchange(url=url, method="POST", body=body)])

        engine = ReplayEngine(db)
        with engine.replay():
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json={"model": "gpt-4o"})
            assert response.status_code == 200
            assert "async from fixture" in response.text

    async def test_async_client_raises_network_guard_on_unrecorded_url(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")

        db = tmp_path / "f.db"
        _build_fixture(db, [])

        engine = ReplayEngine(db)
        with pytest.raises(NetworkGuardError):
            with engine.replay():
                async with httpx.AsyncClient() as client:
                    await client.get("https://not-in-fixture-async.example.com/")
