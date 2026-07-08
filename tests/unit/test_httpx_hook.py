"""
Unit tests for agent_trace.interceptor.httpx_hook.

RecordingTransport / AsyncRecordingTransport / ReplayTransport / AsyncReplayTransport
/ NetworkGuardError.

AGENT_TRACE_NETWORK_GUARD=1 is set by pytest env (pyproject.toml), so
ReplayTransport will raise NetworkGuardError on any unmatched request.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.httpx_hook import (
    AsyncRecordingTransport,
    AsyncReplayTransport,
    NetworkGuardError,
    RecordingTransport,
    ReplayTransport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture(tmp_path, **kwargs) -> Fixture:
    return Fixture(tmp_path / "test.db", trace_id="test-trace")


def _record_one(
    fixture: Fixture, url: str, method: str, body: str, status: int = 200
) -> None:
    fixture.record_exchange(
        url=url,
        method=method,
        request_headers={"content-type": "application/json"},
        request_body="{}",
        response_status=status,
        response_headers={"content-type": "application/json"},
        response_body=body,
    )


# ---------------------------------------------------------------------------
# NetworkGuardError
# ---------------------------------------------------------------------------


class TestNetworkGuardError:
    def test_is_subclass_of_runtime_error(self) -> None:
        assert issubclass(NetworkGuardError, RuntimeError)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(NetworkGuardError):
            raise NetworkGuardError("guard triggered")

    def test_can_be_caught_as_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise NetworkGuardError("also a RuntimeError")


# ---------------------------------------------------------------------------
# RecordingTransport
# ---------------------------------------------------------------------------


class TestRecordingTransport:
    @respx.mock
    def test_records_get_request(self, tmp_path) -> None:
        url = "https://api.example.com/get-test"
        respx.get(url).mock(return_value=httpx.Response(200, json={"status": "ok"}))

        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        )
        with client:
            response = client.get(url)

        assert fixture.exchange_count() == 1
        exchanges = fixture.all_exchanges()
        assert exchanges[0]["url"] == url
        assert exchanges[0]["method"] == "GET"
        fixture.close()

    @respx.mock
    def test_records_post_request_with_json_body(self, tmp_path) -> None:
        url = "https://api.openai.com/v1/chat/completions"
        resp_json = {"choices": [{"message": {"content": "hello"}}]}
        respx.post(url).mock(return_value=httpx.Response(200, json=resp_json))

        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        )
        with client:
            client.post(url, json={"model": "gpt-4o", "messages": []})

        exchanges = fixture.all_exchanges()
        assert len(exchanges) == 1
        assert exchanges[0]["method"] == "POST"
        assert "gpt-4o" in exchanges[0]["request_body"]
        fixture.close()

    @respx.mock
    def test_preserves_response_status_code(self, tmp_path) -> None:
        url = "https://api.example.com/not-found"
        respx.get(url).mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )

        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        )
        with client:
            response = client.get(url)

        assert fixture.all_exchanges()[0]["response_status"] == 404
        assert response.status_code == 404
        fixture.close()

    @respx.mock
    def test_preserves_response_body(self, tmp_path) -> None:
        url = "https://api.example.com/body-test"
        resp_body = {"result": "important-data", "count": 42}
        respx.get(url).mock(return_value=httpx.Response(200, json=resp_body))

        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        )
        with client:
            response = client.get(url)

        recorded_body = fixture.all_exchanges()[0]["response_body"]
        recorded_data = json.loads(recorded_body)
        assert recorded_data["result"] == "important-data"
        assert recorded_data["count"] == 42
        fixture.close()

    @respx.mock
    def test_caller_receives_full_response(self, tmp_path) -> None:
        url = "https://api.example.com/resp-test"
        respx.get(url).mock(return_value=httpx.Response(201, json={"created": True}))

        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        )
        with client:
            response = client.get(url)

        assert response.status_code == 201
        assert response.json()["created"] is True
        fixture.close()


# ---------------------------------------------------------------------------
# RecordingTransport nesting / double-wrap safety
# ---------------------------------------------------------------------------


class TestRecordingTransportDoubleWrapSafety:
    """RecordingTransport must not raise or drop the response if it ends up
    wrapping another RecordingTransport as `inner` (defensive — this can
    happen if instrumentation layers overlap).  In practice
    ``Tracer._patch_httpx`` avoids this entirely with an
    ``isinstance(base_transport, RecordingTransport)`` guard before wrapping,
    so double-recording never happens through the normal Tracer patch path;
    this test only documents/locks in that the class itself stays safe.
    """

    @respx.mock
    def test_wrapping_an_already_recording_transport_still_works(
        self, tmp_path
    ) -> None:
        url = "https://api.example.com/double-wrap"
        respx.get(url).mock(return_value=httpx.Response(200, json={"ok": True}))

        fixture = _make_fixture(tmp_path)
        inner_transport = RecordingTransport(
            fixture, inner=httpx.MockTransport(respx.mock.handler)
        )
        outer_transport = RecordingTransport(fixture, inner=inner_transport)

        client = httpx.Client(transport=outer_transport)
        with client:
            response = client.get(url)

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        # Both layers recorded — documents current low-level behavior; the
        # Tracer-level patch avoids ever nesting RecordingTransports in
        # practice via its isinstance guard.
        assert fixture.exchange_count() == 2
        fixture.close()

    @respx.mock
    async def test_wrapping_an_already_recording_async_transport_still_works(
        self, tmp_path
    ) -> None:
        url = "https://api.example.com/async-double-wrap"
        respx.get(url).mock(return_value=httpx.Response(200, json={"ok": True}))

        fixture = _make_fixture(tmp_path)
        inner_transport = AsyncRecordingTransport(
            fixture, inner=httpx.MockTransport(respx.mock.handler)
        )
        outer_transport = AsyncRecordingTransport(fixture, inner=inner_transport)

        async with httpx.AsyncClient(transport=outer_transport) as client:
            response = await client.get(url)

        assert response.status_code == 200
        assert fixture.exchange_count() == 2
        fixture.close()


# ---------------------------------------------------------------------------
# ReplayTransport
# ---------------------------------------------------------------------------


class TestReplayTransport:
    def test_serves_recorded_exchange(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        _record_one(
            fixture, "https://api.example.com/replay", "GET", '{"replayed": true}'
        )

        transport = ReplayTransport(fixture)
        request = httpx.Request("GET", "https://api.example.com/replay")
        response = transport.handle_request(request)

        assert response.status_code == 200
        assert response.json() == {"replayed": True}
        fixture.close()

    def test_serves_correct_status_code(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        _record_one(
            fixture, "https://api.example.com/created", "POST", '{"id": 1}', status=201
        )

        transport = ReplayTransport(fixture)
        request = httpx.Request("POST", "https://api.example.com/created")
        response = transport.handle_request(request)

        assert response.status_code == 201
        fixture.close()

    def test_raises_network_guard_error_when_exchange_not_found(
        self, tmp_path, monkeypatch
    ) -> None:
        """ReplayTransport raises NetworkGuardError when AGENT_TRACE_NETWORK_GUARD=1."""
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")

        fixture = _make_fixture(tmp_path)
        transport = ReplayTransport(fixture)
        request = httpx.Request("GET", "https://never-recorded.example.com/")

        with pytest.raises(NetworkGuardError):
            transport.handle_request(request)

        fixture.close()

    def test_close_is_noop(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        transport = ReplayTransport(fixture)
        # Should not raise
        transport.close()
        fixture.close()

    def test_serves_exchanges_in_order_for_same_url(self, tmp_path) -> None:
        url = "https://api.example.com/seq"
        fixture = _make_fixture(tmp_path)
        for i in range(3):
            _record_one(fixture, url, "POST", f'{{"step": {i}}}')

        transport = ReplayTransport(fixture)
        for i in range(3):
            request = httpx.Request("POST", url)
            response = transport.handle_request(request)
            assert response.json()["step"] == i

        fixture.close()

    def test_guard_off_missing_exchange_falls_back(self, tmp_path, monkeypatch) -> None:
        """With guard=0 and no fixture entry, ReplayTransport warns and falls back.

        We use respx to intercept the real network call triggered by the fallback.
        """
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "0")

        url = "https://api.example.com/fallback-test"
        fixture = _make_fixture(tmp_path)
        transport = ReplayTransport(fixture)

        with respx.mock:
            respx.get(url).mock(
                return_value=httpx.Response(200, json={"fallback": True})
            )
            request = httpx.Request("GET", url)
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                response = transport.handle_request(request)
                # A warning should have been issued
                assert len(w) >= 1

        assert response.status_code == 200
        fixture.close()


# ---------------------------------------------------------------------------
# AsyncRecordingTransport
# ---------------------------------------------------------------------------


class TestAsyncRecordingTransport:
    @respx.mock
    async def test_records_get_request(self, tmp_path) -> None:
        url = "https://api.example.com/async-get"
        respx.get(url).mock(return_value=httpx.Response(200, json={"async": "ok"}))

        fixture = _make_fixture(tmp_path)
        async with httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        ) as client:
            response = await client.get(url)

        assert fixture.exchange_count() == 1
        exchanges = fixture.all_exchanges()
        assert exchanges[0]["url"] == url
        assert exchanges[0]["method"] == "GET"
        assert response.status_code == 200
        fixture.close()

    @respx.mock
    async def test_records_post_body_and_response(self, tmp_path) -> None:
        url = "https://api.openai.com/v1/chat/completions"
        resp_json = {"choices": [{"message": {"content": "async reply"}}]}
        respx.post(url).mock(return_value=httpx.Response(200, json=resp_json))

        fixture = _make_fixture(tmp_path)
        async with httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        ) as client:
            await client.post(url, json={"model": "gpt-4o", "messages": []})

        ex = fixture.all_exchanges()[0]
        assert ex["method"] == "POST"
        assert "gpt-4o" in ex["request_body"]
        assert "async reply" in ex["response_body"]
        fixture.close()

    @respx.mock
    async def test_caller_receives_correct_response(self, tmp_path) -> None:
        url = "https://api.example.com/async-status"
        respx.get(url).mock(return_value=httpx.Response(201, json={"created": True}))

        fixture = _make_fixture(tmp_path)
        async with httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture,
                inner=httpx.MockTransport(respx.mock.handler),
            )
        ) as client:
            response = await client.get(url)

        assert response.status_code == 201
        assert response.json()["created"] is True
        fixture.close()


# ---------------------------------------------------------------------------
# AsyncReplayTransport
# ---------------------------------------------------------------------------


class TestAsyncReplayTransport:
    async def test_serves_recorded_exchange(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        _record_one(
            fixture, "https://api.example.com/async-replay", "GET", '{"async": true}'
        )

        transport = AsyncReplayTransport(fixture)
        request = httpx.Request("GET", "https://api.example.com/async-replay")
        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert response.json() == {"async": True}
        fixture.close()

    async def test_serves_correct_status_code(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        _record_one(
            fixture,
            "https://api.example.com/async-created",
            "POST",
            '{"id": 1}',
            status=201,
        )

        transport = AsyncReplayTransport(fixture)
        request = httpx.Request("POST", "https://api.example.com/async-created")
        response = await transport.handle_async_request(request)

        assert response.status_code == 201
        fixture.close()

    async def test_raises_network_guard_error_when_exchange_not_found(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")

        fixture = _make_fixture(tmp_path)
        transport = AsyncReplayTransport(fixture)
        request = httpx.Request("GET", "https://never-recorded-async.example.com/")

        with pytest.raises(NetworkGuardError):
            await transport.handle_async_request(request)

        fixture.close()

    async def test_aclose_is_noop(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        transport = AsyncReplayTransport(fixture)
        await transport.aclose()  # must not raise
        fixture.close()

    async def test_serves_exchanges_in_order_for_same_url(self, tmp_path) -> None:
        url = "https://api.example.com/async-seq"
        fixture = _make_fixture(tmp_path)
        for i in range(3):
            _record_one(fixture, url, "POST", f'{{"step": {i}}}')

        transport = AsyncReplayTransport(fixture)
        for i in range(3):
            request = httpx.Request("POST", url)
            response = await transport.handle_async_request(request)
            assert response.json()["step"] == i

        fixture.close()

    async def test_async_client_uses_async_replay_transport(
        self, tmp_path, monkeypatch
    ) -> None:
        """AsyncClient patched with AsyncReplayTransport must serve from fixture."""
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")
        url = "https://api.openai.com/v1/chat/completions"
        body = '{"choices": [{"message": {"content": "from async fixture"}}]}'

        fixture = _make_fixture(tmp_path)
        _record_one(fixture, url, "POST", body)

        async with httpx.AsyncClient(transport=AsyncReplayTransport(fixture)) as client:
            response = await client.post(url, json={"model": "gpt-4o"})

        assert response.status_code == 200
        assert "from async fixture" in response.text
        fixture.close()
