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


class TestRecordingTransportDurationAndFailure:
    """duration_ms capture + failed-before-response persistence."""

    @respx.mock
    def test_records_duration_ms(self, tmp_path) -> None:
        url = "https://api.example.com/duration-test"
        respx.get(url).mock(return_value=httpx.Response(200, json={"ok": True}))

        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=httpx.MockTransport(respx.mock.handler)
            )
        )
        with client:
            client.get(url)

        ex = fixture.all_exchanges()[0]
        assert ex["duration_ms"] is not None
        assert ex["duration_ms"] >= 0
        fixture.close()

    def test_persists_failed_attempt_before_raising(self, tmp_path) -> None:
        """A pre-response exception (connection refused, DNS failure, ...)
        must still be persisted — as a failed-before-response exchange, not
        silently lost — and the original exception must still propagate."""

        def _raising_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        fixture = _make_fixture(tmp_path)
        url = "https://bad-host.invalid/x"
        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=httpx.MockTransport(_raising_handler)
            )
        )
        with client:
            with pytest.raises(httpx.ConnectError):
                client.post(url, json={"model": "gpt-4o"})

        assert fixture.exchange_count() == 1
        assert fixture.failed_exchange_count() == 1
        ex = fixture.all_exchanges()[0]
        assert ex["response_status"] is None
        assert ex["error_type"] == "ConnectError"
        assert "Connection refused" in ex["error_message"]
        assert ex["method"] == "POST"
        assert "gpt-4o" in ex["request_body"]
        fixture.close()

    def test_failed_attempt_records_duration_ms_too(self, tmp_path) -> None:
        def _raising_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out", request=request)

        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=httpx.MockTransport(_raising_handler)
            )
        )
        with client:
            with pytest.raises(httpx.ConnectTimeout):
                client.get("https://bad-host.invalid/x")

        ex = fixture.all_exchanges()[0]
        assert ex["duration_ms"] is not None
        assert ex["duration_ms"] >= 0
        fixture.close()


class TestAsyncRecordingTransportDurationAndFailure:
    """Async equivalents of TestRecordingTransportDurationAndFailure."""

    @respx.mock
    async def test_records_duration_ms(self, tmp_path) -> None:
        url = "https://api.example.com/async-duration-test"
        respx.get(url).mock(return_value=httpx.Response(200, json={"ok": True}))

        fixture = _make_fixture(tmp_path)
        async with httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture, inner=httpx.MockTransport(respx.mock.handler)
            )
        ) as client:
            await client.get(url)

        ex = fixture.all_exchanges()[0]
        assert ex["duration_ms"] is not None
        assert ex["duration_ms"] >= 0
        fixture.close()

    async def test_persists_failed_attempt_before_raising(self, tmp_path) -> None:
        async def _raising_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        fixture = _make_fixture(tmp_path)
        url = "https://bad-host.invalid/async-x"
        async with httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture, inner=httpx.MockTransport(_raising_handler)
            )
        ) as client:
            with pytest.raises(httpx.ConnectError):
                await client.post(url, json={"model": "gpt-4o"})

        assert fixture.exchange_count() == 1
        assert fixture.failed_exchange_count() == 1
        ex = fixture.all_exchanges()[0]
        assert ex["response_status"] is None
        assert ex["error_type"] == "ConnectError"
        assert "Connection refused" in ex["error_message"]
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


# ---------------------------------------------------------------------------
# Non-buffering / pass-through streaming capture mode (stream=True)
# ---------------------------------------------------------------------------
#
# respx-mocked responses are always fully-buffered (a single content= chunk),
# which doesn't exercise a genuine multi-chunk pass-through. These tests use
# a hand-rolled inner transport whose response stream yields multiple
# distinct chunks, matching how a real streamed/SSE HTTP response arrives.

_SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    b"data: [DONE]\n\n",
]


class _FakeSyncStream(httpx.SyncByteStream):
    def __iter__(self):
        yield from _SSE_CHUNKS

    def close(self) -> None:
        pass


class _FakeSyncInnerTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeSyncStream(),
            request=request,
        )

    def close(self) -> None:
        pass


class _FakeAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        for chunk in _SSE_CHUNKS:
            yield chunk

    async def aclose(self) -> None:
        pass


class _FakeAsyncInnerTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeAsyncStream(),
            request=request,
        )

    async def aclose(self) -> None:
        pass


class TestRecordingTransportStreamMode:
    def test_default_mode_unaffected(self, tmp_path) -> None:
        """stream defaults to False — pre-existing eager-buffering behavior
        is unchanged when the flag isn't passed."""
        fixture = _make_fixture(tmp_path)
        transport = RecordingTransport(fixture, inner=_FakeSyncInnerTransport())
        assert transport._stream is False

    def test_caller_receives_every_chunk(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=_FakeSyncInnerTransport(), stream=True
            )
        )
        received: list[bytes] = []
        with client, client.stream("GET", "https://api.example.com/stream") as resp:
            for chunk in resp.iter_bytes():
                received.append(chunk)
        assert received == _SSE_CHUNKS

    def test_fixture_records_full_body_and_chunk_timestamps(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=_FakeSyncInnerTransport(), stream=True
            )
        )
        with client, client.stream("GET", "https://api.example.com/stream") as resp:
            resp.read()

        assert fixture.exchange_count() == 1
        exchange = fixture.all_exchanges()[0]
        assert exchange["response_body"] == b"".join(_SSE_CHUNKS).decode("utf-8")
        assert exchange["response_status"] == 200
        timestamps = exchange["chunk_timestamps"]
        assert timestamps is not None
        assert len(timestamps) == len(_SSE_CHUNKS)
        # Arrival offsets must be non-decreasing (each chunk arrives no
        # earlier than the one before it).
        assert timestamps == sorted(timestamps)

    def test_non_streaming_exchange_has_no_chunk_timestamps(self, tmp_path) -> None:
        """An exchange recorded the default (stream=False) way must not
        carry chunk_timestamps — absence means "not captured this way"."""
        fixture = _make_fixture(tmp_path)
        _record_one(fixture, "https://api.example.com/x", "GET", "{}")
        exchange = fixture.all_exchanges()[0]
        assert exchange["chunk_timestamps"] is None


class TestAsyncRecordingTransportStreamMode:
    async def test_caller_receives_every_chunk(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture, inner=_FakeAsyncInnerTransport(), stream=True
            )
        )
        received: list[bytes] = []
        async with client:
            async with client.stream("GET", "https://api.example.com/stream") as resp:
                async for chunk in resp.aiter_bytes():
                    received.append(chunk)
        assert received == _SSE_CHUNKS

    async def test_fixture_records_full_body_and_chunk_timestamps(
        self, tmp_path
    ) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture, inner=_FakeAsyncInnerTransport(), stream=True
            )
        )
        async with client:
            async with client.stream("GET", "https://api.example.com/stream") as resp:
                await resp.aread()

        assert fixture.exchange_count() == 1
        exchange = fixture.all_exchanges()[0]
        assert exchange["response_body"] == b"".join(_SSE_CHUNKS).decode("utf-8")
        assert exchange["chunk_timestamps"] is not None
        assert len(exchange["chunk_timestamps"]) == len(_SSE_CHUNKS)
