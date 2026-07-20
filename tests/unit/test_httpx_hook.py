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
from agent_trace.core.exceptions import RunawayToolCallLoopError
from agent_trace.interceptor.httpx_hook import (
    AsyncRecordingTransport,
    AsyncReplayTransport,
    NetworkGuardError,
    RecordingTransport,
    ReplayTransport,
    _is_tool_call_only_response,
    correlation_context,
    current_correlation_id,
    pop_correlation_id,
    push_correlation_id,
    raise_on_loop_detected,
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


# ---------------------------------------------------------------------------
# Runaway-tool-call-loop guard — RecordingTransport(loop_guard_threshold=...)
# ---------------------------------------------------------------------------

_TOOL_CALL_ONLY_BODY = json.dumps(
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
)

_FINAL_ANSWER_BODY = json.dumps(
    {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Here's your answer."},
                "finish_reason": "stop",
            }
        ]
    }
)

_ANTHROPIC_TOOL_USE_BODY = json.dumps(
    {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "name": "search", "input": {}}],
    }
)

_ANTHROPIC_TEXT_BODY = json.dumps(
    {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Here you go."}],
    }
)


class TestIsToolCallOnlyResponse:
    def test_openai_style_tool_call_only_is_flagged(self) -> None:
        assert _is_tool_call_only_response(_TOOL_CALL_ONLY_BODY) is True

    def test_openai_style_final_answer_is_not_flagged(self) -> None:
        assert _is_tool_call_only_response(_FINAL_ANSWER_BODY) is False

    def test_anthropic_tool_use_is_flagged(self) -> None:
        assert _is_tool_call_only_response(_ANTHROPIC_TOOL_USE_BODY) is True

    def test_anthropic_text_response_is_not_flagged(self) -> None:
        assert _is_tool_call_only_response(_ANTHROPIC_TEXT_BODY) is False

    def test_malformed_json_is_not_flagged(self) -> None:
        assert _is_tool_call_only_response("not json{{{") is False

    def test_non_dict_json_is_not_flagged(self) -> None:
        assert _is_tool_call_only_response("[1, 2, 3]") is False

    def test_empty_choices_is_not_flagged(self) -> None:
        assert _is_tool_call_only_response(json.dumps({"choices": []})) is False


def _tool_call_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_TOOL_CALL_ONLY_BODY)


def _final_answer_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_FINAL_ANSWER_BODY)


class TestRecordingTransportLoopGuardDisabledByDefault:
    def test_no_warning_without_threshold(self, tmp_path, recwarn) -> None:
        """loop_guard_threshold=None (the default) must never warn/raise,
        no matter how many consecutive tool-call-only responses occur."""
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=httpx.MockTransport(_tool_call_handler)
            )
        )
        with client:
            for _ in range(10):
                client.get("https://api.example.com/chat")

        assert fixture.exchange_count() == 10
        assert len(recwarn) == 0


class TestRecordingTransportLoopGuardWarns:
    def test_warns_once_threshold_reached(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(_tool_call_handler),
                loop_guard_threshold=3,
            )
        )
        with client:
            with pytest.warns(UserWarning, match="runaway tool-call loop"):
                for _ in range(3):
                    client.get("https://api.example.com/chat")

        # All requests are still recorded — the default handler only warns.
        assert fixture.exchange_count() == 3

    def test_no_warning_below_threshold(self, tmp_path, recwarn) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(_tool_call_handler),
                loop_guard_threshold=5,
            )
        )
        with client:
            for _ in range(4):
                client.get("https://api.example.com/chat")

        assert len(recwarn) == 0

    def test_final_answer_resets_the_consecutive_count(self, tmp_path, recwarn) -> None:
        """A tool-call-free response in between must reset the streak, not
        merely pause it — 2 tool-call-only, 1 final answer, 2 more
        tool-call-only must not trigger a threshold=3 guard."""
        fixture = _make_fixture(tmp_path)

        responses = iter(
            [
                _tool_call_handler,
                _tool_call_handler,
                _final_answer_handler,
                _tool_call_handler,
                _tool_call_handler,
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return next(responses)(request)

        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=httpx.MockTransport(handler), loop_guard_threshold=3
            )
        )
        with client:
            for _ in range(5):
                client.get("https://api.example.com/chat")

        assert len(recwarn) == 0

    def test_different_hosts_counted_independently(self, tmp_path, recwarn) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(_tool_call_handler),
                loop_guard_threshold=3,
            )
        )
        with client:
            client.get("https://api.example.com/chat")
            client.get("https://api.other.com/chat")
            client.get("https://api.example.com/chat")
            client.get("https://api.other.com/chat")

        # 2 consecutive per host — below threshold=3 for either host.
        assert len(recwarn) == 0


class TestRecordingTransportLoopGuardRaises:
    def test_raise_on_loop_detected_aborts_the_run(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture,
                inner=httpx.MockTransport(_tool_call_handler),
                loop_guard_threshold=2,
                on_loop_detected=raise_on_loop_detected,
            )
        )
        with client:
            client.get("https://api.example.com/chat")
            with pytest.raises(RunawayToolCallLoopError, match="2 consecutive"):
                client.get("https://api.example.com/chat")

        # The exchange that tripped the guard is still recorded before the
        # exception propagates — the guard aborts the *run*, not the record.
        assert fixture.exchange_count() == 2


class TestAsyncRecordingTransportLoopGuard:
    async def test_warns_once_threshold_reached(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture,
                inner=httpx.MockTransport(_tool_call_handler),
                loop_guard_threshold=2,
            )
        )
        async with client:
            with pytest.warns(UserWarning, match="runaway tool-call loop"):
                for _ in range(2):
                    await client.get("https://api.example.com/chat")

        assert fixture.exchange_count() == 2

    async def test_raise_on_loop_detected_aborts_the_run(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture,
                inner=httpx.MockTransport(_tool_call_handler),
                loop_guard_threshold=2,
                on_loop_detected=raise_on_loop_detected,
            )
        )
        async with client:
            await client.get("https://api.example.com/chat")
            with pytest.raises(RunawayToolCallLoopError, match="2 consecutive"):
                await client.get("https://api.example.com/chat")

        assert fixture.exchange_count() == 2


# ---------------------------------------------------------------------------
# correlation_context() / current_correlation_id() — ties a recorded
# exchange back to the concurrent batch input or graph node that produced
# it (#30924, #6037).
# ---------------------------------------------------------------------------


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


class TestCorrelationContext:
    def test_no_active_context_returns_none(self) -> None:
        assert current_correlation_id() is None

    def test_inside_context_returns_set_value(self) -> None:
        with correlation_context("batch-0"):
            assert current_correlation_id() == "batch-0"
        assert current_correlation_id() is None

    def test_nested_context_restores_outer_value_on_exit(self) -> None:
        with correlation_context("outer"):
            with correlation_context("inner"):
                assert current_correlation_id() == "inner"
            assert current_correlation_id() == "outer"
        assert current_correlation_id() is None

    def test_exception_inside_context_still_restores(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            with correlation_context("batch-0"):
                raise ValueError("boom")
        assert current_correlation_id() is None


class TestPushPopCorrelationId:
    """Manual set()/reset() counterpart to correlation_context() — used by
    LangGraphTracer (#6037), which must push in one callback (on_X_start)
    and pop in a separate one (on_X_end) rather than within a single `with`
    block."""

    def test_push_sets_current_correlation_id(self) -> None:
        assert current_correlation_id() is None
        token = push_correlation_id("span-abc")
        try:
            assert current_correlation_id() == "span-abc"
        finally:
            pop_correlation_id(token)

    def test_pop_restores_previous_value(self) -> None:
        assert current_correlation_id() is None
        token = push_correlation_id("span-abc")
        pop_correlation_id(token)
        assert current_correlation_id() is None

    def test_nested_push_pop_restores_outer_value(self) -> None:
        outer_token = push_correlation_id("outer-span")
        inner_token = push_correlation_id("inner-span")
        try:
            assert current_correlation_id() == "inner-span"
        finally:
            pop_correlation_id(inner_token)
        assert current_correlation_id() == "outer-span"
        pop_correlation_id(outer_token)
        assert current_correlation_id() is None

    def test_push_pop_interoperates_with_correlation_context(self) -> None:
        """push_correlation_id/correlation_context share the same
        underlying contextvar — nesting either inside the other must
        compose correctly."""
        with correlation_context("batch-item-0"):
            token = push_correlation_id("node-span-id")
            try:
                assert current_correlation_id() == "node-span-id"
            finally:
                pop_correlation_id(token)
            assert current_correlation_id() == "batch-item-0"


class TestRecordingTransportCorrelationId:
    @respx.mock
    def test_correlation_id_persisted_when_context_active(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        respx.get("https://api.example.com/a").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        client = httpx.Client(transport=RecordingTransport(fixture))
        with correlation_context("batch-item-0"):
            client.get("https://api.example.com/a")

        exchange = fixture.all_exchanges()[0]
        assert exchange["correlation_id"] == "batch-item-0"

    @respx.mock
    def test_no_correlation_id_when_no_context_active(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        respx.get("https://api.example.com/a").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        client = httpx.Client(transport=RecordingTransport(fixture))
        client.get("https://api.example.com/a")

        exchange = fixture.all_exchanges()[0]
        assert exchange["correlation_id"] is None

    def test_correlation_id_persisted_via_mock_transport(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        client = httpx.Client(
            transport=RecordingTransport(
                fixture, inner=httpx.MockTransport(_ok_handler)
            )
        )
        with correlation_context("batch-item-1"):
            client.get("https://api.example.com/b")

        exchange = fixture.all_exchanges()[0]
        assert exchange["correlation_id"] == "batch-item-1"


class TestAsyncRecordingTransportCorrelationId:
    async def test_correlation_id_isolated_per_concurrent_task(self, tmp_path) -> None:
        import asyncio

        fixture = _make_fixture(tmp_path)
        client = httpx.AsyncClient(
            transport=AsyncRecordingTransport(
                fixture, inner=httpx.MockTransport(_ok_handler)
            )
        )

        async def _call(correlation_id: str, url: str) -> None:
            with correlation_context(correlation_id):
                await asyncio.sleep(0)  # yield control, prove isolation across tasks
                await client.get(url)

        async with client:
            await asyncio.gather(
                _call("batch-0", "https://api.example.com/x"),
                _call("batch-1", "https://api.example.com/y"),
            )

        by_url = {e["url"]: e["correlation_id"] for e in fixture.all_exchanges()}
        assert by_url["https://api.example.com/x"] == "batch-0"
        assert by_url["https://api.example.com/y"] == "batch-1"
