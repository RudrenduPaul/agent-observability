"""
Unit tests for agent_trace.interceptor.requests_patch.

RecordingAdapter / ReplayAdapter / NetworkGuardError.
Uses unittest.mock to avoid real HTTP calls.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from requests import PreparedRequest, Response
from requests.adapters import HTTPAdapter

from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.requests_patch import (
    NetworkGuardError,
    RecordingAdapter,
    ReplayAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture(tmp_path) -> Fixture:
    return Fixture(tmp_path / "req_test.db", trace_id="test-req-trace")


def _make_prepared_request(
    url: str, method: str = "POST", body: str = "{}"
) -> PreparedRequest:
    req = PreparedRequest()
    req.prepare_method(method)
    req.prepare_url(url, {})
    req.prepare_headers({"content-type": "application/json"})
    req.prepare_body(body, None, None)
    return req


def _make_mock_response(status_code: int = 200, body: str = '{"ok": true}') -> Response:
    resp = Response()
    resp.status_code = status_code
    resp.headers.update({"content-type": "application/json"})
    resp._content = body.encode("utf-8")
    resp.encoding = "utf-8"
    resp.url = "https://api.example.com/mock"
    resp.raw = io.BytesIO(resp._content)
    return resp


def _record_one(
    fixture: Fixture,
    url: str = "https://api.example.com/v1/test",
    method: str = "POST",
    body: str = '{"recorded": true}',
    status: int = 200,
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
# RecordingAdapter
# ---------------------------------------------------------------------------


class TestRecordingAdapter:
    def test_inner_adapter_is_used_when_provided(self, tmp_path) -> None:
        """RecordingAdapter must delegate to the inner adapter, not super().send()."""
        from unittest.mock import MagicMock

        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/inner-test"
        req = _make_prepared_request(url, "POST")
        mock_response = _make_mock_response(200, '{"from_inner": true}')

        inner = MagicMock(spec=HTTPAdapter)
        inner.send.return_value = mock_response

        adapter = RecordingAdapter(fixture, inner=inner)
        with patch.object(HTTPAdapter, "send") as base_send:
            response = adapter.send(req)
            # super().send() must NOT be called when inner is provided
            base_send.assert_not_called()

        # Inner adapter was used
        inner.send.assert_called_once()
        assert response is mock_response
        # Exchange was still recorded
        assert fixture.exchange_count() == 1
        fixture.close()

    def test_no_inner_uses_super_send(self, tmp_path) -> None:
        """Without inner, RecordingAdapter falls back to super().send()."""
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/no-inner"
        req = _make_prepared_request(url, "GET")
        mock_response = _make_mock_response(200, '{"ok": true}')

        adapter = RecordingAdapter(fixture)  # no inner
        with patch.object(HTTPAdapter, "send", return_value=mock_response):
            response = adapter.send(req)

        assert response is mock_response
        assert fixture.exchange_count() == 1
        fixture.close()

    def test_send_records_exchange(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/record-test"
        req = _make_prepared_request(url, "POST")
        mock_response = _make_mock_response(200, '{"recorded": true}')

        adapter = RecordingAdapter(fixture)
        with patch.object(HTTPAdapter, "send", return_value=mock_response):
            adapter.send(req)

        assert fixture.exchange_count() == 1
        ex = fixture.all_exchanges()[0]
        assert ex["url"] == url
        assert ex["method"] == "POST"
        fixture.close()

    def test_send_preserves_response_body(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/body-test"
        req = _make_prepared_request(url, "GET")
        body_content = '{"data": "important", "count": 42}'
        mock_response = _make_mock_response(200, body_content)

        adapter = RecordingAdapter(fixture)
        with patch.object(HTTPAdapter, "send", return_value=mock_response):
            response = adapter.send(req)

        ex = fixture.all_exchanges()[0]
        # Body stored in fixture
        assert ex["response_body"] == body_content
        # Original response returned unchanged
        assert response.text == body_content
        fixture.close()

    def test_send_records_status_code(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/status-test"
        req = _make_prepared_request(url, "GET")
        mock_response = _make_mock_response(404, "not found")

        adapter = RecordingAdapter(fixture)
        with patch.object(HTTPAdapter, "send", return_value=mock_response):
            adapter.send(req)

        ex = fixture.all_exchanges()[0]
        assert ex["response_status"] == 404
        fixture.close()

    def test_send_returns_original_response(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/passthrough"
        req = _make_prepared_request(url, "POST")
        mock_response = _make_mock_response(201, '{"created": true}')

        adapter = RecordingAdapter(fixture)
        with patch.object(HTTPAdapter, "send", return_value=mock_response):
            response = adapter.send(req)

        assert response is mock_response
        assert response.status_code == 201
        fixture.close()


class TestRecordingAdapterDurationAndFailure:
    """duration_ms capture + failed-before-response persistence."""

    def test_send_records_duration_ms(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/duration-test"
        req = _make_prepared_request(url, "GET")
        mock_response = _make_mock_response(200, '{"ok": true}')

        adapter = RecordingAdapter(fixture)
        with patch.object(HTTPAdapter, "send", return_value=mock_response):
            adapter.send(req)

        ex = fixture.all_exchanges()[0]
        assert ex["duration_ms"] is not None
        assert ex["duration_ms"] >= 0
        fixture.close()

    def test_send_persists_failed_attempt_before_raising(self, tmp_path) -> None:
        """A connection-level exception (raised before any Response exists)
        must still be persisted — as a failed-before-response exchange, not
        silently lost — and the original exception must still propagate."""
        import requests

        fixture = _make_fixture(tmp_path)
        url = "https://bad-host.invalid/x"
        req = _make_prepared_request(url, "POST", body='{"model": "gpt-4o"}')

        adapter = RecordingAdapter(fixture)
        with patch.object(
            HTTPAdapter,
            "send",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ):
            with pytest.raises(requests.exceptions.ConnectionError):
                adapter.send(req)

        assert fixture.exchange_count() == 1
        assert fixture.failed_exchange_count() == 1
        ex = fixture.all_exchanges()[0]
        assert ex["response_status"] is None
        assert ex["error_type"] == "ConnectionError"
        assert "Connection refused" in ex["error_message"]
        assert ex["url"] == url
        assert ex["method"] == "POST"
        # Request body was still captured — it was constructed before the
        # failure, so it's available regardless of whether the call succeeded.
        assert "gpt-4o" in ex["request_body"]
        fixture.close()

    def test_failed_attempt_records_duration_ms_too(self, tmp_path) -> None:
        import requests

        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request("https://bad-host.invalid/x", "GET")

        adapter = RecordingAdapter(fixture)
        with patch.object(
            HTTPAdapter, "send", side_effect=requests.exceptions.Timeout("timed out")
        ):
            with pytest.raises(requests.exceptions.Timeout):
                adapter.send(req)

        ex = fixture.all_exchanges()[0]
        assert ex["duration_ms"] is not None
        assert ex["duration_ms"] >= 0
        fixture.close()


# ---------------------------------------------------------------------------
# RecordingAdapter nesting / double-wrap safety
# ---------------------------------------------------------------------------


class TestRecordingAdapterDoubleWrapSafety:
    """RecordingAdapter must not raise or drop the response if it ends up
    wrapping another RecordingAdapter as `inner` (defensive — mirrors
    TestRecordingTransportDoubleWrapSafety in test_httpx_hook.py). In
    practice ``Tracer._patch_requests`` avoids this with an
    ``isinstance(inner, RecordingAdapter)`` guard before wrapping.
    """

    def test_wrapping_an_already_recording_adapter_still_works(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/double-wrap"
        req = _make_prepared_request(url, "GET")
        mock_response = _make_mock_response(200, '{"ok": true}')

        with patch.object(HTTPAdapter, "send", return_value=mock_response):
            inner_adapter = RecordingAdapter(fixture)
            outer_adapter = RecordingAdapter(fixture, inner=inner_adapter)
            response = outer_adapter.send(req)

        assert response is mock_response
        # Both layers recorded — documents current low-level behavior; the
        # Tracer-level patch avoids ever nesting RecordingAdapters in
        # practice via its isinstance guard.
        assert fixture.exchange_count() == 2
        fixture.close()


# ---------------------------------------------------------------------------
# ReplayAdapter
# ---------------------------------------------------------------------------


class TestReplayAdapter:
    def test_send_returns_recorded_response(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/replay-test"
        _record_one(fixture, url=url, method="POST", body='{"replayed": true}')

        adapter = ReplayAdapter(fixture)
        req = _make_prepared_request(url, "POST")
        response = adapter.send(req)

        assert response.status_code == 200
        assert response.json() == {"replayed": True}
        fixture.close()

    def test_send_returns_correct_status_code(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/status-replay"
        _record_one(fixture, url=url, method="GET", body="created", status=201)

        adapter = ReplayAdapter(fixture)
        req = _make_prepared_request(url, "GET")
        response = adapter.send(req)

        assert response.status_code == 201
        fixture.close()

    def test_send_returns_response_body_as_bytes(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/bytes-test"
        _record_one(fixture, url=url, method="GET", body='{"bytes": "value"}')

        adapter = ReplayAdapter(fixture)
        req = _make_prepared_request(url, "GET")
        response = adapter.send(req)

        # _content must be bytes
        assert isinstance(response.content, bytes)
        assert response.content == b'{"bytes": "value"}'
        fixture.close()

    def test_send_raises_network_guard_error_when_not_found(
        self, tmp_path, monkeypatch
    ) -> None:
        """ReplayAdapter raises NetworkGuardError when AGENT_TRACE_NETWORK_GUARD=1."""
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")

        fixture = _make_fixture(tmp_path)
        adapter = ReplayAdapter(fixture)
        req = _make_prepared_request(
            "https://not-recorded.example.com/endpoint", "POST"
        )

        with pytest.raises(NetworkGuardError):
            adapter.send(req)

        fixture.close()

    def test_send_fallback_uses_http_adapter_when_guard_off(
        self, tmp_path, monkeypatch
    ) -> None:
        """With guard=0 and no fixture entry, ReplayAdapter falls back to HTTPAdapter.

        Previously the fallback called super().send() which is BaseAdapter.send(),
        an abstract method that raises NotImplementedError unconditionally.
        """
        monkeypatch.delenv("AGENT_TRACE_NETWORK_GUARD", raising=False)

        fixture = _make_fixture(tmp_path)
        adapter = ReplayAdapter(fixture)
        req = _make_prepared_request("https://not-recorded.example.com/fallback", "GET")

        mock_response = _make_mock_response(200, '{"fallback": true}')
        with patch(
            "agent_trace.interceptor.requests_patch.HTTPAdapter.send",
            return_value=mock_response,
        ):
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                response = adapter.send(req)
            assert len(w) == 1
            assert "no fixture entry" in str(w[0].message)

        assert response.status_code == 200
        fixture.close()

    def test_send_serves_multiple_exchanges_in_order(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://api.example.com/multi"
        for i in range(3):
            _record_one(fixture, url=url, method="POST", body=f'{{"index": {i}}}')

        adapter = ReplayAdapter(fixture)
        for i in range(3):
            req = _make_prepared_request(url, "POST")
            response = adapter.send(req)
            assert response.json()["index"] == i

        fixture.close()

    def test_send_independent_cursors_per_url(self, tmp_path) -> None:
        """Different URLs maintain independent read cursors."""
        fixture = _make_fixture(tmp_path)
        url_a = "https://api.example.com/a"
        url_b = "https://api.example.com/b"
        _record_one(fixture, url=url_a, method="GET", body='{"url": "a"}')
        _record_one(fixture, url=url_b, method="GET", body='{"url": "b"}')

        adapter = ReplayAdapter(fixture)
        resp_a = adapter.send(_make_prepared_request(url_a, "GET"))
        resp_b = adapter.send(_make_prepared_request(url_b, "GET"))

        assert resp_a.json()["url"] == "a"
        assert resp_b.json()["url"] == "b"
        fixture.close()
