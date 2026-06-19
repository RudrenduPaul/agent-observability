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

from agent_trace.interceptor.requests_patch import (
    NetworkGuardError,
    RecordingAdapter,
    ReplayAdapter,
)
from agent_trace.replay.fixture import Fixture

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
