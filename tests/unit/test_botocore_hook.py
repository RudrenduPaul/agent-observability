"""
Unit tests for agent_trace.interceptor.botocore_hook.

RecordingSession / ReplaySession / NetworkGuardError, plus an end-to-end
test that a real boto3 client (bedrock-runtime) is captured through
Tracer.start_trace(record=True) against a local loopback HTTP server — no
real AWS calls, no network access beyond 127.0.0.1.

AGENT_TRACE_NETWORK_GUARD=1 is set by pytest env (pyproject.toml), so
ReplaySession will raise NetworkGuardError on any unmatched request.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import boto3
import botocore.awsrequest
import botocore.httpsession
import botocore.response
import pytest

from agent_trace import Tracer
from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.botocore_hook import (
    NetworkGuardError,
    RecordingSession,
    ReplaySession,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BEDROCK_URL = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke"
)


def _make_fixture(tmp_path) -> Fixture:
    return Fixture(tmp_path / "botocore_test.db", trace_id="test-botocore-trace")


def _make_prepared_request(
    url: str = _BEDROCK_URL,
    method: str = "POST",
    body: bytes = b'{"prompt": "hi"}',
    headers: dict | None = None,
    stream_output: bool = False,
) -> botocore.awsrequest.AWSPreparedRequest:
    return botocore.awsrequest.AWSPreparedRequest(
        method=method,
        url=url,
        headers=headers if headers is not None else {"Content-Type": b"application/json"},
        body=body,
        stream_output=stream_output,
    )


class _FakeRawStream:
    """Minimal stand-in for the real urllib3 raw response object.

    ``AWSResponse.content`` (used internally by ``RecordingSession``) only
    calls ``self.raw.stream()``; that's the one method a real
    ``urllib3.HTTPResponse`` provides that ``io.BytesIO`` does not, so tests
    need this rather than a bare ``BytesIO`` for the *initial* response.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data

    def stream(self, *args: object, **kwargs: object) -> Iterator[bytes]:
        yield self._data


def _make_response(
    status_code: int = 200,
    body: bytes = b'{"completion": "hello"}',
    headers: dict | None = None,
    url: str = _BEDROCK_URL,
) -> botocore.awsrequest.AWSResponse:
    return botocore.awsrequest.AWSResponse(
        url,
        status_code,
        headers if headers is not None else {"Content-Type": "application/json"},
        _FakeRawStream(body),
    )


def _record_one(
    fixture: Fixture,
    url: str = _BEDROCK_URL,
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
# RecordingSession
# ---------------------------------------------------------------------------


class TestRecordingSession:
    def test_inner_is_used_and_response_returned(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request()
        resp = _make_response(200, b'{"from_inner": true}')

        inner = MagicMock()
        inner.send.return_value = resp

        session = RecordingSession(fixture, inner=inner)
        result = session.send(req)

        inner.send.assert_called_once_with(req)
        assert result is resp
        fixture.close()

    def test_records_exchange_non_streaming(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse"
        req = _make_prepared_request(url=url, stream_output=False)
        resp = _make_response(200, b'{"output": "hi"}', url=url)

        inner = MagicMock()
        inner.send.return_value = resp

        RecordingSession(fixture, inner=inner).send(req)

        assert fixture.exchange_count() == 1
        ex = fixture.all_exchanges()[0]
        assert ex["url"] == url
        assert ex["method"] == "POST"
        assert ex["response_body"] == '{"output": "hi"}'
        assert ex["response_status"] == 200
        fixture.close()

    def test_records_request_body(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request(body=b'{"modelId": "anthropic.claude-v2"}')
        resp = _make_response(200, b"{}")

        inner = MagicMock()
        inner.send.return_value = resp

        RecordingSession(fixture, inner=inner).send(req)

        ex = fixture.all_exchanges()[0]
        assert ex["request_body"] == '{"modelId": "anthropic.claude-v2"}'
        fixture.close()

    def test_streaming_output_response_stays_readable_after_recording(
        self, tmp_path
    ) -> None:
        """invoke_model-style responses (stream_output=True) must remain
        readable after recording: StreamingBody wraps response.raw directly
        (see botocore.endpoint.convert_to_response_dict), so recording must
        not leave `.raw` exhausted.
        """
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request(stream_output=True)
        body = b'{"completion": "hello from bedrock"}'
        resp = _make_response(
            200, body, headers={"Content-Type": "application/json"}
        )

        inner = MagicMock()
        inner.send.return_value = resp

        result = RecordingSession(fixture, inner=inner).send(req)

        assert fixture.exchange_count() == 1
        assert fixture.all_exchanges()[0]["response_body"] == body.decode()

        # The caller (botocore) wraps whatever `.raw` is at this point in a
        # StreamingBody and reads it once the SDK call returns.
        streaming_body = botocore.response.StreamingBody(result.raw, len(body))
        assert streaming_body.read() == body

    def test_non_streaming_response_raw_untouched(self, tmp_path) -> None:
        """Only stream_output requests get `.raw` replaced; regular JSON API
        responses (body already delivered via `.content`) should not."""
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request(stream_output=False)
        resp = _make_response(200, b"{}")
        original_raw = resp.raw

        inner = MagicMock()
        inner.send.return_value = resp

        result = RecordingSession(fixture, inner=inner).send(req)

        assert result.raw is original_raw
        fixture.close()

    def test_request_headers_with_bytes_values_are_stringified(self, tmp_path) -> None:
        """Real signed AWS requests carry byte-valued headers (Authorization,
        Content-Type, ...); record_exchange() json.dumps()s the header dict,
        which raises TypeError on bytes if left un-decoded."""
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request(
            headers={
                b"Authorization": b"AWS4-HMAC-SHA256 Credential=AKIA.../...",
                "Content-Length": "2",
            },
            body=b"{}",
        )
        resp = _make_response(200, b"{}")

        inner = MagicMock()
        inner.send.return_value = resp

        RecordingSession(fixture, inner=inner).send(req)  # must not raise

        ex = fixture.all_exchanges()[0]
        assert ex["request_headers"]["Authorization"] == (
            "AWS4-HMAC-SHA256 Credential=AKIA.../..."
        )
        fixture.close()

    def test_response_headers_recorded(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request()
        resp = _make_response(
            200, b"{}", headers={"x-amzn-requestid": "abc-123"}
        )

        inner = MagicMock()
        inner.send.return_value = resp

        RecordingSession(fixture, inner=inner).send(req)

        ex = fixture.all_exchanges()[0]
        assert ex["response_headers"]["x-amzn-requestid"] == "abc-123"
        fixture.close()

    def test_default_inner_is_urllib3session(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        session = RecordingSession(fixture)
        assert isinstance(session._inner, botocore.httpsession.URLLib3Session)
        fixture.close()

    def test_close_delegates_to_inner(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        inner = MagicMock()
        session = RecordingSession(fixture, inner=inner)

        session.close()

        inner.close.assert_called_once()
        fixture.close()

    def test_records_str_request_body(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request(body='{"already": "a string"}')
        resp = _make_response(200, b"{}")

        inner = MagicMock()
        inner.send.return_value = resp

        RecordingSession(fixture, inner=inner).send(req)

        ex = fixture.all_exchanges()[0]
        assert ex["request_body"] == '{"already": "a string"}'
        fixture.close()

    def test_records_empty_string_for_none_request_body(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        req = _make_prepared_request(body=None)
        resp = _make_response(200, b"{}")

        inner = MagicMock()
        inner.send.return_value = resp

        RecordingSession(fixture, inner=inner).send(req)

        ex = fixture.all_exchanges()[0]
        assert ex["request_body"] == ""
        fixture.close()

    def test_close_is_noop_when_inner_has_no_close(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)

        class _NoClose:
            def send(self, request):  # pragma: no cover - not exercised
                raise NotImplementedError

        session = RecordingSession(fixture, inner=_NoClose())
        session.close()  # must not raise
        fixture.close()


# ---------------------------------------------------------------------------
# ReplaySession
# ---------------------------------------------------------------------------


class TestReplaySession:
    def test_send_returns_recorded_response(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse"
        _record_one(fixture, url=url, method="POST", body='{"replayed": true}')

        session = ReplaySession(fixture)
        response = session.send(_make_prepared_request(url=url))

        assert response.status_code == 200
        assert response.text == '{"replayed": true}'
        fixture.close()

    def test_send_returns_correct_status_code(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = "https://sagemaker-runtime.us-east-1.amazonaws.com/endpoints/x/invocations"
        _record_one(fixture, url=url, method="POST", body="created", status=201)

        session = ReplaySession(fixture)
        response = session.send(_make_prepared_request(url=url))

        assert response.status_code == 201
        fixture.close()

    def test_send_returns_response_body_as_bytes(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = _BEDROCK_URL
        _record_one(fixture, url=url, body='{"bytes": "value"}')

        session = ReplaySession(fixture)
        response = session.send(_make_prepared_request(url=url))

        assert isinstance(response.content, bytes)
        assert response.content == b'{"bytes": "value"}'
        fixture.close()

    def test_send_raises_network_guard_error_when_not_found(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "1")
        fixture = _make_fixture(tmp_path)
        session = ReplaySession(fixture)
        req = _make_prepared_request(url="https://not-recorded.example.com/endpoint")

        with pytest.raises(NetworkGuardError):
            session.send(req)
        fixture.close()

    def test_send_fallback_uses_urllib3session_when_guard_off(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("AGENT_TRACE_NETWORK_GUARD", raising=False)
        fixture = _make_fixture(tmp_path)
        session = ReplaySession(fixture)
        req = _make_prepared_request(url="https://not-recorded.example.com/fallback")

        mock_response = _make_response(200, b'{"fallback": true}')
        with patch.object(
            botocore.httpsession.URLLib3Session, "send", return_value=mock_response
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                response = session.send(req)
            assert len(w) == 1
            assert "no fixture entry" in str(w[0].message)

        assert response.status_code == 200
        fixture.close()

    def test_send_serves_multiple_exchanges_in_order(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url = _BEDROCK_URL
        for i in range(3):
            _record_one(fixture, url=url, body=f'{{"index": {i}}}')

        session = ReplaySession(fixture)
        for i in range(3):
            response = session.send(_make_prepared_request(url=url))
            assert json.loads(response.text)["index"] == i
        fixture.close()

    def test_send_independent_cursors_per_url(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        url_a = "https://bedrock-runtime.us-east-1.amazonaws.com/model/a/invoke"
        url_b = "https://bedrock-runtime.us-east-1.amazonaws.com/model/b/invoke"
        _record_one(fixture, url=url_a, body='{"url": "a"}')
        _record_one(fixture, url=url_b, body='{"url": "b"}')

        session = ReplaySession(fixture)
        resp_a = session.send(_make_prepared_request(url=url_a))
        resp_b = session.send(_make_prepared_request(url=url_b))

        assert json.loads(resp_a.text)["url"] == "a"
        assert json.loads(resp_b.text)["url"] == "b"
        fixture.close()

    def test_close_is_noop(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        session = ReplaySession(fixture)
        session.close()  # must not raise
        fixture.close()


# ---------------------------------------------------------------------------
# End-to-end: a real boto3 client through Tracer.start_trace(record=True)
# ---------------------------------------------------------------------------


class _FakeBedrockHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler standing in for the Bedrock runtime endpoint."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        payload = json.dumps({"completion": "hello from fake bedrock"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:  # silence stdlib logging
        pass


@pytest.fixture
def local_bedrock_server() -> Iterator[str]:
    """Start a loopback-only HTTP server standing in for Bedrock, yield its base URL."""
    server = HTTPServer(("127.0.0.1", 0), _FakeBedrockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestEndToEndBotocoreRecording:
    """Proves a real boto3 client's traffic is captured with zero extra
    wiring by the caller — the exact claim the botocore-interceptor backlog
    item makes (see [redacted])."""

    def test_real_bedrock_client_call_is_captured(
        self, tmp_path, local_bedrock_server: str
    ) -> None:
        session = boto3.Session(
            region_name="us-east-1",
            aws_access_key_id="AKIAFAKEFAKEFAKEFAKE",
            aws_secret_access_key="fakefakefakefakefakefakefakefakefakefake",
        )
        client = session.client(
            "bedrock-runtime",
            endpoint_url=local_bedrock_server,
            region_name="us-east-1",
        )

        tracer = Tracer(trace_dir=tmp_path)
        with tracer.start_trace("bedrock-e2e", record=True, run_id="bedrock-e2e"):
            result = client.invoke_model(
                modelId="anthropic.claude-v2",
                body=json.dumps({"prompt": "hi", "max_tokens_to_sample": 10}),
                contentType="application/json",
                accept="application/json",
            )
            # The caller's normal StreamingBody read must still work after
            # recording drained the socket to persist the exchange.
            body_bytes = result["body"].read()

        assert json.loads(body_bytes) == {"completion": "hello from fake bedrock"}

        fixture = Fixture(tmp_path / "bedrock-e2e" / "fixture.db")
        assert fixture.exchange_count() == 1
        exchange = fixture.all_exchanges()[0]
        assert exchange["method"] == "POST"
        assert exchange["url"].endswith("/model/anthropic.claude-v2/invoke")
        assert json.loads(exchange["response_body"]) == {
            "completion": "hello from fake bedrock"
        }
        fixture.close()

    def test_botocore_patch_is_uninstalled_after_trace_exits(self, tmp_path) -> None:
        orig_send = botocore.httpsession.URLLib3Session.send

        tracer = Tracer(trace_dir=tmp_path)
        with tracer.start_trace("patch-check", record=True, run_id="patch-check"):
            assert botocore.httpsession.URLLib3Session.send is not orig_send

        assert botocore.httpsession.URLLib3Session.send is orig_send

    def test_nested_record_true_does_not_double_patch_botocore(self, tmp_path) -> None:
        orig_send = botocore.httpsession.URLLib3Session.send

        tracer = Tracer(trace_dir=tmp_path)
        with tracer.start_trace("outer", record=True, run_id="outer-boto"):
            patched_send = botocore.httpsession.URLLib3Session.send
            assert patched_send is not orig_send

            with tracer.start_trace("inner", record=True, run_id="inner-boto"):
                assert botocore.httpsession.URLLib3Session.send is patched_send

            assert botocore.httpsession.URLLib3Session.send is patched_send

        assert botocore.httpsession.URLLib3Session.send is orig_send
