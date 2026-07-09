"""
Unit tests for agent_trace.interceptor.grpc_hook.

Unlike the httpx/requests interceptor tests (which mock at the transport
layer with respx), grpc has no equivalent lightweight mocking story, so
these tests spin up a *real* grpc.Server bound to 127.0.0.1:0 (an ephemeral
port) backed by a tiny compiled proto service (tests/unit/grpc_fixtures/echo.proto).
Everything stays on localhost -- no external network I/O.

AGENT_TRACE_NETWORK_GUARD=1 is set by pytest env (pyproject.toml), so replay
interceptors raise NetworkGuardError on any unmatched request unless a test
explicitly overrides it via monkeypatch.
"""

from __future__ import annotations

import base64
import json
import sys
import warnings
from concurrent import futures
from pathlib import Path

import grpc
import pytest

# echo_pb2 / echo_pb2_grpc are generated code that does a flat top-level
# `import echo_pb2`, so the fixtures directory must be on sys.path directly
# (it's not a package).
_FIXTURES_DIR = Path(__file__).parent / "grpc_fixtures"
sys.path.insert(0, str(_FIXTURES_DIR))
import echo_pb2
import echo_pb2_grpc

from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.grpc_hook import (
    AsyncGRPCRecordingInterceptor,
    AsyncGRPCReplayInterceptor,
    GRPCRecordingInterceptor,
    GRPCReplayInterceptor,
    NetworkGuardError,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Test server
# ---------------------------------------------------------------------------


class _EchoServicer(echo_pb2_grpc.EchoServicer):
    # Method names match the .proto-defined RPCs (PascalCase is grpc
    # convention, not a violation) -- noqa: N802 throughout this class.
    def UnaryEcho(self, request, context):  # noqa: N802
        context.set_trailing_metadata((("x-served-by", "test-server"),))
        return echo_pb2.EchoResponse(message=f"echo:{request.message}")

    def StreamingEcho(self, request, context):  # noqa: N802
        for i in range(3):
            yield echo_pb2.EchoResponse(message=f"{request.message}-{i}")

    def FailingEcho(self, request, context):  # noqa: N802
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "bad request")


@pytest.fixture(scope="module")
def echo_server():
    """Real grpc.Server on 127.0.0.1:<ephemeral>, torn down after the module."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    echo_pb2_grpc.add_EchoServicer_to_server(_EchoServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop(grace=None)


def _make_fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path / "grpc_test.db", trace_id="test-trace")


# ---------------------------------------------------------------------------
# GRPCRecordingInterceptor — unary-unary
# ---------------------------------------------------------------------------


class TestGRPCRecordingInterceptorUnaryUnary:
    def test_records_request_and_response(self, echo_server, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        response = stub.UnaryEcho(echo_pb2.EchoRequest(message="hello"))

        assert response.message == "echo:hello"
        assert fixture.exchange_count() == 1
        exchanges = fixture.all_exchanges()
        assert (
            exchanges[0]["url"]
            == f"grpc://{echo_server}/agenttrace.test.Echo/UnaryEcho"
        )
        assert exchanges[0]["method"] == "GRPC_UNARY_UNARY"
        assert exchanges[0]["response_status"] == 0  # StatusCode.OK
        fixture.close()

    def test_recorded_request_body_round_trips(self, echo_server, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        stub.UnaryEcho(echo_pb2.EchoRequest(message="round-trip-me"))

        exchange = fixture.all_exchanges()[0]
        req = echo_pb2.EchoRequest()
        req.ParseFromString(base64.b64decode(exchange["request_body"]))
        assert req.message == "round-trip-me"
        fixture.close()

    def test_recorded_response_body_round_trips(self, echo_server, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        stub.UnaryEcho(echo_pb2.EchoRequest(message="body-check"))

        exchange = fixture.all_exchanges()[0]
        resp = echo_pb2.EchoResponse()
        resp.ParseFromString(base64.b64decode(exchange["response_body"]))
        assert resp.message == "echo:body-check"
        fixture.close()

    def test_records_trailing_metadata_and_response_type(
        self, echo_server, tmp_path
    ) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        stub.UnaryEcho(echo_pb2.EchoRequest(message="meta"))

        exchange = fixture.all_exchanges()[0]
        assert exchange["response_headers"]["x-served-by"] == "test-server"
        assert (
            exchange["response_headers"]["X-AGENT-TRACE-GRPC-RESPONSE-TYPE"]
            == "agenttrace.test.EchoResponse"
        )
        fixture.close()

    def test_records_request_metadata(self, echo_server, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        stub.UnaryEcho(
            echo_pb2.EchoRequest(message="with-metadata"),
            metadata=(("x-api-key", "secret-123"),),
        )

        exchange = fixture.all_exchanges()[0]
        assert exchange["request_headers"]["x-api-key"] == "secret-123"
        fixture.close()

    def test_error_response_not_recorded(self, echo_server, tmp_path) -> None:
        """A non-OK status raises grpc.RpcError before a response exists to
        record, mirroring RecordingTransport (which never sees a Response
        object for a transport-level failure either)."""
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        with pytest.raises(grpc.RpcError):
            stub.FailingEcho(echo_pb2.EchoRequest(message="boom"))

        assert fixture.exchange_count() == 0
        fixture.close()


# ---------------------------------------------------------------------------
# GRPCRecordingInterceptor — unary-stream
# ---------------------------------------------------------------------------


class TestGRPCRecordingInterceptorUnaryStream:
    def test_caller_receives_full_stream(self, echo_server, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        messages = [
            r.message for r in stub.StreamingEcho(echo_pb2.EchoRequest(message="s"))
        ]

        assert messages == ["s-0", "s-1", "s-2"]
        fixture.close()

    def test_records_all_streamed_messages(self, echo_server, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        list(stub.StreamingEcho(echo_pb2.EchoRequest(message="rec")))

        assert fixture.exchange_count() == 1
        exchange = fixture.all_exchanges()[0]
        assert (
            exchange["url"]
            == f"grpc://{echo_server}/agenttrace.test.Echo/StreamingEcho"
        )
        assert exchange["method"] == "GRPC_UNARY_STREAM"
        items = json.loads(exchange["response_body"])
        assert len(items) == 3
        decoded = [echo_pb2.EchoResponse() for _ in items]
        for msg, raw in zip(decoded, items, strict=True):
            msg.ParseFromString(base64.b64decode(raw))
        assert [m.message for m in decoded] == ["rec-0", "rec-1", "rec-2"]
        fixture.close()

    def test_recording_is_not_persisted_until_stream_exhausted(
        self, echo_server, tmp_path
    ) -> None:
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCRecordingInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        call = stub.StreamingEcho(echo_pb2.EchoRequest(message="lazy"))
        first = next(iter(call))
        assert first.message == "lazy-0"
        # Only one item has been pulled so far — nothing recorded yet.
        assert fixture.exchange_count() == 0

        # Drain the rest; now the exchange should be persisted.
        remaining = list(call)
        assert remaining[-1].message == "lazy-2"
        assert fixture.exchange_count() == 1
        fixture.close()


# ---------------------------------------------------------------------------
# GRPCReplayInterceptor — unary-unary
# ---------------------------------------------------------------------------


def _record_unary_exchange(
    fixture: Fixture,
    target: str,
    method_path: str,
    request: echo_pb2.EchoRequest,
    response: echo_pb2.EchoResponse,
    *,
    status: int = 0,
    extra_headers: dict[str, str] | None = None,
) -> None:
    headers = {
        "X-AGENT-TRACE-GRPC-RESPONSE-TYPE": response.DESCRIPTOR.full_name,
        "X-AGENT-TRACE-GRPC-KIND": "unary",
    }
    if extra_headers:
        headers.update(extra_headers)
    fixture.record_exchange(
        url=f"grpc://{target}{method_path}",
        method="GRPC_UNARY_UNARY",
        request_headers={},
        request_body=base64.b64encode(request.SerializeToString()).decode("ascii"),
        response_status=status,
        response_headers=headers,
        response_body=base64.b64encode(response.SerializeToString()).decode("ascii"),
    )


class TestGRPCReplayInterceptorUnaryUnary:
    def test_serves_recorded_exchange_without_network(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        _record_unary_exchange(
            fixture,
            target,
            "/agenttrace.test.Echo/UnaryEcho",
            echo_pb2.EchoRequest(message="hi"),
            echo_pb2.EchoResponse(message="replayed!"),
        )

        # The underlying channel targets a host that doesn't exist; if the
        # replay interceptor ever called `continuation`, this would hang or
        # fail to connect rather than returning instantly.
        channel = grpc.intercept_channel(
            grpc.insecure_channel(target),
            GRPCReplayInterceptor(fixture, target),
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        response = stub.UnaryEcho(echo_pb2.EchoRequest(message="hi"))

        assert response.message == "replayed!"
        fixture.close()

    def test_serves_correct_status_code(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        _record_unary_exchange(
            fixture,
            target,
            "/agenttrace.test.Echo/UnaryEcho",
            echo_pb2.EchoRequest(message="x"),
            echo_pb2.EchoResponse(message="y"),
            status=grpc.StatusCode.OK.value[0],
        )
        channel = grpc.intercept_channel(
            grpc.insecure_channel(target),
            GRPCReplayInterceptor(fixture, target),
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        _, call = stub.UnaryEcho.with_call(echo_pb2.EchoRequest(message="x"))

        assert call.code() == grpc.StatusCode.OK
        fixture.close()

    def test_raises_network_guard_error_when_exchange_not_found(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        channel = grpc.intercept_channel(
            grpc.insecure_channel(target),
            GRPCReplayInterceptor(fixture, target),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        with pytest.raises(NetworkGuardError):
            stub.UnaryEcho(echo_pb2.EchoRequest(message="never-recorded"))

        fixture.close()

    def test_guard_off_missing_exchange_falls_back_to_real_network(
        self, echo_server, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AGENT_TRACE_NETWORK_GUARD", "0")
        fixture = _make_fixture(tmp_path)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(echo_server),
            GRPCReplayInterceptor(fixture, echo_server),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response = stub.UnaryEcho(echo_pb2.EchoRequest(message="fallback"))
            assert len(caught) >= 1

        assert response.message == "echo:fallback"
        fixture.close()

    def test_serves_exchanges_in_order_for_same_method(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        for i in range(3):
            _record_unary_exchange(
                fixture,
                target,
                "/agenttrace.test.Echo/UnaryEcho",
                echo_pb2.EchoRequest(message="seq"),
                echo_pb2.EchoResponse(message=f"step-{i}"),
            )
        channel = grpc.intercept_channel(
            grpc.insecure_channel(target),
            GRPCReplayInterceptor(fixture, target),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        for i in range(3):
            response = stub.UnaryEcho(echo_pb2.EchoRequest(message="seq"))
            assert response.message == f"step-{i}"

        fixture.close()


class TestGRPCReplayInterceptorUnaryStream:
    def test_serves_recorded_stream_without_network(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        items = [echo_pb2.EchoResponse(message=f"item-{i}") for i in range(3)]
        fixture.record_exchange(
            url=f"grpc://{target}/agenttrace.test.Echo/StreamingEcho",
            method="GRPC_UNARY_STREAM",
            request_headers={},
            request_body=base64.b64encode(
                echo_pb2.EchoRequest(message="s").SerializeToString()
            ).decode("ascii"),
            response_status=0,
            response_headers={
                "X-AGENT-TRACE-GRPC-RESPONSE-TYPE": "agenttrace.test.EchoResponse",
                "X-AGENT-TRACE-GRPC-KIND": "stream",
            },
            response_body=json.dumps(
                [base64.b64encode(m.SerializeToString()).decode("ascii") for m in items]
            ),
        )

        channel = grpc.intercept_channel(
            grpc.insecure_channel(target),
            GRPCReplayInterceptor(fixture, target),
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        messages = [
            r.message for r in stub.StreamingEcho(echo_pb2.EchoRequest(message="s"))
        ]

        assert messages == ["item-0", "item-1", "item-2"]
        fixture.close()

    def test_raises_network_guard_error_when_stream_not_found(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        channel = grpc.intercept_channel(
            grpc.insecure_channel(target),
            GRPCReplayInterceptor(fixture, target),
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        with pytest.raises(NetworkGuardError):
            list(stub.StreamingEcho(echo_pb2.EchoRequest(message="never-recorded")))

        fixture.close()


class TestGRPCReplayInterceptorUnsupportedStreamingShapes:
    """stream-unary / stream-stream replay raises rather than silently
    falling through to the real network (see grpc_hook.py module docstring).
    """

    def test_stream_unary_raises_not_implemented(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        interceptor = GRPCReplayInterceptor(fixture, target)

        with pytest.raises(NotImplementedError):
            interceptor.intercept_stream_unary(
                lambda details, it: None,
                _FakeCallDetails("/agenttrace.test.Echo/ClientStream"),
                iter([]),
            )
        fixture.close()

    def test_stream_stream_raises_not_implemented(self, tmp_path) -> None:
        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        interceptor = GRPCReplayInterceptor(fixture, target)

        with pytest.raises(NotImplementedError):
            interceptor.intercept_stream_stream(
                lambda details, it: None,
                _FakeCallDetails("/agenttrace.test.Echo/Bidi"),
                iter([]),
            )
        fixture.close()


class _FakeCallDetails:
    def __init__(self, method: str) -> None:
        self.method = method
        self.metadata = None


# ---------------------------------------------------------------------------
# grpc.aio (async) recording / replay — unary-unary
# ---------------------------------------------------------------------------


class TestAsyncGRPCRecordingInterceptor:
    async def test_records_and_returns_response(self, echo_server, tmp_path) -> None:
        from grpc import aio

        fixture = _make_fixture(tmp_path)
        channel = aio.insecure_channel(
            echo_server,
            interceptors=[AsyncGRPCRecordingInterceptor(fixture, echo_server)],
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        response = await stub.UnaryEcho(echo_pb2.EchoRequest(message="async-hi"))

        assert response.message == "echo:async-hi"
        assert fixture.exchange_count() == 1
        exchange = fixture.all_exchanges()[0]
        assert exchange["url"] == f"grpc://{echo_server}/agenttrace.test.Echo/UnaryEcho"
        assert exchange["method"] == "GRPC_UNARY_UNARY"

        await channel.close()
        fixture.close()


class TestAsyncGRPCReplayInterceptor:
    async def test_serves_recorded_exchange_without_network(self, tmp_path) -> None:
        from grpc import aio

        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        _record_unary_exchange(
            fixture,
            target,
            "/agenttrace.test.Echo/UnaryEcho",
            echo_pb2.EchoRequest(message="hi"),
            echo_pb2.EchoResponse(message="async-replayed!"),
        )

        channel = aio.insecure_channel(
            target,
            interceptors=[AsyncGRPCReplayInterceptor(fixture, target)],
        )
        stub = echo_pb2_grpc.EchoStub(channel)
        response = await stub.UnaryEcho(echo_pb2.EchoRequest(message="hi"))

        assert response.message == "async-replayed!"
        await channel.close()
        fixture.close()

    async def test_raises_network_guard_error_when_exchange_not_found(
        self, tmp_path
    ) -> None:
        from grpc import aio

        fixture = _make_fixture(tmp_path)
        target = "never-connects.invalid:443"
        channel = aio.insecure_channel(
            target,
            interceptors=[AsyncGRPCReplayInterceptor(fixture, target)],
        )
        stub = echo_pb2_grpc.EchoStub(channel)

        with pytest.raises(NetworkGuardError):
            await stub.UnaryEcho(echo_pb2.EchoRequest(message="never-recorded"))

        await channel.close()
        fixture.close()


# ---------------------------------------------------------------------------
# End-to-end: Tracer.start_trace(record=True) wiring (the actual backlog
# requirement -- "wire it into Tracer._install_recording_transport /
# _uninstall_recording_transport so it activates/deactivates alongside the
# existing httpx/requests patches"). These tests never construct a
# GRPCRecordingInterceptor directly -- they only call the public Tracer API
# and plain `grpc.insecure_channel(...)`, exactly as an unmodified SDK would.
# ---------------------------------------------------------------------------


class TestTracerGRPCWiring:
    def test_start_trace_record_true_captures_grpc_call(
        self, echo_server, tmp_path
    ) -> None:
        from agent_trace import Tracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("grpc-e2e", record=True) as trace:
            run_id = trace.run_id
            # Unmodified SDK-style usage: no interceptor constructed by hand.
            channel = grpc.insecure_channel(echo_server)
            stub = echo_pb2_grpc.EchoStub(channel)
            response = stub.UnaryEcho(echo_pb2.EchoRequest(message="tracer-e2e"))
            assert response.message == "echo:tracer-e2e"

        fixture_db = tmp_path / run_id / "fixture.db"
        assert fixture_db.exists()
        with Fixture(fixture_db) as fixture:
            assert fixture.exchange_count() == 1
            exchange = fixture.all_exchanges()[0]
            assert (
                exchange["url"]
                == f"grpc://{echo_server}/agenttrace.test.Echo/UnaryEcho"
            )

    def test_grpc_channel_factories_restored_after_trace_exit(self, tmp_path) -> None:
        from agent_trace import Tracer

        original_insecure = grpc.insecure_channel
        original_secure = grpc.secure_channel

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("grpc-patch-check", record=True):
            assert grpc.insecure_channel is not original_insecure
            assert grpc.secure_channel is not original_secure

        assert grpc.insecure_channel is original_insecure
        assert grpc.secure_channel is original_secure

    def test_nested_start_trace_does_not_double_patch_grpc(self, tmp_path) -> None:
        """Mirrors test_tracer.py's nested-httpx-patch test for grpc."""
        from agent_trace import Tracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("outer", record=True, run_id="outer-grpc"):
            patched_after_outer = grpc.insecure_channel
            with t.start_trace("inner", record=True, run_id="inner-grpc"):
                assert grpc.insecure_channel is patched_after_outer
            # Inner exit must not restore the original while outer is active.
            assert grpc.insecure_channel is patched_after_outer
        assert grpc.insecure_channel is not patched_after_outer

    def test_replay_context_serves_grpc_call_from_fixture(
        self, echo_server, tmp_path
    ) -> None:
        """Full round-trip: record via Tracer, then replay via replay()."""
        from agent_trace import Tracer, replay

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("grpc-roundtrip", record=True) as trace:
            run_id = trace.run_id
            channel = grpc.insecure_channel(echo_server)
            stub = echo_pb2_grpc.EchoStub(channel)
            stub.UnaryEcho(echo_pb2.EchoRequest(message="roundtrip"))

        # AGENT_TRACE_NETWORK_GUARD=1 is set by pytest env, so if replay ever
        # missed the fixture and fell through to the real network, it would
        # raise NetworkGuardError instead of silently succeeding -- a clean
        # response here proves this was served from the fixture.
        with replay(run_id, trace_dir=tmp_path):
            channel = grpc.insecure_channel(echo_server)
            stub = echo_pb2_grpc.EchoStub(channel)
            response = stub.UnaryEcho(echo_pb2.EchoRequest(message="roundtrip"))
            assert response.message == "echo:roundtrip"
