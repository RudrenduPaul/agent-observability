"""
Unit tests for the public Tracer API (agent_trace.Tracer).

Tests cover start_trace(), span(), instrument(), and the replay() factory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_trace import SpanStatus, Trace, Tracer, replay, tracer
from agent_trace.core.clock import FixtureClock, WallClock

# ---------------------------------------------------------------------------
# Tracer instantiation
# ---------------------------------------------------------------------------


class TestTracerInit:
    def test_default_trace_dir(self) -> None:
        t = Tracer()
        assert t._trace_dir == Path.home() / ".agent-trace" / "runs"

    def test_custom_trace_dir(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path / "my-traces")
        assert t._trace_dir == tmp_path / "my-traces"

    def test_global_tracer_is_tracer_instance(self) -> None:
        assert isinstance(tracer, Tracer)


# ---------------------------------------------------------------------------
# Tracer.start_trace()
# ---------------------------------------------------------------------------


class TestStartTrace:
    def test_start_trace_yields_trace(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("test-trace") as trace:
            assert isinstance(trace, Trace)

    def test_start_trace_creates_run_dir(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("test-trace") as trace:
            run_id = trace.run_id
        run_dir = tmp_path / run_id
        assert run_dir.is_dir()

    def test_start_trace_writes_trace_json(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("test-trace") as trace:
            run_id = trace.run_id
        trace_json = tmp_path / run_id / "trace.json"
        assert trace_json.exists()

    def test_start_trace_json_is_valid(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("json-test") as trace:
            run_id = trace.run_id
        data = json.loads((tmp_path / run_id / "trace.json").read_text())
        assert "trace_id" in data
        assert "spans" in data

    def test_start_trace_record_true_creates_fixture_db(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("recorded", record=True) as trace:
            run_id = trace.run_id
        fixture_db = tmp_path / run_id / "fixture.db"
        assert fixture_db.exists()

    def test_start_trace_record_false_no_fixture_db(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("unrecorded", record=False) as trace:
            run_id = trace.run_id
        fixture_db = tmp_path / run_id / "fixture.db"
        assert not fixture_db.exists()

    def test_start_trace_active_trace_is_set(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        assert t.active_trace is None
        with t.start_trace("active-test") as trace:
            assert t.active_trace is trace

    def test_start_trace_active_trace_restored_after_exit(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("outer"):
            pass
        assert t.active_trace is None

    def test_nested_start_trace_restores_outer(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("outer") as outer:
            outer_ref = t.active_trace
            with t.start_trace("inner") as inner:
                assert t.active_trace is inner
                assert t.active_trace is not outer
            # After inner exits, outer is restored
            assert t.active_trace is outer

    def test_start_trace_with_custom_run_id(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("custom-id", run_id="my-custom-run") as trace:
            pass
        assert (tmp_path / "my-custom-run").is_dir()

    def test_start_trace_trace_id_is_hex_and_differs_from_run_id(
        self, tmp_path: Path
    ) -> None:
        """trace_id must be 128-bit hex so OTLP can parse it; run_id is human-readable."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("hex-check") as trace:
            # 32 hex chars = 128 bits
            assert len(trace.trace_id) == 32
            int(trace.trace_id, 16)  # raises ValueError if not valid hex
            # trace_id and run_id must be independent
            assert trace.trace_id != trace.run_id

    def test_start_trace_explicit_trace_id_override(self, tmp_path: Path) -> None:
        """trace_id=... overrides the default random uuid4 — the mechanism
        for cross-worker run correlation (issue #7417)."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("explicit-id", trace_id="deadbeef" * 4) as trace:
            assert trace.trace_id == "deadbeef" * 4

    def test_start_trace_no_trace_id_still_random(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("a") as trace_a:
            id_a = trace_a.trace_id
        with t.start_trace("b") as trace_b:
            id_b = trace_b.trace_id
        assert id_a != id_b


# ---------------------------------------------------------------------------
# Tracer.start_trace(remote_backend=...) — durable/remote fixture sync
# (issue #7417)
# ---------------------------------------------------------------------------


class TestStartTraceRemoteBackend:
    def test_remote_backend_receives_synced_exchanges_during_recording(
        self, tmp_path: Path
    ) -> None:
        from agent_trace.exporters.remote_fixture import LocalDirRemoteFixtureBackend

        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        t = Tracer(trace_dir=tmp_path / "local")
        with t.start_trace("remote-test", record=True, remote_backend=backend) as trace:
            run_id = trace.run_id
            fixture = t._active_fixture_var.get()
            fixture.record_exchange(
                url="https://api.example.com",
                method="POST",
                request_headers={},
                request_body="",
                response_status=200,
                response_headers={},
                response_body="ok",
            )
        synced = backend.list_keys(f"{run_id}/exchanges/")
        assert len(synced) == 1

    def test_remote_backend_syncs_trace_json_and_fixture_on_exit(
        self, tmp_path: Path
    ) -> None:
        from agent_trace.exporters.remote_fixture import LocalDirRemoteFixtureBackend

        backend = LocalDirRemoteFixtureBackend(tmp_path / "remote")
        t = Tracer(trace_dir=tmp_path / "local")
        with t.start_trace("remote-test-2", record=True, remote_backend=backend) as trace:
            run_id = trace.run_id
        assert backend.get_bytes(f"{run_id}/trace.json") is not None
        assert backend.get_bytes(f"{run_id}/fixture.db") is not None

    def test_no_remote_backend_is_unaffected(self, tmp_path: Path) -> None:
        """Omitting remote_backend (the default) behaves exactly as before."""
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("no-remote", record=True) as trace:
            run_id = trace.run_id
        assert (tmp_path / run_id / "fixture.db").exists()

    def test_remote_backend_failure_does_not_break_local_trace(
        self, tmp_path: Path
    ) -> None:
        class _BoomBackend:
            def put_bytes(self, key: str, data: bytes) -> None:
                raise RuntimeError("remote store unreachable")

            def get_bytes(self, key: str) -> bytes | None:
                return None

            def list_keys(self, prefix: str) -> list[str]:
                return []

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace(
            "remote-failure", record=True, remote_backend=_BoomBackend()
        ) as trace:
            run_id = trace.run_id
        # Local trace.json/fixture.db still written despite remote failure.
        assert (tmp_path / run_id / "trace.json").exists()
        assert (tmp_path / run_id / "fixture.db").exists()


# ---------------------------------------------------------------------------
# Tracer.span()
# ---------------------------------------------------------------------------


class TestTracerSpan:
    def test_span_added_to_active_trace(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("span-test") as trace:
            with t.span("my-span"):
                pass
            assert len(trace.spans) == 1
            assert trace.spans[0].name == "my-span"

    def test_span_auto_ends_with_ok_on_success(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("ok-test") as trace:
            with t.span("ok-span"):
                pass
            assert trace.spans[0].status == SpanStatus.OK
            assert trace.spans[0].end_time is not None

    def test_span_records_exception_and_reraises(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("err-test") as trace:
            with pytest.raises(ValueError):
                with t.span("error-span"):
                    raise ValueError("test error")
            span = trace.spans[0]
            assert span.status == SpanStatus.ERROR
            assert any(e.name == "exception" for e in span.events)

    def test_span_end_time_set_on_error(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("err-time") as trace:
            with pytest.raises(RuntimeError):
                with t.span("err-span"):
                    raise RuntimeError("err")
            assert trace.spans[0].end_time is not None

    def test_multiple_spans_added(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("multi") as trace:
            with t.span("span-a"):
                pass
            with t.span("span-b"):
                pass
        assert len(trace.spans) == 2


# ---------------------------------------------------------------------------
# Tracer.instrument()
# ---------------------------------------------------------------------------


class TestTracerInstrument:
    def test_instrument_wraps_function(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        @t.instrument()
        def my_fn() -> str:
            return "result"

        assert my_fn() == "result"

    def test_instrument_preserves_function_name(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        @t.instrument()
        def my_named_fn() -> None:
            """My docstring."""

        assert my_named_fn.__name__ == "my_named_fn"

    def test_instrument_preserves_docstring(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        @t.instrument()
        def fn_with_doc() -> None:
            """Important docstring."""

        assert "Important docstring" in (fn_with_doc.__doc__ or "")

    def test_instrument_record_false_no_fixture(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        @t.instrument(record=False)
        def plain_fn() -> str:
            return "x"

        plain_fn()

        # Find any run directory and check no fixture.db exists
        for run_dir in tmp_path.iterdir():
            assert not (run_dir / "fixture.db").exists()

    def test_instrument_creates_trace_json(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        @t.instrument()
        def traced_fn() -> None:
            pass

        traced_fn()

        # At least one run directory with trace.json should exist
        json_files = list(tmp_path.glob("*/trace.json"))
        assert len(json_files) == 1

    def test_instrument_reraises_exception(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        @t.instrument()
        def raising_fn() -> None:
            raise ValueError("re-raised")

        with pytest.raises(ValueError, match="re-raised"):
            raising_fn()

    def test_instrument_custom_name(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)

        @t.instrument(name="custom-trace-name")
        def fn() -> None:
            pass

        fn()

        json_files = list(tmp_path.glob("*/trace.json"))
        assert len(json_files) >= 1
        data = json.loads(json_files[0].read_text())
        assert data.get("metadata", {}).get("name") == "custom-trace-name"


# ---------------------------------------------------------------------------
# replay() factory and ReplayContext
# ---------------------------------------------------------------------------


class TestReplayFactory:
    def test_replay_returns_replay_context(self, tmp_path: Path) -> None:
        from agent_trace import ReplayContext

        # Create a fixture.db first
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("replay-test", record=True) as trace:
            run_id = trace.run_id

        ctx = replay(run_id, trace_dir=tmp_path)
        assert isinstance(ctx, ReplayContext)

    def test_replay_raises_file_not_found_if_no_fixture(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            replay("nonexistent-run-id", trace_dir=tmp_path)

    def test_replay_context_manager_works(self, tmp_path: Path) -> None:
        from agent_trace._replay.fixture import Fixture

        # Build a fixture manually
        run_id = "test-run-001"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        fixture_path = run_dir / "fixture.db"
        with Fixture(fixture_path) as f:
            f.set_metadata("input", "test-input")

        ctx = replay(run_id, trace_dir=tmp_path)
        with ctx as active_ctx:
            assert active_ctx is ctx

    def test_replay_context_get_metadata(self, tmp_path: Path) -> None:
        from agent_trace._replay.fixture import Fixture

        run_id = "meta-run-001"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        fixture_path = run_dir / "fixture.db"
        with Fixture(fixture_path) as f:
            f.set_metadata("agent_version", "v2")

        with replay(run_id, trace_dir=tmp_path) as ctx:
            assert ctx.get_metadata("agent_version") == "v2"

    def test_replay_context_exit_restores_clock(self, tmp_path: Path) -> None:
        from agent_trace._replay.fixture import Fixture

        run_id = "clock-run-001"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        fixture_path = run_dir / "fixture.db"
        with Fixture(fixture_path):
            pass

        with replay(run_id, trace_dir=tmp_path):
            from agent_trace.core.clock import get_clock

            assert isinstance(get_clock(), FixtureClock)

        from agent_trace.core.clock import get_clock

        assert isinstance(get_clock(), WallClock)

    def test_replay_context_fixture_raises_when_not_entered(
        self, tmp_path: Path
    ) -> None:
        from agent_trace._replay.fixture import Fixture

        run_id = "no-enter-run"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        with Fixture(run_dir / "fixture.db"):
            pass

        ctx = replay(run_id, trace_dir=tmp_path)
        with pytest.raises(RuntimeError, match="context manager"):
            _ = ctx.fixture

    def test_replay_accepts_direct_fixture_file_path(self, tmp_path: Path) -> None:
        """replay() must accept a path pointing directly to fixture.db, not just dirs."""
        from agent_trace._replay.fixture import Fixture

        fixture_path = tmp_path / "fixture.db"
        with Fixture(fixture_path) as f:
            f.set_metadata("source", "direct-file")

        with replay(fixture_path) as ctx:
            assert ctx.get_metadata("source") == "direct-file"

    def test_replay_accepts_direct_fixture_file_path_string(
        self, tmp_path: Path
    ) -> None:
        """replay() must accept a str path pointing directly to fixture.db."""
        from agent_trace._replay.fixture import Fixture

        fixture_path = tmp_path / "fixture.db"
        with Fixture(fixture_path):
            pass

        with replay(str(fixture_path)) as ctx:
            assert ctx is not None


# ---------------------------------------------------------------------------
# Path traversal rejection
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_start_trace_rejects_traversal_in_run_id(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            with t.start_trace("test", run_id="../../etc/passwd"):
                pass

    def test_replay_rejects_traversal_in_run_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            with replay("../../etc/passwd", trace_dir=tmp_path):
                pass

    def test_start_trace_accepts_normal_run_id(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("test", run_id="safe-run-id") as trace:
            assert trace.run_id == "safe-run-id"


# ---------------------------------------------------------------------------
# Async instrument() support
# ---------------------------------------------------------------------------


class TestAsyncInstrument:
    def test_instrument_detects_async_function(self, tmp_path: Path) -> None:
        import inspect

        t = Tracer(trace_dir=tmp_path)

        @t.instrument(name="async-agent")
        async def async_agent(x: int) -> int:
            return x * 2

        assert inspect.iscoroutinefunction(async_agent)

    def test_instrument_async_returns_correct_result(self, tmp_path: Path) -> None:
        import asyncio

        t = Tracer(trace_dir=tmp_path)

        @t.instrument(name="async-agent")
        async def async_agent(x: int) -> int:
            return x * 3

        result = asyncio.run(async_agent(7))
        assert result == 21

    def test_instrument_async_creates_trace_json(self, tmp_path: Path) -> None:
        import asyncio

        t = Tracer(trace_dir=tmp_path)

        @t.instrument(record=False, name="async-agent")
        async def async_agent() -> str:
            return "done"

        asyncio.run(async_agent())

        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        trace_json = run_dirs[0] / "trace.json"
        assert trace_json.exists()

    def test_instrument_async_propagates_exception(self, tmp_path: Path) -> None:
        import asyncio

        t = Tracer(trace_dir=tmp_path)

        @t.instrument(name="failing-agent")
        async def failing_agent() -> None:
            raise ValueError("async failure")

        with pytest.raises(ValueError, match="async failure"):
            asyncio.run(failing_agent())


# ---------------------------------------------------------------------------
# Recording transport nesting counter
# ---------------------------------------------------------------------------


class TestRecordingTransportNesting:
    """Recording is now patched at request-dispatch time
    (``httpx.Client._transport_for_url`` / ``AsyncClient._transport_for_url``),
    not at ``Client.__init__`` time — see ``Tracer._patch_httpx``.  These tests
    assert against ``_transport_for_url`` accordingly.
    """

    def test_nested_record_true_does_not_double_patch(self, tmp_path: Path) -> None:
        """Two nested start_trace(record=True) must not double-patch httpx."""
        import httpx

        t = Tracer(trace_dir=tmp_path)
        original = httpx.Client._transport_for_url

        with t.start_trace("outer", record=True, run_id="outer"):
            patched = httpx.Client._transport_for_url
            assert patched is not original

            with t.start_trace("inner", record=True, run_id="inner"):
                # The patch should NOT have changed (no double-wrap)
                assert httpx.Client._transport_for_url is patched

            # After inner exits, outer's patch should still be in place
            assert httpx.Client._transport_for_url is patched

        # After outer exits, original is restored
        assert httpx.Client._transport_for_url is original

    def test_depth_returns_to_zero_after_nested_exit(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("outer", record=True, run_id="outer-d"):
            with t.start_trace("inner", record=True, run_id="inner-d"):
                assert t._transport_depth == 2
            assert t._transport_depth == 1
        assert t._transport_depth == 0

    def test_async_client_is_also_patched_during_recording(
        self, tmp_path: Path
    ) -> None:
        """AsyncClient._transport_for_url must be patched alongside Client."""
        import httpx

        t = Tracer(trace_dir=tmp_path)
        orig_async = httpx.AsyncClient._transport_for_url
        orig_sync = httpx.Client._transport_for_url

        with t.start_trace("async-patch-test", record=True, run_id="async-patch"):
            assert httpx.Client._transport_for_url is not orig_sync
            assert httpx.AsyncClient._transport_for_url is not orig_async, (
                "AsyncClient._transport_for_url was not patched — async HTTP "
                "calls during record mode would be silently unintercepted"
            )

        # Both restored on exit
        assert httpx.Client._transport_for_url is orig_sync
        assert httpx.AsyncClient._transport_for_url is orig_async


# ---------------------------------------------------------------------------
# Pre-existing clients / caller-supplied transports are still captured
# ---------------------------------------------------------------------------


class TestPreExistingClientCapture:
    """RecordingTransport must cover httpx.Client instances constructed
    *before* recording activates — the patch works at request-dispatch time
    (``_transport_for_url``), not at __init__ time, so it doesn't matter
    when the client object itself was built.
    """

    def test_httpx_client_constructed_before_recording_is_captured(
        self, tmp_path: Path
    ) -> None:
        import httpx

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)

        # Built before any recording is active — mirrors an LLM client
        # constructed once at module-import time (e.g. `langgraph dev`'s
        # `make_graph()` entry point, imported once at server startup).
        pre_existing_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"pre": "existing"})
            )
        )
        try:
            with t.start_trace("pre-existing", record=True, run_id="pre-existing"):
                pre_existing_client.get("https://api.example.com/pre-existing")
        finally:
            pre_existing_client.close()

        with Fixture(tmp_path / "pre-existing" / "fixture.db") as f:
            exchanges = f.all_exchanges()

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == "https://api.example.com/pre-existing"

    async def test_httpx_async_client_constructed_before_recording_is_captured(
        self, tmp_path: Path
    ) -> None:
        import httpx

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)

        pre_existing_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"pre": "existing-async"})
            )
        )
        try:
            with t.start_trace(
                "pre-existing-async", record=True, run_id="pre-existing-async"
            ):
                await pre_existing_client.get(
                    "https://api.example.com/pre-existing-async"
                )
        finally:
            await pre_existing_client.aclose()

        with Fixture(tmp_path / "pre-existing-async" / "fixture.db") as f:
            exchanges = f.all_exchanges()

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == "https://api.example.com/pre-existing-async"

    def test_requests_session_constructed_before_recording_is_captured(
        self, tmp_path: Path
    ) -> None:
        """requests.Session.get_adapter() is already resolved per-request, so
        this already worked — locked in here as a regression guard."""
        from unittest.mock import patch as mock_patch

        import requests
        from requests.adapters import HTTPAdapter

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)
        session = requests.Session()

        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response._content = b'{"pre": "existing-requests"}'
        mock_response.url = "https://api.example.com/pre-existing-requests"

        with t.start_trace(
            "pre-existing-requests", record=True, run_id="pre-existing-requests"
        ):
            with mock_patch.object(HTTPAdapter, "send", return_value=mock_response):
                session.get("https://api.example.com/pre-existing-requests")

        with Fixture(tmp_path / "pre-existing-requests" / "fixture.db") as f:
            exchanges = f.all_exchanges()

        assert len(exchanges) == 1
        assert (
            exchanges[0]["url"] == "https://api.example.com/pre-existing-requests"
        )


class TestCallerSuppliedTransportWrapped:
    """A caller-supplied transport= (e.g. langchain-openai's TCP-keepalive
    transport, or the Anthropic SDK's default transport) must still be
    recorded, not silently bypassed the way `kwargs.setdefault(...)` would.
    """

    def test_explicit_httpx_transport_is_still_recorded(
        self, tmp_path: Path
    ) -> None:
        import httpx

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)
        custom_transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"via": "custom-transport"})
        )

        with t.start_trace("explicit-transport", record=True, run_id="explicit"):
            client = httpx.Client(transport=custom_transport)
            try:
                client.get("https://api.example.com/explicit")
            finally:
                client.close()

        with Fixture(tmp_path / "explicit" / "fixture.db") as f:
            exchanges = f.all_exchanges()

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == "https://api.example.com/explicit"

    async def test_explicit_async_httpx_transport_is_still_recorded(
        self, tmp_path: Path
    ) -> None:
        import httpx

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)
        custom_transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"via": "custom-async-transport"}
            )
        )

        with t.start_trace(
            "explicit-async-transport", record=True, run_id="explicit-async"
        ):
            client = httpx.AsyncClient(transport=custom_transport)
            try:
                await client.get("https://api.example.com/explicit-async")
            finally:
                await client.aclose()

        with Fixture(tmp_path / "explicit-async" / "fixture.db") as f:
            exchanges = f.all_exchanges()

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == "https://api.example.com/explicit-async"

    def test_requests_mounted_custom_adapter_is_still_recorded(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock

        import requests
        from requests.adapters import HTTPAdapter

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)
        session = requests.Session()

        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response._content = b'{"via": "mounted-adapter"}'
        mock_response.url = "https://api.example.com/mounted"

        custom_adapter = MagicMock(spec=HTTPAdapter)
        custom_adapter.send.return_value = mock_response
        session.mount("https://", custom_adapter)

        with t.start_trace("mounted-adapter", record=True, run_id="mounted-adapter"):
            session.get("https://api.example.com/mounted")

        custom_adapter.send.assert_called_once()
        with Fixture(tmp_path / "mounted-adapter" / "fixture.db") as f:
            exchanges = f.all_exchanges()

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == "https://api.example.com/mounted"


# ---------------------------------------------------------------------------
# Concurrent-recording isolation
# ---------------------------------------------------------------------------


class TestConcurrentRecordingIsolation:
    """Two overlapping start_trace(record=True) contexts must route HTTP
    exchanges into their own fixture, never bleed into each other's — even
    when their lifetimes genuinely overlap (e.g. two in-flight requests
    being recorded concurrently inside a server process).
    """

    def test_nested_traces_route_to_correct_fixture(self, tmp_path: Path) -> None:
        """Sequential nesting: inner trace's calls go to inner's fixture,
        and the outer trace resumes recording to its own fixture afterwards."""
        import httpx

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)

        def _handler(tag: str):
            return lambda request: httpx.Response(200, json={"tag": tag})

        with t.start_trace("outer", record=True, run_id="outer-route"):
            with httpx.Client(transport=httpx.MockTransport(_handler("outer"))) as c:
                c.get("https://api.example.com/outer-1")

            with t.start_trace("inner", record=True, run_id="inner-route"):
                with httpx.Client(
                    transport=httpx.MockTransport(_handler("inner"))
                ) as c:
                    c.get("https://api.example.com/inner-1")

            with httpx.Client(transport=httpx.MockTransport(_handler("outer"))) as c:
                c.get("https://api.example.com/outer-2")

        with Fixture(tmp_path / "outer-route" / "fixture.db") as f:
            outer_exchanges = f.all_exchanges()
        with Fixture(tmp_path / "inner-route" / "fixture.db") as f:
            inner_exchanges = f.all_exchanges()

        assert [e["url"] for e in outer_exchanges] == [
            "https://api.example.com/outer-1",
            "https://api.example.com/outer-2",
        ]
        assert [e["url"] for e in inner_exchanges] == [
            "https://api.example.com/inner-1",
        ]

    async def test_overlapping_async_traces_record_into_separate_fixtures(
        self, tmp_path: Path
    ) -> None:
        """Two genuinely concurrent (not nested) recordings, synchronized so
        both are simultaneously active when each makes its HTTP call, must
        not cross-contaminate fixtures.

        Regression test for the bug where `_install_recording_transport`
        only patched on the *first* overlapping call (nesting-counter > 1
        was a no-op) — every concurrent trace's traffic then silently landed
        in whichever fixture happened to be outermost.
        """
        import asyncio

        import httpx

        from agent_trace._replay.fixture import Fixture

        t = Tracer(trace_dir=tmp_path)
        b_entered = asyncio.Event()
        a_done = asyncio.Event()

        def handler_a(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"trace": "a"})

        def handler_b(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"trace": "b"})

        async def run_a() -> None:
            with t.start_trace("trace-a", record=True, run_id="trace-a"):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler_a)
                ) as client:
                    # Don't make our call until B has also become active —
                    # guarantees a genuine overlap window.
                    await b_entered.wait()
                    await client.get("https://api.example.com/a")
            a_done.set()

        async def run_b() -> None:
            with t.start_trace("trace-b", record=True, run_id="trace-b"):
                b_entered.set()
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler_b)
                ) as client:
                    await client.get("https://api.example.com/b")
                # Stay active until A has also finished its call, so both
                # traces are provably active at the same time.
                await a_done.wait()

        await asyncio.gather(run_a(), run_b())

        with Fixture(tmp_path / "trace-a" / "fixture.db") as f:
            exchanges_a = f.all_exchanges()
        with Fixture(tmp_path / "trace-b" / "fixture.db") as f:
            exchanges_b = f.all_exchanges()

        assert [e["url"] for e in exchanges_a] == ["https://api.example.com/a"]
        assert [e["url"] for e in exchanges_b] == ["https://api.example.com/b"]
