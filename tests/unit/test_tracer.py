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
    def test_nested_record_true_does_not_double_patch(self, tmp_path: Path) -> None:
        """Two nested start_trace(record=True) must not double-patch httpx."""
        import httpx

        t = Tracer(trace_dir=tmp_path)
        original_init = httpx.Client.__init__

        with t.start_trace("outer", record=True, run_id="outer"):
            patched_init = httpx.Client.__init__
            assert patched_init is not original_init

            with t.start_trace("inner", record=True, run_id="inner"):
                # The patch should NOT have changed (no double-wrap)
                assert httpx.Client.__init__ is patched_init

            # After inner exits, outer's patch should still be in place
            assert httpx.Client.__init__ is patched_init

        # After outer exits, original is restored
        assert httpx.Client.__init__ is original_init

    def test_depth_returns_to_zero_after_nested_exit(self, tmp_path: Path) -> None:
        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("outer", record=True, run_id="outer-d"):
            with t.start_trace("inner", record=True, run_id="inner-d"):
                assert t._transport_depth == 2
            assert t._transport_depth == 1
        assert t._transport_depth == 0
