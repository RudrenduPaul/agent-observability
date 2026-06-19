"""
Unit tests for agent_trace.core.clock.

Invariant: ALL spans use get_time() from this module — never time.time() directly.
The FixtureClock enables deterministic replay by controlling what get_time() returns.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent_trace.core.clock import (
    FixtureClock,
    WallClock,
    get_clock,
    get_time,
    restore_clock,
    set_clock,
)


class TestWallClock:
    def test_now_returns_float(self) -> None:
        clock = WallClock()
        result = clock.now()
        assert isinstance(result, float)

    def test_now_close_to_time_time(self) -> None:
        clock = WallClock()
        before = time.time()
        result = clock.now()
        after = time.time()
        assert before <= result <= after + 0.01

    def test_now_increases_over_time(self) -> None:
        clock = WallClock()
        t1 = clock.now()
        time.sleep(0.01)
        t2 = clock.now()
        assert t2 > t1


class TestFixtureClock:
    def test_default_now_returns_zero(self) -> None:
        clock = FixtureClock()
        assert clock.now() == 0.0

    def test_advance_updates_now(self) -> None:
        clock = FixtureClock()
        clock.advance(1_000_000.0)
        assert clock.now() == 1_000_000.0

    def test_advance_to_arbitrary_value(self) -> None:
        clock = FixtureClock()
        clock.advance(9999.5)
        assert clock.now() == 9999.5

    def test_advance_can_go_backwards(self) -> None:
        """advance() is a setter, not monotonic — allows replaying any timestamp."""
        clock = FixtureClock()
        clock.advance(500.0)
        clock.advance(100.0)
        assert clock.now() == 100.0

    def test_does_not_use_wall_time(self) -> None:
        """FixtureClock must NOT call time.time() — its value only changes via advance()."""
        clock = FixtureClock()
        t1 = clock.now()
        time.sleep(0.05)
        t2 = clock.now()
        # Both reads should be identical (0.0) because we never called advance()
        assert t1 == t2 == 0.0

    def test_advance_then_stable(self) -> None:
        clock = FixtureClock()
        clock.advance(42.0)
        # Reading multiple times returns same value until advance() is called again
        assert clock.now() == clock.now() == 42.0


class TestSetClock:
    def test_set_clock_changes_get_clock(self) -> None:
        fixture_clock = FixtureClock()
        token = set_clock(fixture_clock)
        try:
            assert get_clock() is fixture_clock
        finally:
            restore_clock(token)

    def test_set_clock_returns_token(self) -> None:
        fixture_clock = FixtureClock()
        token = set_clock(fixture_clock)
        assert token is not None
        restore_clock(token)

    def test_restore_clock_undoes_set(self) -> None:
        original = get_clock()
        fixture_clock = FixtureClock()
        token = set_clock(fixture_clock)
        restore_clock(token)
        assert get_clock() is original

    def test_restore_clock_multiple_nesting(self) -> None:
        clock_a = FixtureClock()
        clock_b = FixtureClock()

        token_a = set_clock(clock_a)
        assert get_clock() is clock_a

        token_b = set_clock(clock_b)
        assert get_clock() is clock_b

        restore_clock(token_b)
        assert get_clock() is clock_a

        restore_clock(token_a)


class TestGetTime:
    def test_get_time_returns_float(self) -> None:
        result = get_time()
        assert isinstance(result, float)

    def test_get_time_uses_active_clock(self) -> None:
        clock = FixtureClock()
        clock.advance(12345.678)
        token = set_clock(clock)
        try:
            assert get_time() == 12345.678
        finally:
            restore_clock(token)

    def test_get_time_after_set_fixture_clock_returns_zero_not_wall_time(self) -> None:
        """Core invariant: after set_clock(FixtureClock()), get_time() returns 0.0."""
        clock = FixtureClock()  # default _current = 0.0
        token = set_clock(clock)
        try:
            result = get_time()
            assert result == 0.0
            # Confirm it is NOT wall time
            wall = time.time()
            assert result != pytest.approx(wall, abs=1.0)
        finally:
            restore_clock(token)

    def test_get_time_updates_after_advance(self) -> None:
        clock = FixtureClock()
        token = set_clock(clock)
        try:
            assert get_time() == 0.0
            clock.advance(999.0)
            assert get_time() == 999.0
        finally:
            restore_clock(token)


class TestClockPerContext:
    def test_clock_is_per_context_var_not_global(self) -> None:
        """Changing the clock in one thread must not affect another thread.

        ContextVar is inherited from the parent at thread creation time, but
        mutations within the child are isolated.
        """
        results: dict[str, float] = {}
        barrier = threading.Barrier(2)

        def thread_a() -> None:
            clock_a = FixtureClock()
            clock_a.advance(1111.0)
            token = set_clock(clock_a)
            barrier.wait()  # sync: both threads now have different clocks set
            results["a"] = get_time()
            restore_clock(token)

        def thread_b() -> None:
            clock_b = FixtureClock()
            clock_b.advance(2222.0)
            token = set_clock(clock_b)
            barrier.wait()
            results["b"] = get_time()
            restore_clock(token)

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert results["a"] == 1111.0
        assert results["b"] == 2222.0
        # The two threads had independent clocks
        assert results["a"] != results["b"]

    def test_main_thread_clock_unaffected_by_child_thread(self) -> None:
        """A child thread's clock mutation must not bleed back to the main thread."""
        original_time = get_time()

        def child() -> None:
            clock = FixtureClock()
            clock.advance(99999.0)
            set_clock(clock)
            # deliberately do NOT restore — in the child this leaks, but
            # the main thread should be unaffected because ContextVar is
            # not shared back.

        t = threading.Thread(target=child)
        t.start()
        t.join()

        # Main thread's clock should be unchanged
        after_child = get_time()
        # Both should be wall-clock-ish (not 99999.0)
        assert after_child != 99999.0
        assert abs(after_child - original_time) < 5.0  # within 5 seconds
