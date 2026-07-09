"""
ReplayEngine — orchestrates deterministic replay of a recorded agent run.

The engine does three things in the right order:
1. Installs FixtureClock so all get_time() calls return recorded timestamps.
2. Patches httpx.Client, requests.Session, and (when installed) grpc's
   channel factories so outbound HTTP/gRPC calls are served from the fixture
   instead of hitting real endpoints.
3. Tears everything down in a finally block so no patch leaks out.

Why monkey-patch rather than dependency-inject?
AI SDKs construct their own httpx.Client / requests.Session instances
internally.  We cannot inject transports into them without forking each SDK.
Patching httpx.Client.__init__ and requests.Session.get_adapter is the least
invasive approach that works across Anthropic, OpenAI, and other SDK clients
without modification.
"""

from __future__ import annotations

import logging
import unittest.mock
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from contextvars import Token
from pathlib import Path
from typing import Any

import httpx

from agent_trace._replay.fixture import Fixture
from agent_trace.core.clock import FixtureClock, restore_clock, set_clock
from agent_trace.interceptor.httpx_hook import AsyncReplayTransport, ReplayTransport

__all__ = ["ReplayEngine", "replay_context"]

logger = logging.getLogger(__name__)


def _build_grpc_replay_patch(fixture: Fixture) -> Any:
    """Return a context manager that patches grpc's sync channel factories.

    grpc is an optional dependency (installed transitively by SDKs such as
    google-generativeai / google-cloud-aiplatform). grpc.insecure_channel /
    secure_channel are plain module-level functions rather than a shared
    base-class method, so we patch the module attributes directly -- see
    grpc_hook.py's module docstring for why this is the correct interception
    point. Returns nullcontext() when grpc isn't installed.
    """
    try:
        import grpc as _grpc

        from agent_trace.interceptor.grpc_hook import GRPCReplayInterceptor

        orig_insecure = _grpc.insecure_channel
        orig_secure = _grpc.secure_channel

        def _patched_insecure(
            target: str, options: Any = None, compression: Any = None
        ) -> Any:
            channel = orig_insecure(target, options=options, compression=compression)
            return _grpc.intercept_channel(
                channel, GRPCReplayInterceptor(fixture, target)
            )

        def _patched_secure(
            target: str,
            credentials: Any,
            options: Any = None,
            compression: Any = None,
        ) -> Any:
            channel = orig_secure(
                target, credentials, options=options, compression=compression
            )
            return _grpc.intercept_channel(
                channel, GRPCReplayInterceptor(fixture, target)
            )

        return unittest.mock.patch.multiple(
            _grpc,
            insecure_channel=_patched_insecure,
            secure_channel=_patched_secure,
        )
    except ImportError:
        return nullcontext()


def _build_grpc_aio_replay_patch(fixture: Fixture) -> Any:
    """Return a context manager that patches grpc.aio's channel factories.

    Unary-unary only -- see grpc_hook.py's module docstring for why async
    streaming RPCs are out of scope for this pass. Returns nullcontext()
    when grpc isn't installed.
    """
    try:
        from grpc import aio as _grpc_aio

        from agent_trace.interceptor.grpc_hook import AsyncGRPCReplayInterceptor

        orig_insecure = _grpc_aio.insecure_channel
        orig_secure = _grpc_aio.secure_channel

        def _patched_insecure(target: str, **kwargs: Any) -> Any:
            interceptors = list(kwargs.pop("interceptors", None) or [])
            interceptors.append(AsyncGRPCReplayInterceptor(fixture, target))
            return orig_insecure(target, interceptors=interceptors, **kwargs)

        def _patched_secure(target: str, credentials: Any, **kwargs: Any) -> Any:
            interceptors = list(kwargs.pop("interceptors", None) or [])
            interceptors.append(AsyncGRPCReplayInterceptor(fixture, target))
            return orig_secure(target, credentials, interceptors=interceptors, **kwargs)

        return unittest.mock.patch.multiple(
            _grpc_aio,
            insecure_channel=_patched_insecure,
            secure_channel=_patched_secure,
        )
    except ImportError:
        return nullcontext()


class ReplayEngine:
    """Coordinates fixture loading, clock replacement, and transport patching.

    Parameters
    ----------
    fixture_path:
        Path to the SQLite fixture file produced by a recording run.
    """

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path

    @contextmanager
    def replay(self) -> Generator[Fixture, None, None]:
        """Context manager that activates full replay mode.

        Yields the open Fixture so callers can inspect exchange counts or
        advance the FixtureClock manually between spans.

        Usage::

            engine = ReplayEngine(Path("fixtures/run.db"))
            with engine.replay() as fixture:
                # All httpx and requests calls are served from fixture.
                # All get_time() calls return recorded timestamps.
                result = my_agent.run(prompt)
        """
        with Fixture(self._fixture_path) as fixture:
            fixture.reset_read_cursor()

            clock = FixtureClock(initial=fixture.earliest_timestamp())
            token: Token[Any] = set_clock(clock)

            # --- httpx patch -----------------------------------------------
            # httpx.Client (sync) uses ReplayTransport; httpx.AsyncClient uses
            # AsyncReplayTransport.  Injecting a sync BaseTransport into an
            # AsyncClient silently succeeds at init but raises AttributeError on
            # the first request — hence the separate async variant.
            # The clock is threaded through so each served exchange advances
            # the FixtureClock, reproducing recorded execution timing.
            original_httpx_init = httpx.Client.__init__
            original_httpx_async_init = httpx.AsyncClient.__init__

            def patched_httpx_init(
                client_self: httpx.Client, *args: Any, **kwargs: Any
            ) -> None:
                kwargs.setdefault("transport", ReplayTransport(fixture, clock=clock))
                original_httpx_init(client_self, *args, **kwargs)

            def patched_httpx_async_init(
                client_self: httpx.AsyncClient, *args: Any, **kwargs: Any
            ) -> None:
                kwargs.setdefault(
                    "transport", AsyncReplayTransport(fixture, clock=clock)
                )
                original_httpx_async_init(client_self, *args, **kwargs)

            # --- requests patch (optional) ---------------------------------
            # requests is an optional dependency.  When absent, use nullcontext
            # so the with-statement below is a no-op.
            try:
                import requests as _requests

                from agent_trace.interceptor.requests_patch import ReplayAdapter

                def patched_get_adapter(
                    session_self: Any, url: str, **kwargs: Any
                ) -> Any:
                    return ReplayAdapter(fixture)

                requests_patch: Any = unittest.mock.patch.object(
                    _requests.Session, "get_adapter", patched_get_adapter
                )
            except ImportError:
                requests_patch = nullcontext()

            # --- grpc patch (optional) --------------------------------------
            grpc_patch = _build_grpc_replay_patch(fixture)
            grpc_aio_patch = _build_grpc_aio_replay_patch(fixture)

            try:
                with (
                    unittest.mock.patch.object(
                        httpx.Client, "__init__", patched_httpx_init
                    ),
                    unittest.mock.patch.object(
                        httpx.AsyncClient, "__init__", patched_httpx_async_init
                    ),
                    requests_patch,
                    grpc_patch,
                    grpc_aio_patch,
                ):
                    logger.debug(
                        "agent-trace replay active: fixture=%s exchanges=%d",
                        self._fixture_path,
                        fixture.exchange_count(),
                    )
                    yield fixture
            finally:
                restore_clock(token)

    def fixture_exchange_count(self) -> int:
        """Return the number of recorded exchanges in the fixture.

        Opens and closes the fixture transiently — use sparingly; prefer
        checking inside a replay() block where the fixture is already open.
        """
        with Fixture(self._fixture_path) as f:
            return f.exchange_count()


@contextmanager
def replay_context(fixture_path: Path) -> Generator[Fixture, None, None]:
    """Convenience wrapper around ReplayEngine.replay().

    Usage::

        with replay_context(Path("fixtures/run.db")) as fixture:
            result = my_agent.run(prompt)
    """
    engine = ReplayEngine(fixture_path)
    with engine.replay() as fixture:
        yield fixture
