"""
ReplayEngine — orchestrates deterministic replay of a recorded agent run.

The engine does three things in the right order:
1. Installs FixtureClock so all get_time() calls return recorded timestamps.
2. Patches httpx.Client and requests.Session so outbound HTTP calls are served
   from the fixture instead of hitting real endpoints.
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
from contextlib import contextmanager
from contextvars import Token
from pathlib import Path
from typing import Any

import httpx

from agent_trace.core.clock import FixtureClock, restore_clock, set_clock
from agent_trace.interceptor.httpx_hook import ReplayTransport
from agent_trace.replay.fixture import Fixture

__all__ = ["ReplayEngine", "replay_context"]

logger = logging.getLogger(__name__)


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
        fixture = Fixture(self._fixture_path)
        fixture.reset_read_cursor()

        clock = FixtureClock()
        token: Token[Any] = set_clock(clock)

        # --- httpx patch ---------------------------------------------------
        # We patch httpx.Client.__init__ to inject our ReplayTransport as the
        # default transport.  The original __init__ is called first so SSL,
        # timeout, and other settings are preserved.
        original_httpx_init = httpx.Client.__init__

        def patched_httpx_init(
            client_self: httpx.Client, *args: Any, **kwargs: Any
        ) -> None:
            # Inject our transport before the real __init__ can set the default
            # so SDK-created clients pick it up without any other change.
            kwargs.setdefault("transport", ReplayTransport(fixture))
            original_httpx_init(client_self, *args, **kwargs)

        try:
            # requests is optional — only patch if installed
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
                requests_patch = unittest.mock.MagicMock()
                requests_patch.__enter__ = lambda s: None
                requests_patch.__exit__ = lambda s, *a: None

            with (
                unittest.mock.patch.object(
                    httpx.Client, "__init__", patched_httpx_init
                ),
                requests_patch,
            ):
                logger.debug(
                    "agent-trace replay active: fixture=%s exchanges=%d",
                    self._fixture_path,
                    fixture.exchange_count(),
                )
                yield fixture
        finally:
            restore_clock(token)
            fixture.close()

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
