"""
Unit tests for agent_trace.interceptor.aiohttp_hook.

make_recording_request.

Uses aiohttp.test_utils.TestServer to serve real (loopback) HTTP responses
instead of a third-party mocking library, so these tests exercise the real
aiohttp request/response machinery agent-trace is patching (ClientResponse
body caching semantics, header types, etc.) rather than a mocked stand-in.

Tests monkey-patch `aiohttp.ClientSession._request` directly (the same
class-level patch `Tracer._patch_aiohttp` installs) rather than subclassing
`aiohttp.ClientSession` — the installed aiohttp raises
`DeprecationWarning: Inheritance class ... from ClientSession is
discouraged` for any subclass, so aiohttp_hook.py deliberately has no
subclass to test.
"""

from __future__ import annotations

import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.aiohttp_hook import make_recording_request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture(tmp_path) -> Fixture:
    return Fixture(tmp_path / "aiohttp_test.db", trace_id="test-aiohttp-trace")


async def _echo_handler(request: web.Request) -> web.Response:
    """Echo the request body back, plus a stable model field."""
    body = await request.text()
    return web.json_response({"echo": body, "model": "gpt-4o"}, status=200)


async def _not_found_handler(request: web.Request) -> web.Response:
    return web.json_response({"error": "not found"}, status=404)


async def _ok_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


@pytest.fixture
async def server():
    """A real (loopback) aiohttp server for GET/POST/404 test routes."""
    app = web.Application()
    app.router.add_get("/get-test", _ok_handler)
    app.router.add_post("/v1/chat/completions", _echo_handler)
    app.router.add_get("/not-found", _not_found_handler)
    srv = TestServer(app)
    await srv.start_server()
    try:
        yield srv
    finally:
        await srv.close()


@pytest.fixture
def patched_client_session(request):
    """Monkey-patch aiohttp.ClientSession._request with a recording wrapper
    for the duration of one test, restoring the original afterwards -- the
    same install/uninstall shape as Tracer._patch_aiohttp/_unpatch_aiohttp.
    """
    fixture_holder: dict = {}

    def _install(fixture: Fixture) -> None:
        orig_request = aiohttp.ClientSession._request
        fixture_holder["orig"] = orig_request
        aiohttp.ClientSession._request = make_recording_request(fixture, orig_request)

    def _uninstall() -> None:
        if "orig" in fixture_holder:
            aiohttp.ClientSession._request = fixture_holder["orig"]

    yield _install
    _uninstall()


# ---------------------------------------------------------------------------
# make_recording_request
# ---------------------------------------------------------------------------


class TestMakeRecordingRequest:
    async def test_records_get_request(
        self, tmp_path, server, patched_client_session
    ) -> None:
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/get-test")
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            assert response.status == 200

        assert fixture.exchange_count() == 1
        exchanges = fixture.all_exchanges()
        assert exchanges[0]["url"] == str(url)
        assert exchanges[0]["method"] == "GET"
        fixture.close()

    async def test_records_post_request_with_json_body(
        self, tmp_path, server, patched_client_session
    ) -> None:
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/v1/chat/completions")
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"model": "gpt-4o", "messages": []})

        exchanges = fixture.all_exchanges()
        assert len(exchanges) == 1
        assert exchanges[0]["method"] == "POST"
        assert "gpt-4o" in exchanges[0]["request_body"]
        fixture.close()

    async def test_preserves_response_status_code(
        self, tmp_path, server, patched_client_session
    ) -> None:
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/not-found")
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            assert response.status == 404

        assert fixture.all_exchanges()[0]["response_status"] == 404
        fixture.close()

    async def test_preserves_response_body(
        self, tmp_path, server, patched_client_session
    ) -> None:
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/v1/chat/completions")
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"model": "gpt-4o"})

        recorded_body = fixture.all_exchanges()[0]["response_body"]
        recorded_data = json.loads(recorded_body)
        assert recorded_data["model"] == "gpt-4o"
        fixture.close()

    async def test_caller_can_still_read_response_after_recording(
        self, tmp_path, server, patched_client_session
    ) -> None:
        """The recorder eagerly reads the body; the caller must still be able to."""
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/get-test")
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            # Body was already consumed internally by the recorder; caller
            # must still be able to read it via the cached ClientResponse._body.
            data = await response.json()
            assert data == {"status": "ok"}
            text = await response.text()
            assert json.loads(text) == {"status": "ok"}

        fixture.close()

    async def test_records_request_headers(
        self, tmp_path, server, patched_client_session
    ) -> None:
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/get-test")
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            await session.get(url, headers={"Authorization": "Bearer sk-test"})

        req_headers = fixture.all_exchanges()[0]["request_headers"]
        assert req_headers.get("Authorization") == "Bearer sk-test"
        fixture.close()

    async def test_records_data_bytes_body(
        self, tmp_path, server, patched_client_session
    ) -> None:
        """`data=` bytes (e.g. LiteLLM's aiohttp transport pre-serializes JSON
        to bytes rather than passing `json=`) is decoded, not dropped."""
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/v1/chat/completions")
        raw = b'{"model": "gpt-4o", "messages": []}'
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            await session.post(
                url, data=raw, headers={"content-type": "application/json"}
            )

        assert fixture.all_exchanges()[0]["request_body"] == raw.decode()
        fixture.close()

    async def test_multiple_requests_all_recorded(
        self, tmp_path, server, patched_client_session
    ) -> None:
        fixture = _make_fixture(tmp_path)
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            await session.get(server.make_url("/get-test"))
            await session.post(server.make_url("/v1/chat/completions"), json={})
            await session.get(server.make_url("/not-found"))

        assert fixture.exchange_count() == 3
        fixture.close()

    async def test_unpatch_restores_original_behaviour(
        self, tmp_path, server, patched_client_session
    ) -> None:
        """After unpatching, new sessions are no longer recorded."""
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/get-test")

        orig_request = aiohttp.ClientSession._request
        aiohttp.ClientSession._request = make_recording_request(fixture, orig_request)
        aiohttp.ClientSession._request = orig_request  # immediately unpatch

        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            assert response.status == 200

        assert fixture.exchange_count() == 0
        fixture.close()

    async def test_original_request_still_invoked(
        self, tmp_path, server, patched_client_session
    ) -> None:
        """The wrapped call must actually reach the network, not short-circuit."""
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/v1/chat/completions")
        patched_client_session(fixture)

        async with aiohttp.ClientSession() as session:
            response = await session.post(url, json={"model": "gpt-4o"})
            data = await response.json()
            assert data["model"] == "gpt-4o"

        fixture.close()

    async def test_no_fixture_entry_when_never_patched(self, tmp_path, server) -> None:
        """Sanity check: a plain, unpatched ClientSession records nothing."""
        fixture = _make_fixture(tmp_path)
        url = server.make_url("/get-test")

        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            assert response.status == 200

        assert fixture.exchange_count() == 0
        fixture.close()
