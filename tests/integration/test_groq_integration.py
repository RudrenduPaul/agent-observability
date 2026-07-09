"""
Integration test verifying agent-trace's httpx interceptor against the
*real* ``groq`` Python SDK — not just a reproduction of its httpx-client
construction pattern.

Context (code-changes-backlog.md, "Verify HTTP-interceptor capture against
the real groq Python SDK"): the claim that agent-trace's global
``httpx.Client``/``httpx.AsyncClient`` monkeypatch (``Tracer._patch_httpx``)
also intercepts Groq SDK traffic previously rested only on a static read of
the ``groq`` PyPI package's ``_base_client.py`` (confirming
``SyncAPIClient``/``AsyncAPIClient`` never pass an explicit ``transport=``
kwarg, so the patch's ``kwargs.setdefault``-based interception would apply)
— never a live or mocked end-to-end capture. This test closes that gap by
actually installing the ``groq`` package, running a real ``groq.Groq``/
``groq.AsyncGroq`` client through ``Tracer.start_trace(record=True)``, and
asserting the exchange lands in ``fixture.db``.

No live network call is made — the HTTP layer is mocked with ``respx`` so
this test is deterministic, fast, and needs no ``GROQ_API_KEY``. This
matches the backlog item's own "real or mocked" phrasing: what was missing
was *any* reproducible test exercising the real SDK's HTTP client through
RecordingTransport, not specifically a live API call.

Run with: uv run pytest tests/integration/test_groq_integration.py
Requires: pip install agent-trace[groq]  (or: pip install groq)
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

pytest.importorskip("groq", reason="groq not installed (pip install agent-trace[groq])")

from agent_trace import Tracer
from agent_trace._replay.fixture import Fixture
from agent_trace.interceptor.httpx_hook import (
    AsyncRecordingTransport,
    RecordingTransport,
)

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

_GROQ_RESPONSE_JSON = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "model": "llama-3.1-8b-instant",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from groq"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


@pytest.mark.integration
class TestGroqSdkCapturedByGlobalMonkeypatch:
    """Exercises the real global ``Tracer._patch_httpx`` monkeypatch
    (installed via ``start_trace(record=True)``, exactly as a real agent
    would use it) against the real ``groq`` SDK's internal httpx client —
    not a hand-constructed transport."""

    @respx.mock
    def test_sync_groq_client_is_captured(self, tmp_path: Path) -> None:
        import groq

        respx.post(_GROQ_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_GROQ_RESPONSE_JSON)
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("groq-sync-test", record=True, run_id="groq-sync"):
            client = groq.Groq(api_key="test-key")
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response.choices[0].message.content == "hello from groq"

        with Fixture(tmp_path / "groq-sync" / "fixture.db") as fixture:
            exchanges = fixture.all_exchanges()

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == _GROQ_CHAT_URL
        assert exchanges[0]["method"] == "POST"
        assert "llama-3.1-8b-instant" in exchanges[0]["request_body"]
        assert exchanges[0]["response_status"] == 200
        # Confirms the real Groq SDK's internal httpx.Client actually went
        # through RecordingTransport, not a stray fallback path.
        assert "hello from groq" in exchanges[0]["response_body"]

    @respx.mock
    async def test_async_groq_client_is_captured(self, tmp_path: Path) -> None:
        import groq

        respx.post(_GROQ_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_GROQ_RESPONSE_JSON)
        )

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("groq-async-test", record=True, run_id="groq-async"):
            client = groq.AsyncGroq(api_key="test-key")
            response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response.choices[0].message.content == "hello from groq"

        with Fixture(tmp_path / "groq-async" / "fixture.db") as fixture:
            exchanges = fixture.all_exchanges()

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == _GROQ_CHAT_URL
        assert exchanges[0]["method"] == "POST"

    @respx.mock
    def test_pre_existing_groq_client_is_still_captured(self, tmp_path: Path) -> None:
        """Confirms the request-dispatch-time patch (`_transport_for_url`,
        not `__init__`) captures a Groq client constructed *before*
        recording activates — the deployment shape typical of a
        `langgraph dev`/`make_graph()` entry point importing its model
        client once at process start."""
        import groq

        respx.post(_GROQ_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_GROQ_RESPONSE_JSON)
        )

        client = groq.Groq(api_key="test-key")  # constructed before recording

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("groq-pre-existing", record=True, run_id="groq-pre"):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response.choices[0].message.content == "hello from groq"

        with Fixture(tmp_path / "groq-pre" / "fixture.db") as fixture:
            assert fixture.exchange_count() == 1

    @respx.mock
    def test_groq_client_transport_is_recording_transport_while_active(
        self, tmp_path: Path
    ) -> None:
        """Directly asserts the isinstance() check the backlog item asked
        for: while recording is active, the Groq SDK's internal httpx
        client's resolved transport is a RecordingTransport instance."""
        import groq

        respx.post(_GROQ_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_GROQ_RESPONSE_JSON)
        )

        t = Tracer(trace_dir=tmp_path)
        client = groq.Groq(api_key="test-key")
        inner_httpx_client = client._client  # groq.Groq._client is an httpx.Client subclass

        with t.start_trace("groq-isinstance-check", record=True, run_id="groq-iso"):
            resolved = inner_httpx_client._transport_for_url(
                httpx.URL(_GROQ_CHAT_URL)
            )
            assert isinstance(resolved, RecordingTransport)

    @respx.mock
    async def test_async_groq_client_transport_is_async_recording_transport(
        self, tmp_path: Path
    ) -> None:
        import groq

        respx.post(_GROQ_CHAT_URL).mock(
            return_value=httpx.Response(200, json=_GROQ_RESPONSE_JSON)
        )

        t = Tracer(trace_dir=tmp_path)
        client = groq.AsyncGroq(api_key="test-key")
        inner_httpx_client = client._client

        with t.start_trace("groq-async-isinstance-check", record=True, run_id="groq-aiso"):
            resolved = inner_httpx_client._transport_for_url(
                httpx.URL(_GROQ_CHAT_URL)
            )
            assert isinstance(resolved, AsyncRecordingTransport)
