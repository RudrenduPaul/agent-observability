"""
Shared pytest fixtures for agent-trace tests.

Unit tests (tests/unit/): no network calls, no real HTTP, uses FixtureClock
Integration tests (tests/integration/): real APIs, tagged @pytest.mark.integration
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_trace._replay.fixture import Fixture
from agent_trace.core.clock import FixtureClock
from agent_trace.core.span import Span
from agent_trace.core.trace import Trace


@pytest.fixture
def tmp_fixture_path(tmp_path: Path) -> Path:
    """Return a temp path for a fixture.db (file not created yet)."""
    return tmp_path / "fixture.db"


@pytest.fixture
def sample_fixture(tmp_path: Path) -> Fixture:
    """Return a Fixture pre-populated with 3 recorded HTTP exchanges.

    Exchanges:
      1. POST https://api.openai.com/v1/chat/completions  → 200
      2. POST https://api.anthropic.com/v1/messages       → 200
      3. GET  https://api.example.com/tool-result         → 200 {"result": "tool output"}
    """
    fixture = Fixture(tmp_path / "sample_fixture.db", trace_id="test-trace-001")

    # Exchange 1 — OpenAI chat completions
    openai_body = json.dumps(
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from fixture",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    fixture.record_exchange(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        request_headers={
            "content-type": "application/json",
            "authorization": "Bearer sk-test",
        },
        request_body=json.dumps(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        ),
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body=openai_body,
    )

    # Exchange 2 — Anthropic messages
    anthropic_body = json.dumps(
        {
            "id": "msg_01abc",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi from Anthropic"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 6},
        }
    )
    fixture.record_exchange(
        url="https://api.anthropic.com/v1/messages",
        method="POST",
        request_headers={
            "content-type": "application/json",
            "x-api-key": "sk-ant-test",
        },
        request_body=json.dumps(
            {"model": "claude-3-5-sonnet-20241022", "messages": []}
        ),
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body=anthropic_body,
    )

    # Exchange 3 — tool result
    fixture.record_exchange(
        url="https://api.example.com/tool-result",
        method="GET",
        request_headers={"accept": "application/json"},
        request_body="",
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body=json.dumps({"result": "tool output"}),
    )

    return fixture


@pytest.fixture
def fixture_clock() -> FixtureClock:
    """Return a FixtureClock starting at 1_000_000.0."""
    clock = FixtureClock()
    clock.advance(1_000_000.0)
    return clock


@pytest.fixture
def sample_span(fixture_clock: FixtureClock) -> Span:
    """Return a Span created with the fixture clock installed."""
    from agent_trace.core.clock import restore_clock, set_clock

    token = set_clock(fixture_clock)
    try:
        span = Span(name="sample-span")
    finally:
        restore_clock(token)
    return span


@pytest.fixture
def sample_trace() -> Trace:
    """Return a Trace with 3 spans in parent → child → grandchild structure."""
    trace = Trace(trace_id="trace-abc", run_id="run-abc")

    root = Span(name="root", span_id="span-root", trace_id="trace-abc", parent_id=None)
    child = Span(
        name="child", span_id="span-child", trace_id="trace-abc", parent_id="span-root"
    )
    grandchild = Span(
        name="grandchild",
        span_id="span-grandchild",
        trace_id="trace-abc",
        parent_id="span-child",
    )

    root.end()
    child.end()
    grandchild.end()

    trace.add_span(root)
    trace.add_span(child)
    trace.add_span(grandchild)

    return trace
