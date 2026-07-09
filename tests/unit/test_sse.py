"""
Unit tests for agent_trace.interceptor.sse — SSE-aware parsing of streamed
HTTP response bodies stored in a Fixture.
"""

from __future__ import annotations

from agent_trace.interceptor.sse import (
    is_sse_exchange,
    parse_sse_events,
    reconstruct_streamed_message,
)

# ---------------------------------------------------------------------------
# is_sse_exchange
# ---------------------------------------------------------------------------


class TestIsSseExchange:
    def test_true_via_content_type_header(self) -> None:
        exchange = {
            "response_headers": {"Content-Type": "text/event-stream; charset=utf-8"},
            "response_body": "",
        }
        assert is_sse_exchange(exchange) is True

    def test_content_type_header_matching_is_case_insensitive(self) -> None:
        exchange = {
            "response_headers": {"content-type": "TEXT/EVENT-STREAM"},
            "response_body": "",
        }
        assert is_sse_exchange(exchange) is True

    def test_true_via_body_sniff_when_headers_missing(self) -> None:
        exchange = {
            "response_headers": {},
            "response_body": 'data: {"a": 1}\n\n',
        }
        assert is_sse_exchange(exchange) is True

    def test_false_for_plain_json_body(self) -> None:
        exchange = {
            "response_headers": {"content-type": "application/json"},
            "response_body": '{"a": 1}',
        }
        assert is_sse_exchange(exchange) is False

    def test_false_for_empty_exchange(self) -> None:
        assert is_sse_exchange({}) is False


# ---------------------------------------------------------------------------
# parse_sse_events
# ---------------------------------------------------------------------------


class TestParseSseEvents:
    def test_parses_json_data_lines_in_order(self) -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        )
        events = parse_sse_events(body)
        assert events == [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
        ]

    def test_skips_done_sentinel(self) -> None:
        body = 'data: {"a": 1}\n\ndata: [DONE]\n\n'
        events = parse_sse_events(body)
        assert events == [{"a": 1}]

    def test_non_json_payload_kept_as_raw_string(self) -> None:
        body = "data: not-json-at-all\n\n"
        events = parse_sse_events(body)
        assert events == ["not-json-at-all"]

    def test_ignores_non_data_lines(self) -> None:
        body = 'event: message\ndata: {"a": 1}\nid: 42\n\n'
        events = parse_sse_events(body)
        assert events == [{"a": 1}]

    def test_empty_body_returns_empty_list(self) -> None:
        assert parse_sse_events("") == []

    def test_blank_data_line_skipped(self) -> None:
        body = "data: \n\ndata: {\"a\": 1}\n\n"
        assert parse_sse_events(body) == [{"a": 1}]


# ---------------------------------------------------------------------------
# reconstruct_streamed_message
# ---------------------------------------------------------------------------


class TestReconstructStreamedMessage:
    def test_merges_content_deltas(self) -> None:
        events = [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
        ]
        result = reconstruct_streamed_message(events)
        assert result["content"] == "Hello"
        assert result["tool_calls"] == {}

    def test_merges_tool_call_argument_fragments_by_index(self) -> None:
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "get_w", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city"'}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ':"SF"}'}}
                            ]
                        }
                    }
                ]
            },
        ]
        result = reconstruct_streamed_message(events)
        assert result["tool_calls"] == {
            0: {
                "id": "call_1",
                "function": {"name": "get_w", "arguments": '{"city":"SF"}'},
            }
        }

    def test_handles_parallel_tool_calls_by_distinct_index(self) -> None:
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {"name": "a", "arguments": "1"},
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "function": {"name": "b", "arguments": "2"},
                                },
                            ]
                        }
                    }
                ]
            },
        ]
        result = reconstruct_streamed_message(events)
        assert set(result["tool_calls"].keys()) == {0, 1}
        assert result["tool_calls"][0]["id"] == "call_a"
        assert result["tool_calls"][1]["id"] == "call_b"

    def test_ignores_non_openai_shaped_events(self) -> None:
        events = ["raw string event", {"unrelated": True}, 42]
        result = reconstruct_streamed_message(events)
        assert result == {"content": "", "tool_calls": {}}

    def test_empty_events_list(self) -> None:
        assert reconstruct_streamed_message([]) == {"content": "", "tool_calls": {}}


# ---------------------------------------------------------------------------
# End-to-end: a fixture-shaped exchange through the full pipeline
# ---------------------------------------------------------------------------


def test_end_to_end_fixture_exchange_to_reconstructed_message() -> None:
    exchange = {
        "response_headers": {"content-type": "text/event-stream"},
        "response_body": (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            "data: [DONE]\n\n"
        ),
    }
    assert is_sse_exchange(exchange)
    events = parse_sse_events(exchange["response_body"])
    merged = reconstruct_streamed_message(events)
    assert merged["content"] == "Hello"
