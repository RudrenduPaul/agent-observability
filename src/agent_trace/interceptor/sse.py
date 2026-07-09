"""
SSE-aware parsing of streamed HTTP response bodies stored in a Fixture.

RecordingTransport/AsyncRecordingTransport already capture the full raw
Server-Sent-Events text off the wire — framework-agnostic, regardless of
provider (OpenAI/Azure/Groq-style ``data: {...}`` chunks, Anthropic's
``event: ...`` + ``data: ...`` pairs, ...). What's missing is turning that
opaque text into diffable, addressable per-chunk data: today a developer has
to hand-parse ``response_body`` themselves to see what each streamed delta
actually contained.

This module is deliberately read-only and operates purely on an already
-recorded ``response_body`` string (or a fixture exchange dict) — it does not
change how RecordingTransport captures bytes off the wire. Use it against
``Fixture.all_exchanges()``/``Fixture.next_exchange()`` output, e.g.:

    from agent_trace.interceptor.sse import (
        is_sse_exchange,
        parse_sse_events,
        reconstruct_streamed_message,
    )

    for exchange in fixture.all_exchanges():
        if is_sse_exchange(exchange):
            events = parse_sse_events(exchange["response_body"])
            merged = reconstruct_streamed_message(events)
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "is_sse_exchange",
    "parse_sse_events",
    "reconstruct_streamed_message",
]

# Sentinel value some providers (OpenAI-style) send as the final SSE data
# line to signal "no more chunks" — not JSON, must be special-cased rather
# than fed to json.loads.
_DONE_SENTINEL = "[DONE]"


def is_sse_exchange(exchange: dict[str, Any]) -> bool:
    """True if *exchange* (as returned by Fixture.all_exchanges()/
    next_exchange()) looks like a captured Server-Sent-Events response.

    Checks the recorded ``Content-Type`` response header first (the
    authoritative signal); falls back to sniffing the body for an SSE
    ``data: `` line when headers are missing/lowercased differently than
    expected, since some proxies/test fixtures don't preserve casing.
    """
    headers = exchange.get("response_headers") or {}
    for key, value in headers.items():
        if str(key).lower() != "content-type":
            continue
        if "text/event-stream" in str(value).lower():
            return True
    body = exchange.get("response_body") or ""
    return _looks_like_sse_body(body)


def _looks_like_sse_body(body: str) -> bool:
    """Best-effort sniff: at least one line starts with the SSE ``data: ``
    field prefix. Deliberately conservative — a false negative just means a
    caller has to fall back to treating the body as an opaque string, which
    is the pre-existing behavior anyway."""
    for line in body.splitlines():
        if line.startswith("data:"):
            return True
    return False


def parse_sse_events(body: str) -> list[dict[str, Any] | str]:
    """Split a raw SSE response body on ``data: `` boundaries and parse each
    event's payload.

    Returns an ordered list — one entry per ``data:`` line found, in the
    order they appear in *body* (the order they were streamed). Each entry
    is either:

    - a parsed JSON object, for the common ``data: {...}`` shape every major
      provider's streaming chat-completions endpoint uses, or
    - the raw string payload, when the line isn't valid JSON (e.g. the
      ``[DONE]`` sentinel, or a provider using a non-JSON SSE payload).

    A blank body, or a body with no ``data:`` lines at all, returns ``[]`` —
    callers should treat that as "not an SSE payload" rather than an error.
    """
    events: list[dict[str, Any] | str] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == _DONE_SENTINEL:
            continue
        try:
            events.append(json.loads(payload))
        except (json.JSONDecodeError, TypeError):
            events.append(payload)
    return events


def _delta_from_event(event: dict[str, Any] | str) -> dict[str, Any] | None:
    """Extract the OpenAI-style ``choices[0].delta`` dict from one parsed SSE
    event, or None if this event isn't that shape (a non-dict event, a
    provider using a different streaming schema, etc.)."""
    if not isinstance(event, dict):
        return None
    choices = event.get("choices")
    if not choices or not isinstance(choices, list):
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    return delta if isinstance(delta, dict) else None


def reconstruct_streamed_message(
    events: list[dict[str, Any] | str],
) -> dict[str, Any]:
    """Merge an ordered list of parsed SSE events (from parse_sse_events())
    into one addressable, reassembled message.

    Concatenates every ``delta.content`` fragment into ``content``, and
    merges every ``delta.tool_calls[]`` fragment — keyed by its ``index``
    field, the same index OpenAI-style providers use to identify which
    parallel tool call a given argument fragment belongs to — into
    ``tool_calls``, concatenating each tool call's ``function.arguments``
    string across chunks in the order they arrived.

    Returns ``{"content": str, "tool_calls": {index: {...}}}``. Events that
    don't match the OpenAI-style ``choices[0].delta`` shape are ignored
    (not raised on) — this reconstruction targets that one well-known wire
    shape; a caller working with a different provider's streaming schema
    should read ``events`` directly instead.
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}

    for event in events:
        delta = _delta_from_event(event)
        if delta is None:
            continue

        content_piece = delta.get("content")
        if isinstance(content_piece, str):
            content_parts.append(content_piece)

        for tc_fragment in delta.get("tool_calls") or []:
            if not isinstance(tc_fragment, dict):
                continue
            index = tc_fragment.get("index", 0)
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = 0
            entry = tool_calls.setdefault(
                index,
                {"id": None, "function": {"name": "", "arguments": ""}},
            )
            if tc_fragment.get("id"):
                entry["id"] = tc_fragment["id"]
            fn_fragment = tc_fragment.get("function") or {}
            if fn_fragment.get("name"):
                entry["function"]["name"] += fn_fragment["name"]
            if fn_fragment.get("arguments"):
                entry["function"]["arguments"] += fn_fragment["arguments"]

    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
    }
