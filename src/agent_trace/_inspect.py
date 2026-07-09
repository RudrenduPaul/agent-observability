"""
Pattern-check library backing ``agent-trace inspect`` and related CLI
diagnostics.

Every function here is a pure, read-only analysis over already-captured data
(``Fixture.all_exchanges()`` rows and/or ``trace.json`` span dicts) — none of
them touch the network, mutate the fixture, or require any framework to be
installed. They turn the manual "read raw JSON, notice the anomaly" step a
developer does today into an automated flag, which is the recurring gap this
module closes across a long list of GitHub issues (see
``[redacted]`` for the full issue-by-issue mapping).

Every check function returns a ``list[dict]`` of "flags" — plain dicts with
at least a ``"check"`` key (this function's name) and a human-readable
``"detail"`` string, plus whatever exchange/span identifying fields make
sense for that check (``url``, ``method``, ``sequence_num``, ``span``, ...).
Checks never raise on malformed/unexpected input shapes — a body that isn't
valid JSON, or doesn't match the shape a check is looking for, is simply not
flagged, the same best-effort tradeoff the rest of agent-trace's interceptor
layer already makes.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "check_action_name_not_registered",
    "check_anthropic_thinking_in_tool_result",
    "check_duplicate_json_blocks",
    "check_empty_content_not_final",
    "check_endpoint_host_mismatch",
    "check_get_post_field_mismatch",
    "check_json_schema_lookaround_or_anyof",
    "check_malformed_tool_call_arguments",
    "check_missing_extra_kwarg",
    "check_missing_tool_call_id",
    "check_null_content_with_tool_calls",
    "check_orphaned_tool_call_ids",
    "check_reserved_kwarg_collision",
    "check_stream_merge_validity",
    "check_tool_call_boundary_leak",
    "check_tool_call_name_dotted_compound",
    "check_tool_call_name_fuzzy_match",
    "check_tool_calling_disabled",
    "check_tools_with_response_format",
    "detect_response_shape_anomalies",
    "field_present_on_wire_absent_downstream",
    "find_near_duplicate_sibling_content",
    "flag_4xx_5xx_exchanges",
    "match_known_error_patterns",
    "multi_block_llm_responses",
    "run_all_exchange_checks",
]


def _loads(body: str | None) -> Any:
    """``json.loads`` that returns ``None`` instead of raising on anything
    that isn't valid JSON (empty body, plain text, truncated stream, ...)."""
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _flag(
    check: str, exchange: dict[str, Any], detail: str, **extra: Any
) -> dict[str, Any]:
    row = {
        "check": check,
        "url": exchange.get("url"),
        "method": exchange.get("method"),
        "sequence_num": exchange.get("sequence_num"),
        "detail": detail,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# 1. Orphaned tool_call_ids — requested (assistant tool_calls[].id) vs
#    responded-to (tool-role messages' tool_call_id) within one exchange's
#    request body message list. (#531)
# ---------------------------------------------------------------------------


def check_orphaned_tool_call_ids(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        messages = body.get("messages")
        if not isinstance(messages, list):
            continue

        requested: set[str] = set()
        responded: set[str] = set()
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                    requested.add(tc["id"])
            if msg.get("role") == "tool" and isinstance(msg.get("tool_call_id"), str):
                responded.add(msg["tool_call_id"])

        orphaned = requested - responded
        if orphaned:
            flags.append(
                _flag(
                    "orphaned_tool_call_ids",
                    exchange,
                    f"{len(orphaned)} tool_call_id(s) requested but never "
                    f"responded to: {sorted(orphaned)}",
                    orphaned_ids=sorted(orphaned),
                )
            )
    return flags


# ---------------------------------------------------------------------------
# 2. Provider tool-call-boundary leak markers, e.g. "to=functions." (#7845)
# ---------------------------------------------------------------------------

_BOUNDARY_LEAK_MARKERS = ("to=functions.", "to=multi_tool_use.", "<|python_tag|>")


def check_tool_call_boundary_leak(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = exchange.get("response_body") or ""
        for marker in _BOUNDARY_LEAK_MARKERS:
            if marker in body:
                flags.append(
                    _flag(
                        "tool_call_boundary_leak",
                        exchange,
                        f"provider tool-call-boundary marker {marker!r} leaked into "
                        "response body content",
                        marker=marker,
                    )
                )
                break
    return flags


# ---------------------------------------------------------------------------
# 3. json.loads validity on reconstructed tool_calls[].function.arguments —
#    catches concatenated/malformed streaming fragments. (#6843)
# ---------------------------------------------------------------------------


def check_malformed_tool_call_arguments(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue
        choices = body.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            for tc in message.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if not isinstance(args, str) or not args:
                    continue
                if _loads(args) is None:
                    flags.append(
                        _flag(
                            "malformed_tool_call_arguments",
                            exchange,
                            f"tool_calls[].function.arguments for "
                            f"{fn.get('name', '<unnamed>')!r} is not valid JSON: "
                            f"{args[:120]!r}",
                            tool_name=fn.get("name"),
                        )
                    )
    return flags


# ---------------------------------------------------------------------------
# 4. Response message content is null/non-string alongside function_call /
#    tool_calls field. (#6761)
# ---------------------------------------------------------------------------


def check_null_content_with_tool_calls(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue
        for choice in body.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            has_calls = bool(message.get("tool_calls") or message.get("function_call"))
            content = message.get("content")
            if has_calls and content is not None and not isinstance(content, str):
                flags.append(
                    _flag(
                        "null_content_with_tool_calls",
                        exchange,
                        "message.content is non-string while a function_call/"
                        "tool_calls field is present (content type: "
                        f"{type(content).__name__})",
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# 5. Request URL host doesn't match the framework's configured endpoint.
#    (#5204)
# ---------------------------------------------------------------------------


def check_endpoint_host_mismatch(
    exchanges: list[dict[str, Any]], configured_host: str
) -> list[dict[str, Any]]:
    import httpx

    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        url = exchange.get("url") or ""
        try:
            host = httpx.URL(url).host
        except Exception:
            logger.debug(
                "agent-trace inspect: could not parse URL %r", url, exc_info=True
            )
            continue
        if host and host != configured_host:
            flags.append(
                _flag(
                    "endpoint_host_mismatch",
                    exchange,
                    f"request host {host!r} does not match configured endpoint "
                    f"{configured_host!r}",
                    actual_host=host,
                    configured_host=configured_host,
                )
            )
    return flags


# ---------------------------------------------------------------------------
# 6. Request body carries both `tools` and `response_format`/`response_model`
#    simultaneously. (#5472)
# ---------------------------------------------------------------------------


def check_tools_with_response_format(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        has_tools = bool(body.get("tools"))
        has_response_shape = bool(
            body.get("response_format") or body.get("response_model")
        )
        if has_tools and has_response_shape:
            flags.append(
                _flag(
                    "tools_with_response_format",
                    exchange,
                    "request body carries both `tools` and a `response_format`/"
                    "`response_model` constraint simultaneously",
                )
            )
    return flags


# ---------------------------------------------------------------------------
# 7. Anthropic request body: a `type: "thinking"` block nested inside a
#    `tool_result.content` array. (#4175)
# ---------------------------------------------------------------------------


def check_anthropic_thinking_in_tool_result(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                nested = block.get("content")
                if not isinstance(nested, list):
                    continue
                for inner in nested:
                    if isinstance(inner, dict) and inner.get("type") == "thinking":
                        flags.append(
                            _flag(
                                "anthropic_thinking_in_tool_result",
                                exchange,
                                "a `thinking` content block is nested inside a "
                                "`tool_result.content` array — malformed shape "
                                "Anthropic's API rejects with a 400",
                            )
                        )
    return flags


# ---------------------------------------------------------------------------
# 8. Assistant-role message with empty content ([] or "") that isn't the
#    final message in the array. (#3168)
# ---------------------------------------------------------------------------


def check_empty_content_not_final(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        last_index = len(messages) - 1
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            is_empty = content in ("", [])
            if is_empty and i != last_index:
                flags.append(
                    _flag(
                        "empty_content_not_final",
                        exchange,
                        f"assistant message at index {i} has empty content and is "
                        "not the final message — Anthropic rejects this shape with "
                        '"all messages must have non-empty content"',
                        message_index=i,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# 9. Plain-text ReAct-style "Action: <name>" line naming a tool not in the
#    run's registered tool list (formatting noise stripped first). (#22358)
# ---------------------------------------------------------------------------

_ACTION_LINE_RE = re.compile(r"^\s*Action:\s*(.+?)\s*$", re.MULTILINE)


def _strip_formatting_noise(name: str) -> str:
    return name.strip().strip("`'\"[]").strip()


def check_action_name_not_registered(
    exchanges: list[dict[str, Any]], registered_tools: set[str]
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = exchange.get("response_body") or ""
        for match in _ACTION_LINE_RE.finditer(body):
            name = _strip_formatting_noise(match.group(1))
            if name and name not in registered_tools:
                flags.append(
                    _flag(
                        "action_name_not_registered",
                        exchange,
                        f"plain-text `Action: {name}` line names a tool not in the "
                        "run's registered tool list",
                        action_name=name,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# 10. JSON-schema `pattern` containing regex lookaround, or an `anyOf` branch
#     mixing incompatible JSON types under `strict: true`. (#5508)
# ---------------------------------------------------------------------------

_LOOKAROUND_MARKERS = ("(?=", "(?!", "(?<=", "(?<!")

_JSON_SCALAR_TYPES = {"string", "number", "integer", "boolean", "null"}


def _walk_schema(node: Any) -> Any:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_schema(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_schema(item)


def check_json_schema_lookaround_or_anyof(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        strict = bool(body.get("strict")) or "strict" in json.dumps(body).lower()
        for node in _walk_schema(body):
            pattern = node.get("pattern")
            has_lookaround = isinstance(pattern, str) and any(
                m in pattern for m in _LOOKAROUND_MARKERS
            )
            if has_lookaround:
                flags.append(
                    _flag(
                        "json_schema_lookaround",
                        exchange,
                        "JSON-schema `pattern` uses regex lookaround syntax: "
                        f"{pattern!r} — OpenAI's structured-output path rejects "
                        "this construct",
                        pattern=pattern,
                    )
                )
            any_of = node.get("anyOf")
            if strict and isinstance(any_of, list) and len(any_of) > 1:
                types: set[str] = {
                    branch["type"]
                    for branch in any_of
                    if isinstance(branch, dict) and isinstance(branch.get("type"), str)
                }
                if len(types & _JSON_SCALAR_TYPES) > 1:
                    flags.append(
                        _flag(
                            "json_schema_anyof_type_mismatch",
                            exchange,
                            f"`anyOf` branch mixes incompatible JSON scalar types "
                            f"{sorted(types)} under strict mode",
                            types=sorted(types),
                        )
                    )
    return flags


# ---------------------------------------------------------------------------
# 11. Near-identical/overlapping JSON blocks repeated within a single
#     captured request/response body. (#4919)
# ---------------------------------------------------------------------------


def _top_level_blocks(body: Any) -> list[Any]:
    """Best-effort extraction of a list of "blocks" worth de-duplication
    from a parsed JSON body — the `input`/`messages`/`content` array most
    provider request shapes use for repeated structured items."""
    if isinstance(body, dict):
        for key in ("input", "messages", "content"):
            value = body.get(key)
            if isinstance(value, list):
                return value
    return []


def check_duplicate_json_blocks(
    exchanges: list[dict[str, Any]], min_repeats: int = 3
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        for field_name in ("request_body", "response_body"):
            body = _loads(exchange.get(field_name))
            blocks = _top_level_blocks(body)
            if len(blocks) < min_repeats:
                continue
            counts: dict[str, int] = {}
            for block in blocks:
                try:
                    key = json.dumps(block, sort_keys=True)
                except TypeError:
                    continue
                counts[key] = counts.get(key, 0) + 1
            duplicated = {k: v for k, v in counts.items() if v >= min_repeats}
            if duplicated:
                total_dupes = sum(duplicated.values())
                flags.append(
                    _flag(
                        "duplicate_json_blocks",
                        exchange,
                        f"{field_name} contains {len(duplicated)} distinct block(s) "
                        f"repeated {total_dupes} times total out of {len(blocks)} "
                        "top-level blocks",
                        field=field_name,
                        distinct_duplicated_blocks=len(duplicated),
                        total_repeats=total_dupes,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# 12. Expected extra-config kwarg silently absent from the wire request.
#     (#18635)
# ---------------------------------------------------------------------------


def _get_path(obj: Any, path: str) -> tuple[bool, Any]:
    """Return (found, value) for a dotted path like
    'extra_body.chat_template_kwargs.thinking'."""
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def check_missing_extra_kwarg(
    exchanges: list[dict[str, Any]], kwarg_path: str
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        found, _value = _get_path(body, kwarg_path)
        if not found:
            flags.append(
                _flag(
                    "missing_extra_kwarg",
                    exchange,
                    f"expected kwarg path {kwarg_path!r} is absent from the request "
                    "body that actually reached the wire",
                    kwarg_path=kwarg_path,
                )
            )
    return flags


# ---------------------------------------------------------------------------
# 13/14. Tool-call name checks against a run's registered tool list: fuzzy
#        (edit-distance) near-miss (#7170), and dotted-compound-of-two-
#        registered-names (#9688).
# ---------------------------------------------------------------------------


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = cur
    return prev[-1]


def _extract_tool_call_names(exchange: dict[str, Any]) -> list[str]:
    body = _loads(exchange.get("response_body"))
    names: list[str] = []
    if not isinstance(body, dict):
        return names
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        for tc in message.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name")
                if isinstance(name, str):
                    names.append(name)
    return names


def check_tool_call_name_fuzzy_match(
    exchanges: list[dict[str, Any]],
    registered_tools: set[str],
    max_distance: int = 3,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        for name in _extract_tool_call_names(exchange):
            if name in registered_tools:
                continue
            best = min(
                registered_tools,
                key=lambda r: _edit_distance(name, r),
                default=None,
            )
            if best is None:
                continue
            dist = _edit_distance(name, best)
            if 0 < dist <= max_distance:
                flags.append(
                    _flag(
                        "tool_call_name_fuzzy_match",
                        exchange,
                        f"tool call name {name!r} is not registered — nearest "
                        f"registered name is {best!r} (edit distance {dist})",
                        called_name=name,
                        nearest_registered_name=best,
                        edit_distance=dist,
                    )
                )
    return flags


def check_tool_call_name_dotted_compound(
    exchanges: list[dict[str, Any]], registered_tools: set[str]
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        for name in _extract_tool_call_names(exchange):
            if name in registered_tools or "." not in name:
                continue
            parts = name.split(".")
            if len(parts) == 2 and all(p in registered_tools for p in parts):
                flags.append(
                    _flag(
                        "tool_call_name_dotted_compound",
                        exchange,
                        f"tool call name {name!r} is a dotted compound of two "
                        f"registered tool names: {parts}",
                        called_name=name,
                        compound_parts=parts,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# 15. choices[].message.tool_calls[] entries missing an `id` key (or
#     `id: null`) before the framework constructs a ToolMessage/ToolCall.
#     (#3992)
# ---------------------------------------------------------------------------


def check_missing_tool_call_id(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue
        for choice in body.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            for tc in message.get("tool_calls") or []:
                if isinstance(tc, dict) and not tc.get("id"):
                    fn = tc.get("function") or {}
                    flags.append(
                        _flag(
                            "missing_tool_call_id",
                            exchange,
                            f"tool_calls[] entry for {fn.get('name', '<unnamed>')!r} "
                            "is missing an `id` (or `id` is null)",
                            tool_name=fn.get("name"),
                        )
                    )
    return flags


# ---------------------------------------------------------------------------
# 16. A field retrieved via an earlier GET doesn't match the value sent for
#     that same field in a later, causally-related POST referencing the same
#     resource id. (#2620)
# ---------------------------------------------------------------------------


def check_get_post_field_mismatch(
    exchanges: list[dict[str, Any]],
    field_path: str,
    get_id_field: str = "id",
    post_id_field: str | None = None,
) -> list[dict[str, Any]]:
    """*post_id_field* defaults to *get_id_field* — pass it explicitly when
    the POST body references the resource under a different key than the
    GET response used (e.g. GET .../assistants/{id} returns ``{"id": ...}``
    while POST .../runs sends ``{"assistant_id": ...}`` referencing the same
    resource — the real OpenAI Assistants API shape behind #2620)."""
    resolved_post_id_field = post_id_field or get_id_field
    flags: list[dict[str, Any]] = []
    known_values: dict[str, Any] = {}  # resource id -> value seen via GET

    ordered = sorted(exchanges, key=lambda e: e.get("sequence_num", 0))
    for exchange in ordered:
        method = (exchange.get("method") or "").upper()
        if method == "GET":
            body = _loads(exchange.get("response_body"))
            if not isinstance(body, dict):
                continue
            resource_id = body.get(get_id_field)
            found, value = _get_path(body, field_path)
            if resource_id is not None and found:
                known_values[str(resource_id)] = value
        elif method == "POST":
            body = _loads(exchange.get("request_body"))
            if not isinstance(body, dict):
                continue
            resource_id = body.get(resolved_post_id_field)
            if resource_id is None:
                continue
            found, sent_value = _get_path(body, field_path)
            key = str(resource_id)
            if found and key in known_values and known_values[key] != sent_value:
                flags.append(
                    _flag(
                        "get_post_field_mismatch",
                        exchange,
                        f"POST for resource {key!r} sends {field_path}="
                        f"{sent_value!r}, which differs from the value last seen "
                        f"via GET: {known_values[key]!r}",
                        resource_id=key,
                        field_path=field_path,
                        get_value=known_values[key],
                        post_value=sent_value,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# Companion diagnostics (distinct [redacted]s, same "raw capture with zero
# automated diagnosis" gap that motivates the big `inspect` cluster above).
# ---------------------------------------------------------------------------


def flag_4xx_5xx_exchanges(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Auto-flag 4xx/5xx HTTP exchanges as errors instead of leaving them as
    undifferentiated raw rows indistinguishable from a normal 200."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        status = exchange.get("response_status")
        if isinstance(status, int) and status >= 400:
            flags.append(
                _flag(
                    "http_error_status",
                    exchange,
                    f"HTTP {status} response",
                    status=status,
                )
            )
    return flags


def check_tool_calling_disabled(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a captured LLM request where tool-calling was explicitly
    disabled (Gemini `tool_config.function_calling_config.mode`, or
    OpenAI/Anthropic `tool_choice`) while `tools` were also declared —
    the exact "tool calling silently disabled by the client itself" shape
    behind #18937."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        has_tools = bool(body.get("tools"))
        if not has_tools:
            continue

        found, mode = _get_path(body, "tool_config.function_calling_config.mode")
        if found and isinstance(mode, str) and mode.upper() == "NONE":
            flags.append(
                _flag(
                    "tool_calling_disabled",
                    exchange,
                    "request declares `tools` but "
                    "`tool_config.function_calling_config.mode` is NONE — "
                    "tool calling was explicitly disabled by the client",
                    provider="gemini",
                )
            )

        tool_choice = body.get("tool_choice")
        if isinstance(tool_choice, str) and tool_choice.lower() == "none":
            flags.append(
                _flag(
                    "tool_calling_disabled",
                    exchange,
                    "request declares `tools` but `tool_choice` is 'none' — "
                    "tool calling was explicitly disabled by the client",
                    provider="openai/anthropic",
                )
            )
    return flags


def match_known_error_patterns(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scan ERROR-status spans' exception.message text for recognizable
    cross-provider error shapes and surface a flagged summary line, starting
    with "Unsupported content block type" (#36531)."""
    patterns: dict[str, str] = {
        "Unsupported content block type": "cross_provider_content_block_mismatch",
        "all messages must have non-empty content": "empty_content_message",
        "invalid_request_error": "provider_invalid_request",
    }
    flags: list[dict[str, Any]] = []
    for span in spans:
        if span.get("status") != "ERROR":
            continue
        for event in span.get("events") or []:
            if not isinstance(event, dict) or event.get("name") != "exception":
                continue
            attrs = event.get("attributes") or {}
            message = str(attrs.get("exception.message", ""))
            for substring, label in patterns.items():
                if substring in message:
                    flags.append(
                        {
                            "check": "known_error_pattern",
                            "span": span.get("name"),
                            "pattern": label,
                            "detail": f"span {span.get('name')!r} exception text "
                            f"matches known pattern {label!r} ({substring!r})",
                        }
                    )
    return flags


_RESERVED_KWARGS = frozenset({"config", "runtime", "store", "writer", "state"})
_MISSING_ARG_RE = re.compile(r"missing \d+ required positional argument[s]?: '(.+?)'")


def check_reserved_kwarg_collision(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag a failed tool call whose exception message shows a missing
    positional argument that is also a framework-reserved kwarg name AND was
    present in the tool's captured input — the "tool argument shadowed by a
    framework-injected kwarg" pattern (#34029)."""
    flags: list[dict[str, Any]] = []
    for span in spans:
        if span.get("status") != "ERROR":
            continue
        attrs = span.get("attributes") or {}
        input_str = str(attrs.get("tool.input_str", attrs.get("tool.input", "")))
        for event in span.get("events") or []:
            if not isinstance(event, dict) or event.get("name") != "exception":
                continue
            message = str((event.get("attributes") or {}).get("exception.message", ""))
            match = _MISSING_ARG_RE.search(message)
            if not match:
                continue
            missing_name = match.group(1)
            if missing_name in _RESERVED_KWARGS and missing_name in input_str:
                flags.append(
                    {
                        "check": "reserved_kwarg_collision",
                        "span": span.get("name"),
                        "reserved_kwarg": missing_name,
                        "detail": f"span {span.get('name')!r}: missing positional "
                        f"argument {missing_name!r} is a framework-reserved kwarg "
                        "name that was also present in the tool's own input — "
                        "likely shadowed by a framework-injected kwarg",
                    }
                )
    return flags


def multi_block_llm_responses(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag a response whose content contains more than one top-level
    text/message block (e.g. OpenAI Responses API's `phase: commentary` /
    `phase: final_answer` pattern) — naive concatenation of these blocks
    produces invalid output (#36290)."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue
        output = body.get("output")
        if isinstance(output, list):
            text_blocks = [
                item
                for item in output
                if isinstance(item, dict) and item.get("type") in ("message", "text")
            ]
            phases = {
                item.get("phase")
                for item in text_blocks
                if isinstance(item, dict) and item.get("phase")
            }
            if len(text_blocks) > 1 and (len(phases) > 1 or not phases):
                flags.append(
                    _flag(
                        "multi_block_response",
                        exchange,
                        f"response `output` contains {len(text_blocks)} top-level "
                        "text/message blocks — naive concatenation would produce "
                        "invalid output",
                        block_count=len(text_blocks),
                        phases=sorted(p for p in phases if p),
                    )
                )
    return flags


def check_stream_merge_validity(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stream-vs-merged tool-call diff mode: reconstruct the merged tool
    call(s) from a captured SSE exchange's raw deltas and flag any merged
    `function.arguments` that doesn't parse as valid JSON — the exact "wire
    deltas were correct, your framework's merge logic mangled them" shape
    behind #5165."""
    from agent_trace.interceptor.sse import (
        is_sse_exchange,
        parse_sse_events,
        reconstruct_streamed_message,
    )

    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        if not is_sse_exchange(exchange):
            continue
        events = parse_sse_events(exchange.get("response_body") or "")
        if not events:
            continue
        merged = reconstruct_streamed_message(events)
        for index, tool_call in sorted(merged.get("tool_calls", {}).items()):
            fn = tool_call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str) and args and _loads(args) is None:
                flags.append(
                    _flag(
                        "stream_merge_invalid_json",
                        exchange,
                        f"merged tool_call[{index}] ({fn.get('name', '<unnamed>')!r}) "
                        f"arguments do not parse as valid JSON after stream-merge: "
                        f"{args[:120]!r}",
                        tool_call_index=index,
                        tool_name=fn.get("name"),
                    )
                )
    return flags


def detect_response_shape_anomalies(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag captured exchanges where top-level JSON fields are null/missing
    while a populated nested dict field exists, and/or where repeated calls
    to the same URL return materially different top-level key sets across a
    run (#531, #3994, #1442)."""
    flags: list[dict[str, Any]] = []

    shapes_by_url: dict[str, set[frozenset[str]]] = {}
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue
        url = str(exchange.get("url"))
        shapes_by_url.setdefault(url, set()).add(frozenset(body.keys()))

        # null/missing top-level with populated nested dict.
        for key, value in body.items():
            if value not in (None,):
                continue
            for other_key, other_value in body.items():
                not_a_populated_dict = (
                    not isinstance(other_value, dict) or not other_value
                )
                if other_key == key or not_a_populated_dict:
                    continue
                flags.append(
                    _flag(
                        "null_field_with_populated_nested",
                        exchange,
                        f"top-level field {key!r} is null/missing while nested "
                        f"dict field {other_key!r} is populated",
                        null_field=key,
                        populated_field=other_key,
                    )
                )
                break

    for url, shapes in shapes_by_url.items():
        if len(shapes) > 1:
            flags.append(
                {
                    "check": "inconsistent_response_shape",
                    "url": url,
                    "detail": f"{len(shapes)} distinct top-level key sets seen "
                    f"across responses to {url}",
                    "distinct_shape_count": len(shapes),
                }
            )
    return flags


def field_present_on_wire_absent_downstream(
    exchanges: list[dict[str, Any]], spans: list[dict[str, Any]], field_name: str
) -> list[dict[str, Any]]:
    """Diff/inspect: compare a raw captured LLM response field against the
    framework's final serialized span attributes, flagging when a field
    present on the wire never appears downstream (#3936 — "is Azure
    returning usage data that something downstream strips")."""
    wire_has_field = any(
        isinstance(body := _loads(e.get("response_body")), dict) and field_name in body
        for e in exchanges
    )
    if not wire_has_field:
        return []

    span_has_field = False
    for span in spans:
        attrs = span.get("attributes") or {}
        if any(str(k).startswith(f"llm.{field_name}") for k in attrs):
            span_has_field = True
            break

    if wire_has_field and not span_has_field:
        return [
            {
                "check": "field_present_on_wire_absent_downstream",
                "field": field_name,
                "detail": f"field {field_name!r} is present in captured raw HTTP "
                "response bodies but no span attribute reflects it — likely "
                "stripped somewhere between the wire and the framework's final "
                "serialized message",
            }
        ]
    return []


def find_near_duplicate_sibling_content(
    spans: list[dict[str, Any]], threshold: float = 0.9
) -> list[dict[str, Any]]:
    """CLI content-diff view: compare a tool span's captured output text
    against its parent/sibling llm span's captured content text and flag
    near-identical/overlapping content as "same content surfaced via two
    message types" (#3062)."""
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        by_parent.setdefault(span.get("parent_id"), []).append(span)

    flags: list[dict[str, Any]] = []
    for siblings in by_parent.values():
        tool_spans = [
            s
            for s in siblings
            if str(s.get("name", "")).startswith("tool:")
            and (s.get("attributes") or {}).get("tool.output")
        ]
        llm_spans = [
            s
            for s in siblings
            if str(s.get("name", "")).startswith("llm:")
            and (s.get("attributes") or {}).get("llm.content")
        ]
        for tool_span in tool_spans:
            tool_text = str((tool_span.get("attributes") or {}).get("tool.output", ""))
            for llm_span in llm_spans:
                llm_attrs = llm_span.get("attributes") or {}
                llm_text = str(llm_attrs.get("llm.content", ""))
                if not tool_text or not llm_text:
                    continue
                ratio = difflib.SequenceMatcher(None, tool_text, llm_text).ratio()
                if ratio >= threshold:
                    flags.append(
                        {
                            "check": "near_duplicate_sibling_content",
                            "tool_span": tool_span.get("name"),
                            "llm_span": llm_span.get("name"),
                            "similarity": round(ratio, 3),
                            "detail": f"{tool_span.get('name')!r} output and "
                            f"{llm_span.get('name')!r} content are "
                            f"{ratio * 100:.0f}% similar — likely the same content "
                            "surfaced via two message types",
                        }
                    )
    return flags


def content_hash(method: str, url: str, request_body: str) -> str:
    """Stable identity for "this is the same logical request" — used to
    group retry attempts (see Fixture's attempt_group column).

    sha1 here is a content-identity fingerprint, not a security boundary —
    usedforsecurity=False documents that and avoids FIPS-mode failures.
    """
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(method.upper().encode("utf-8"))
    digest.update(b"\0")
    digest.update(url.encode("utf-8"))
    digest.update(b"\0")
    digest.update((request_body or "").encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Orchestration: run every exchange-scoped check that needs no extra
# framework-specific arguments (registered tool lists, configured hosts,
# expected kwarg paths, ... are opt-in via dedicated CLI flags instead).
# ---------------------------------------------------------------------------


def run_all_exchange_checks(
    exchanges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Run every parameter-free exchange-scoped check and return
    ``{check_name: [flags]}`` for every check that found at least one flag."""
    checks: dict[str, Any] = {
        "orphaned_tool_call_ids": check_orphaned_tool_call_ids,
        "tool_call_boundary_leak": check_tool_call_boundary_leak,
        "malformed_tool_call_arguments": check_malformed_tool_call_arguments,
        "null_content_with_tool_calls": check_null_content_with_tool_calls,
        "tools_with_response_format": check_tools_with_response_format,
        "anthropic_thinking_in_tool_result": check_anthropic_thinking_in_tool_result,
        "empty_content_not_final": check_empty_content_not_final,
        "json_schema_lookaround_or_anyof": check_json_schema_lookaround_or_anyof,
        "duplicate_json_blocks": check_duplicate_json_blocks,
        "missing_tool_call_id": check_missing_tool_call_id,
        "http_error_status": flag_4xx_5xx_exchanges,
        "tool_calling_disabled": check_tool_calling_disabled,
        "multi_block_response": multi_block_llm_responses,
        "stream_merge_invalid_json": check_stream_merge_validity,
        "response_shape_anomaly": detect_response_shape_anomalies,
    }
    results: dict[str, list[dict[str, Any]]] = {}
    for name, fn in checks.items():
        flags = fn(exchanges)
        if flags:
            results[name] = flags
    return results
