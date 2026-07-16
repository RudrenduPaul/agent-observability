"""
Pattern-check library backing ``agent-trace inspect`` and related CLI
diagnostics.

Every function here is a pure, read-only analysis over already-captured data
(``Fixture.all_exchanges()`` rows and/or ``trace.json`` span dicts) — none of
them touch the network, mutate the fixture, or require any framework to be
installed. They turn the manual "read raw JSON, notice the anomaly" step a
developer does today into an automated flag, closing a recurring gap seen
across a long list of real-world GitHub issues from downstream frameworks.

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
import itertools
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "check_action_name_not_registered",
    "check_all_tool_calls_no_terminal_response",
    "check_anthropic_thinking_in_tool_result",
    "check_content_block_missing_type",
    "check_duplicate_concurrent_tool_calls",
    "check_duplicate_json_blocks",
    "check_empty_content_not_final",
    "check_endpoint_host_mismatch",
    "check_forced_tool_call_unfulfilled",
    "check_get_post_field_mismatch",
    "check_json_schema_lookaround_or_anyof",
    "check_malformed_tool_call_arguments",
    "check_markdown_fenced_json_response",
    "check_missing_extra_kwarg",
    "check_missing_tool_call_id",
    "check_non_ok_finish_reason",
    "check_null_content_with_tool_calls",
    "check_null_or_missing_sse_delta",
    "check_orphaned_tool_call_ids",
    "check_phantom_tool_call",
    "check_reserved_kwarg_collision",
    "check_restart_vs_resume",
    "check_stream_merge_validity",
    "check_system_prompt_dropped",
    "check_tool_call_boundary_leak",
    "check_tool_call_name_absent_from_request_tools",
    "check_tool_call_name_dotted_compound",
    "check_tool_call_name_fuzzy_match",
    "check_tool_call_name_not_registered",
    "check_tool_calling_disabled",
    "check_tools_with_response_format",
    "detect_response_shape_anomalies",
    "field_present_on_wire_absent_downstream",
    "find_near_duplicate_sibling_content",
    "flag_4xx_5xx_exchanges",
    "match_known_error_patterns",
    "multi_block_llm_responses",
    "check_orphaned_responses_api_call_ids",
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
# 1b. The same orphaned-id shape as above, but for the OpenAI Responses API's
#     distinct message shape: a flat `input` list of `function_call`/
#     `function_call_output` items keyed by `call_id`, instead of Chat
#     Completions' nested `messages[].tool_calls[].id` /
#     `messages[].tool_call_id` shape check_orphaned_tool_call_ids and
#     check_missing_tool_call_id both assume (#33895 — "No call message
#     found for call_*", the Responses-API-specific pairing failure neither
#     of those two checks can see since they never look at `input`/
#     `function_call`/`function_call_output` at all).
# ---------------------------------------------------------------------------


def check_orphaned_responses_api_call_ids(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        items = body.get("input")
        if not isinstance(items, list):
            continue

        requested: set[str] = set()
        responded: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                continue
            if item.get("type") == "function_call":
                requested.add(call_id)
            elif item.get("type") == "function_call_output":
                responded.add(call_id)

        orphaned = requested - responded
        if orphaned:
            flags.append(
                _flag(
                    "orphaned_responses_api_call_ids",
                    exchange,
                    f"{len(orphaned)} Responses API call_id(s) requested "
                    f"(function_call) but never responded to "
                    f"(function_call_output): {sorted(orphaned)}",
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
# 4b. Request-side malformed content-array shape: a `messages[].content`
#     list item that is a dict missing a `type` key — the exact shape a
#     tool returning a raw list of dicts (e.g. Tavily's
#     `[{"url": ..., "content": ...}]`) produces once it lands back in a
#     tool-role message's `content` array without ever being normalized
#     into a proper content block. Distinct from check_null_content_with_
#     tool_calls (#6761), which inspects RESPONSE message content, not
#     REQUEST-side tool-message content blocks. (#1069)
# ---------------------------------------------------------------------------


def check_content_block_missing_type(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a captured request body where a `messages[].content` array
    contains a dict item with no `type` key — the exact malformed shape
    behind #1069's `messages[3].content[0].type` `BadRequestError`, produced
    when a tool (there, Tavily) returns a plain list of dicts that gets
    threaded straight into a tool-message's `content` array with no
    normalization into a proper `{"type": ..., ...}` content block."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("request_body"))
        if not isinstance(body, dict):
            continue
        messages = body.get("messages")
        if not isinstance(messages, list):
            continue
        for msg_index, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block_index, block in enumerate(content):
                if isinstance(block, dict) and "type" not in block:
                    flags.append(
                        _flag(
                            "content_block_missing_type",
                            exchange,
                            f"messages[{msg_index}].content[{block_index}] is a "
                            "dict content block with no `type` key — malformed "
                            "shape a naive json.loads()-then-forward of a raw "
                            "tool return value produces (e.g. a tool returning "
                            "`[{'url': ..., 'content': ...}]` with no `type` "
                            "field)",
                            message_index=msg_index,
                            content_index=block_index,
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
    'extra_body.chat_template_kwargs.thinking'.

    Also resolves numeric segments as list indices, so a path like
    'choices.0.message.reasoning_content' walks into a JSON array the same
    way it would walk into a nested dict — needed for provider response
    shapes where the field of interest is nested inside a list (e.g.
    DeepSeek's ``choices[0].message.reasoning_content``, #5526)."""
    current = obj
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        elif isinstance(current, list):
            if not re.fullmatch(r"-?\d+", part):
                return False, None
            index = int(part)
            if not (-len(current) <= index < len(current)):
                return False, None
            current = current[index]
        else:
            return False, None
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
# 14b. Structured-JSON-field sibling of check_action_name_not_registered
#      (which reads plain-text `Action: <name>` lines): flags ANY captured
#      `tool_calls[].function.name` not present in the run's registered
#      tool list, unconditionally — no edit-distance threshold. Distinct
#      from check_tool_call_name_fuzzy_match's near-miss-only scope (#325 —
#      no retry mechanism for ModelBehaviorError when the model hallucinates
#      a nonexistent tool name; this surfaces every such hallucination, not
#      just the ones close enough to a real name to be a plausible typo).
# ---------------------------------------------------------------------------


def check_tool_call_name_not_registered(
    exchanges: list[dict[str, Any]], registered_tools: set[str]
) -> list[dict[str, Any]]:
    """Flag every captured `tool_calls[].function.name` not present in
    *registered_tools*, with no edit-distance/near-miss threshold — the
    unconditional counterpart to check_tool_call_name_fuzzy_match (#325)."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        for name in _extract_tool_call_names(exchange):
            if name not in registered_tools:
                flags.append(
                    _flag(
                        "tool_call_name_not_registered",
                        exchange,
                        f"tool call name {name!r} is not in the run's "
                        "registered tool list",
                        called_name=name,
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
# 17. More than one tool_calls[] entry naming the same tool within a single
#     assistant turn — a candidate "non-reentrant tool invoked concurrently"
#     pattern (#6882: AutoGen's parallel_tool_calls=True calling the same
#     team/tool twice in one turn).
# ---------------------------------------------------------------------------


def check_duplicate_concurrent_tool_calls(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag an assistant-message response whose ``tool_calls[]`` names the
    same tool more than once — the exact shape behind #6882, where a
    maintainer confirmed the root cause was "calling the same team
    concurrently" once ``parallel_tool_calls=True`` was enabled. Any tool/
    wrapper that isn't safely reentrant under parallel tool calling can hit
    this, not just AutoGen's team-as-tool pattern."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue
        for choice in body.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or len(tool_calls) < 2:
                continue

            names: dict[str, int] = {}
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name")
                if isinstance(name, str) and name:
                    names[name] = names.get(name, 0) + 1

            duplicated = {name: count for name, count in names.items() if count > 1}
            if duplicated:
                flags.append(
                    _flag(
                        "duplicate_concurrent_tool_calls",
                        exchange,
                        f"tool_calls[] in one assistant turn names the same "
                        f"tool more than once — candidate non-reentrant "
                        f"tool invoked concurrently: {duplicated}",
                        duplicated_tool_counts=duplicated,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# 18. A whole run consisting of nothing but tool-call-only responses, with
#     no terminal (final, human-readable) response ever emitted — the
#     infinite-tool-call-loop shape #3097 asked to auto-flag after the
#     fact, distinct from RecordingTransport's live loop_guard_threshold
#     (which only fires *during* an active recording session).
# ---------------------------------------------------------------------------


def check_all_tool_calls_no_terminal_response(
    exchanges: list[dict[str, Any]],
    min_exchanges: int = 2,
) -> list[dict[str, Any]]:
    """Flag a run where every captured chat-completion exchange is a
    tool-call-only response (per the same heuristic RecordingTransport's
    live loop guard uses) and none ever produced a terminal, non-tool-call
    response — i.e. the run's own recorded evidence shows it never
    resolved. Requires at least *min_exchanges* qualifying exchanges (a
    single in-flight turn isn't a loop)."""
    from agent_trace.interceptor.httpx_hook import _is_tool_call_only_response

    tool_call_only_flags: list[bool] = []
    last_exchange: dict[str, Any] | None = None
    for exchange in exchanges:
        response_body = exchange.get("response_body")
        if not response_body:
            continue
        body = _loads(response_body)
        if not isinstance(body, dict):
            continue
        # Only consider exchanges that look like an LLM completion response
        # (has "choices" or an Anthropic-shaped "content"/"stop_reason").
        if "choices" not in body and "stop_reason" not in body:
            continue
        tool_call_only_flags.append(_is_tool_call_only_response(response_body))
        last_exchange = exchange

    if len(tool_call_only_flags) < min_exchanges or last_exchange is None:
        return []

    if all(tool_call_only_flags):
        return [
            _flag(
                "all_tool_calls_no_terminal_response",
                last_exchange,
                f"all {len(tool_call_only_flags)} captured LLM completion "
                f"exchange(s) in this run are tool-call-only responses — "
                f"no terminal (final, non-tool-call) response was ever "
                f"recorded, a candidate infinite-tool-call-loop shape",
                tool_call_only_exchange_count=len(tool_call_only_flags),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# 19. A captured response's text content is wrapped in a markdown code
#     fence (e.g. "```json\n{...}\n```") when the calling code expects to
#     json.loads() that content directly — a naive parse breaks on the
#     fence markers. Seen on Gemini/google-genai even when a pure-JSON
#     response was requested, but not provider-specific (#4509).
# ---------------------------------------------------------------------------

_MARKDOWN_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n")


def _texts_from_response_body(body: dict[str, Any]) -> list[str]:
    """Extract every assistant-message text field worth checking for a
    markdown fence, across the two response shapes this check cares
    about: OpenAI/Groq-style `choices[].message.content` and
    Gemini-style `candidates[].content.parts[].text`."""
    texts: list[str] = []

    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)

    for candidate in body.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])

    return texts


def check_markdown_fenced_json_response(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a captured response whose assistant-message text content
    starts with a markdown code fence (```` ``` ```` or ```` ```json ````)
    — a shape that breaks a naive `json.loads(content)` downstream even
    though the *HTTP response envelope itself* is valid JSON. Not
    provider-specific: seen on Gemini/google-genai even when a pure-JSON
    response was explicitly requested (#4509), but the same fencing habit
    shows up from other providers/prompted models too."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue
        for text in _texts_from_response_body(body):
            if _MARKDOWN_FENCE_RE.match(text):
                flags.append(
                    _flag(
                        "markdown_fenced_json_response",
                        exchange,
                        "response content is wrapped in a markdown code "
                        "fence (```/```json) — a naive json.loads() on "
                        "this content will fail even though the HTTP "
                        "response envelope itself is valid JSON",
                    )
                )
                break  # one flag per exchange is enough
    return flags


# ---------------------------------------------------------------------------
# Companion diagnostics (same "raw capture with zero automated diagnosis"
# gap that motivates the big `inspect` cluster above).
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


# ---------------------------------------------------------------------------
# A response calls a tool name that was never declared in that *same*
# exchange's own request `tools` list — the self-contained "the model
# hallucinated a tool name the client never offered it" shape behind #6037's
# `transfer_back_to_supervisor is not a valid tool` error inside a LangGraph
# supervisor multi-agent topology. Unlike check_tool_call_name_fuzzy_match/
# _dotted_compound (which need a caller-supplied `registered_tools` set
# spanning the whole run, opted into via `--registered-tools`), this check
# is entirely self-contained within a single exchange's request/response
# pair, so it needs no CLI flag and is wired unconditionally into
# run_all_exchange_checks().
# ---------------------------------------------------------------------------


def check_tool_call_name_absent_from_request_tools(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a captured exchange whose response's `tool_calls[]` names a
    function not present in that exchange's own request `tools[]` list
    (OpenAI-shape: `tools[].function.name`) — the model calling a tool the
    client never declared in this request at all (#6037)."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        request_body = _loads(exchange.get("request_body"))
        if not isinstance(request_body, dict):
            continue
        declared_tools: set[str] = set()
        for tool in request_body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str):
                declared_tools.add(name)
        if not declared_tools:
            # No declared tools at all to compare against — not this
            # check's concern (see check_tool_calling_disabled for that
            # shape).
            continue

        for name in _extract_tool_call_names(exchange):
            if name not in declared_tools:
                flags.append(
                    _flag(
                        "tool_call_name_absent_from_request_tools",
                        exchange,
                        f"response tool call names {name!r}, which is not "
                        "present in this exchange's own request `tools` "
                        "list — the model called a tool the client never "
                        "declared in this request",
                        called_name=name,
                        declared_tools=sorted(declared_tools),
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# Response finish/stop reason other than a normal-completion value — e.g.
# 'length' (context/max-tokens truncation), 'content_filter', Gemini's
# MALFORMED_FUNCTION_CALL, or Anthropic's 'max_tokens'/'refusal'. Today
# nothing in agent-trace surfaces this without a developer manually reading
# the raw response body — the exact manual step behind the intermittent,
# hard-to-reproduce `LengthFinishReasonError` reports (#30924).
# ---------------------------------------------------------------------------

_OK_FINISH_REASONS = {"stop", "tool_calls", "end_turn", "tool_use", "function_call"}


def check_non_ok_finish_reason(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a captured response whose finish/stop reason is not one of the
    normal-completion values (OpenAI-style `choices[].finish_reason`, or
    Anthropic's top-level `stop_reason`) — surfacing `'length'` (the
    response was truncated by max_tokens, #30924's `LengthFinishReasonError`
    shape), `'content_filter'`, Gemini's `MALFORMED_FUNCTION_CALL`, and any
    other non-`'stop'`/`'tool_calls'`/`'end_turn'` value a developer would
    otherwise only notice by manually reading the raw response body."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if not isinstance(body, dict):
            continue

        for choice in body.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if isinstance(reason, str) and reason.lower() not in _OK_FINISH_REASONS:
                flags.append(
                    _flag(
                        "non_ok_finish_reason",
                        exchange,
                        f"choices[].finish_reason={reason!r} is not a "
                        "normal-completion value ('stop'/'tool_calls'/"
                        "'end_turn') — the response was truncated/filtered "
                        "rather than completed normally",
                        finish_reason=reason,
                        choice_index=choice.get("index"),
                    )
                )

        stop_reason = body.get("stop_reason")
        if (
            isinstance(stop_reason, str)
            and stop_reason.lower() not in _OK_FINISH_REASONS
        ):
            flags.append(
                _flag(
                    "non_ok_finish_reason",
                    exchange,
                    f"stop_reason={stop_reason!r} is not a normal-completion "
                    "value ('stop'/'tool_calls'/'end_turn') — the response "
                    "was truncated/filtered rather than completed normally",
                    finish_reason=stop_reason,
                )
            )
    return flags


# ---------------------------------------------------------------------------
# A request declares a forced/single tool choice but the corresponding
# response's message carries no (or empty) tool_calls — the framework asked
# the provider to guarantee a tool call and the provider silently didn't
# deliver one (#3153: ChatOllama accepts a forced tool_choice but Ollama
# doesn't honor it, so downstream code that assumes `tool_calls[0]` exists
# crashes with `TypeError: 'NoneType' object is not subscriptable`).
# Distinct from check_tool_calling_disabled, which flags the opposite,
# explicitly-disabled ('none') shape.
# ---------------------------------------------------------------------------

_GENERIC_TOOL_CHOICE_KEYWORDS = {"auto", "none", "required", "any"}


def _forced_tool_choice_name(body: dict[str, Any]) -> str | None:
    """Return the forced tool name if *body* declares a forced/single tool
    choice, else None. Handles OpenAI's dict shape (``{"type": "function",
    "function": {"name": ...}}``), Anthropic's dict shape (``{"type":
    "tool", "name": ...}``), a plain tool-name string (``tool_choice:
    "my_tool"``, distinct from the generic 'auto'/'none'/'required'/'any'
    keywords), and Gemini's equivalent forced-choice signal on its tool
    config (`function_calling_config.mode == 'ANY'` restricted to exactly
    one entry in `allowed_function_names`)."""
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function")
            fn_name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(fn_name, str):
                return fn_name
        elif tool_choice.get("type") == "tool":
            name = tool_choice.get("name")
            if isinstance(name, str):
                return name
    elif isinstance(tool_choice, str) and tool_choice:
        if tool_choice.lower() not in _GENERIC_TOOL_CHOICE_KEYWORDS:
            return tool_choice

    found, mode = _get_path(body, "tool_config.function_calling_config.mode")
    if found and isinstance(mode, str) and mode.upper() == "ANY":
        _, names = _get_path(
            body, "tool_config.function_calling_config.allowed_function_names"
        )
        if isinstance(names, list) and len(names) == 1 and isinstance(names[0], str):
            return names[0]
    return None


def check_forced_tool_call_unfulfilled(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a captured exchange where the request declared a forced/single
    tool choice but the response's message has no (or empty) `tool_calls` —
    "expected forced tool call, got none" (#3153)."""
    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        request_body = _loads(exchange.get("request_body"))
        if not isinstance(request_body, dict):
            continue
        forced_name = _forced_tool_choice_name(request_body)
        if not forced_name:
            continue

        response_body = _loads(exchange.get("response_body"))
        if not isinstance(response_body, dict):
            continue
        choices = response_body.get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        got_tool_call = any(
            isinstance(choice, dict)
            and isinstance(choice.get("message"), dict)
            and choice["message"].get("tool_calls")
            for choice in choices
        )
        if not got_tool_call:
            flags.append(
                _flag(
                    "forced_tool_call_unfulfilled",
                    exchange,
                    f"request declared a forced/single tool_choice "
                    f"({forced_name!r}) but the response's message has no "
                    "(or empty) tool_calls — expected forced tool call, "
                    "got none",
                    forced_tool_name=forced_name,
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


def check_null_or_missing_sse_delta(
    exchanges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a captured SSE exchange where a streamed event's
    `choices[0].delta` is null/missing while the event still carries other
    populated fields — e.g. Azure OpenAI's async content-filter shape,
    where a chunk arrives with `delta: null` alongside a populated
    `content_filter_offsets`/`content_filter_results` on the same choice.

    `_delta_from_event()` (`interceptor/sse.py`) already treats a null/
    missing delta as "no delta" and silently skips it during stream
    reconstruction — correct behavior for merging, but it means a caller
    consuming only the reconstructed message never learns a chunk carrying
    real signal (a content-filter hit, a finish_reason) was dropped. This
    check is the automated flag for that dropped-chunk shape (#797)."""
    from agent_trace.interceptor.sse import is_sse_exchange, parse_sse_events

    flags: list[dict[str, Any]] = []
    for exchange in exchanges:
        if not is_sse_exchange(exchange):
            continue
        events = parse_sse_events(exchange.get("response_body") or "")
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                continue  # a real (possibly empty) delta — nothing dropped.
            other_populated = {
                key: value
                for key, value in choice.items()
                if key not in ("delta", "index") and value not in (None, "", [], {})
            }
            if other_populated:
                flags.append(
                    _flag(
                        "null_or_missing_sse_delta",
                        exchange,
                        "SSE event has a null/missing `choices[0].delta` while "
                        f"carrying other populated fields ({sorted(other_populated)}) "
                        "— the chunk's signal is silently dropped by naive stream "
                        "merging instead of surfaced",
                        event_index=event_index,
                        populated_fields=sorted(other_populated),
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
    returning usage data that something downstream strips").

    *field_name* may be a plain top-level key (``"usage"``) or a dotted/
    nested path resolved via ``_get_path`` (``"choices.0.message.
    reasoning_content"``) — needed for provider fields nested inside the
    response body rather than sitting at the top level, e.g. DeepSeek's
    ``choices[0].message.reasoning_content`` (#5526), which a plain
    ``field_name in body`` top-level-key check can never see."""
    wire_has_field = False
    for exchange in exchanges:
        body = _loads(exchange.get("response_body"))
        if isinstance(body, dict) and _get_path(body, field_name)[0]:
            wire_has_field = True
            break
    if not wire_has_field:
        return []

    # Span attributes are flat (e.g. "llm.content", "llm.finish_reason"),
    # never namespaced by a full nested path, so match on the path's last
    # segment for the downstream-presence check.
    leaf_field = field_name.rsplit(".", 1)[-1]
    span_has_field = False
    for span in spans:
        attrs = span.get("attributes") or {}
        if any(str(k).startswith(f"llm.{leaf_field}") for k in attrs):
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


def check_system_prompt_dropped(
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag a pydantic-ai `system_prompt`-vs-`message_history` regression
    within one trace (#3277): `agent.run(..., message_history=...)` silently
    drops the agent's configured `SystemPromptPart` on the follow-up call.

    `integrations/pydantic_ai.py`'s `_open_llm_span` already persists
    `llm.has_system_prompt_part` (bool) per LLM span. This walks a trace's
    `llm:*` spans in call order (by `start_time`), grouped by
    `(agent.name, llm.model)`, and flags any transition where an earlier
    call had a system prompt part present and a later call for the same
    agent/model does not — surfacing the drop automatically instead of
    requiring a developer to already suspect it and hand-diff two
    fixtures."""
    llm_spans = [
        s
        for s in spans
        if str(s.get("name", "")).startswith("llm:")
        and "llm.has_system_prompt_part" in (s.get("attributes") or {})
    ]
    llm_spans.sort(key=lambda s: s.get("start_time") or 0)

    by_group: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for span in llm_spans:
        attrs = span.get("attributes") or {}
        key = (attrs.get("agent.name"), attrs.get("llm.model"))
        by_group.setdefault(key, []).append(span)

    flags: list[dict[str, Any]] = []
    for (agent_name, model), group_spans in by_group.items():
        for earlier, later in itertools.pairwise(group_spans):
            earlier_attrs = earlier.get("attributes") or {}
            later_attrs = later.get("attributes") or {}
            if earlier_attrs.get("llm.has_system_prompt_part") and not later_attrs.get(
                "llm.has_system_prompt_part"
            ):
                flags.append(
                    {
                        "check": "system_prompt_dropped",
                        "agent_name": agent_name,
                        "model": model,
                        "earlier_span": earlier.get("span_id"),
                        "later_span": later.get("span_id"),
                        "detail": f"agent {agent_name!r} (model {model!r}) sent a "
                        "system prompt on an earlier call but a later call in the "
                        "same trace has no SystemPromptPart — likely dropped when "
                        "message_history was passed",
                    }
                )
    return flags


def check_phantom_tool_call(
    spans: list[dict[str, Any]], exchanges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Flag a claimed tool invocation with zero corresponding downstream HTTP
    exchanges in its time window (#13449 — a ReActAgent transcript claiming a
    tool ran when nothing actually executed).

    Now that a tool-call span/event can be tied back to the HTTP exchanges
    it produced (`push_correlation_id(span.span_id)` at the point a tool
    span opens, recoverable via `Fixture.exchanges_for_correlation_id`; see
    `integrations/langgraph.py` and `integrations/llama_index.py`), this
    walks every span, identifies the ones representing a tool call — a
    LangGraph `tool:<name>` span, or a llama_index span carrying a
    `tool_call` event — and flags any whose `span_id` has *no* correlated
    HTTP exchange at all as a likely phantom/silently-skipped invocation:
    the exact thing a developer previously had to work out by hand from
    console logs.

    Deliberately over-inclusive rather than silent: a genuinely pure-Python
    tool (no network call, e.g. a local calculator) will also have zero
    correlated exchanges and gets flagged too — callers who know a given
    tool never makes HTTP calls should filter this check's output by
    `tool_name` rather than treat every flag as proof of a bug."""
    correlated_ids: set[str] = {
        str(exchange["correlation_id"])
        for exchange in exchanges
        if exchange.get("correlation_id")
    }

    flags: list[dict[str, Any]] = []
    for span in spans:
        span_id = span.get("span_id")
        if not span_id:
            continue
        attrs = span.get("attributes") or {}
        events = span.get("events") or []
        tool_call_event = next(
            (e for e in events if isinstance(e, dict) and e.get("name") == "tool_call"),
            None,
        )
        is_tool_span = str(span.get("name", "")).startswith("tool:")
        if not is_tool_span and tool_call_event is None and "tool.name" not in attrs:
            continue

        tool_name = attrs.get("tool.name")
        if tool_name is None and tool_call_event is not None:
            tool_name = (tool_call_event.get("attributes") or {}).get("tool.name")
        tool_name = tool_name or span.get("name")

        if str(span_id) not in correlated_ids:
            flags.append(
                {
                    "check": "phantom_tool_call",
                    "span": span.get("name"),
                    "span_id": span_id,
                    "tool_name": tool_name,
                    "detail": f"tool call {tool_name!r} (span {span.get('name')!r}) "
                    "has zero correlated HTTP exchanges — the transcript claims "
                    "this tool ran but no downstream network call was recorded "
                    "for it",
                }
            )
    return flags


# ---------------------------------------------------------------------------
# `agent-trace diff`-style restart-vs-resume detection: a later run's ROOT
# chain span (`parent_id` is None) carries the same LangGraph `thread_id` —
# recovered from its captured `chain.metadata`, see LangGraphTracer.
# on_chain_start — as an earlier run, but its `langgraph_step` does not
# continue from the earlier run's last recorded step for that thread_id.
# That is exactly the "new root span with no parent, same thread_id as a
# prior interrupted run" shape #161 asked to auto-flag: the later run
# started the graph over from scratch (a *restart*) rather than resuming it
# from its last checkpoint (a *resume*). cmd_diff (which already loads both
# runs' spans to compare exchanges) wires this in unconditionally.
# ---------------------------------------------------------------------------


def _chain_metadata(span: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort parse of a span's captured `chain.metadata` attribute
    (set by LangGraphTracer.on_chain_start, JSON-serialized) back into a
    dict, or None if the span has no (or unparsable) chain metadata."""
    attrs = span.get("attributes")
    if not isinstance(attrs, dict):
        return None
    raw = attrs.get("chain.metadata")
    if not isinstance(raw, str):
        return None
    parsed = _loads(raw)
    return parsed if isinstance(parsed, dict) else None


def _last_langgraph_step_by_thread(spans: list[dict[str, Any]]) -> dict[str, int]:
    """The highest `langgraph_step` recorded for each `thread_id` across
    *every* chain span in a run — not just root spans — since a resumed run
    should pick up from wherever the earlier run's graph execution actually
    last got to, not just wherever its root span happened to be."""
    last_step: dict[str, int] = {}
    for span in spans:
        metadata = _chain_metadata(span)
        if metadata is None:
            continue
        thread_id = metadata.get("thread_id")
        step = metadata.get("langgraph_step")
        if not isinstance(thread_id, str) or not isinstance(step, int):
            continue
        if step > last_step.get(thread_id, -1):
            last_step[thread_id] = step
    return last_step


def check_restart_vs_resume(
    spans_a: list[dict[str, Any]], spans_b: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Flag a candidate LangGraph restart-vs-resume mismatch (#161): run
    *spans_b* (assumed to be the later/second of the two runs) has a root
    chain span (`parent_id` is None, with `chain.metadata` present) whose
    `thread_id` also appears in *spans_a* (the earlier/first run), but whose
    `langgraph_step` does not continue from *spans_a*'s last recorded step
    for that same `thread_id` — i.e. the later run's graph execution
    appears to have restarted from scratch instead of resuming the
    interrupted run from its last checkpoint."""
    last_steps_a = _last_langgraph_step_by_thread(spans_a)
    if not last_steps_a:
        return []

    flags: list[dict[str, Any]] = []
    for span in spans_b:
        if span.get("parent_id") is not None:
            continue
        metadata = _chain_metadata(span)
        if metadata is None:
            continue
        thread_id = metadata.get("thread_id")
        step = metadata.get("langgraph_step")
        if not isinstance(thread_id, str) or not isinstance(step, int):
            continue
        prior_last_step = last_steps_a.get(thread_id)
        if prior_last_step is None:
            continue
        if step <= prior_last_step:
            flags.append(
                {
                    "check": "restart_vs_resume",
                    "span": span.get("name"),
                    "thread_id": thread_id,
                    "prior_last_step": prior_last_step,
                    "root_langgraph_step": step,
                    "detail": (
                        f"root span {span.get('name')!r} in the later run "
                        f"shares thread_id={thread_id!r} with an earlier "
                        f"run, but its langgraph_step={step} does not "
                        f"continue from that earlier run's last recorded "
                        f"step ({prior_last_step}) for the same thread — "
                        "candidate restart-vs-resume: the graph appears to "
                        "have restarted from scratch rather than resumed "
                        "from its last checkpoint"
                    ),
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
        "orphaned_responses_api_call_ids": check_orphaned_responses_api_call_ids,
        "tool_call_boundary_leak": check_tool_call_boundary_leak,
        "malformed_tool_call_arguments": check_malformed_tool_call_arguments,
        "null_content_with_tool_calls": check_null_content_with_tool_calls,
        "content_block_missing_type": check_content_block_missing_type,
        "tools_with_response_format": check_tools_with_response_format,
        "tool_call_name_absent_from_request_tools": (
            check_tool_call_name_absent_from_request_tools
        ),
        "anthropic_thinking_in_tool_result": check_anthropic_thinking_in_tool_result,
        "empty_content_not_final": check_empty_content_not_final,
        "json_schema_lookaround_or_anyof": check_json_schema_lookaround_or_anyof,
        "duplicate_json_blocks": check_duplicate_json_blocks,
        "duplicate_concurrent_tool_calls": check_duplicate_concurrent_tool_calls,
        "missing_tool_call_id": check_missing_tool_call_id,
        "http_error_status": flag_4xx_5xx_exchanges,
        "all_tool_calls_no_terminal_response": check_all_tool_calls_no_terminal_response,
        "markdown_fenced_json_response": check_markdown_fenced_json_response,
        "tool_calling_disabled": check_tool_calling_disabled,
        "forced_tool_call_unfulfilled": check_forced_tool_call_unfulfilled,
        "non_ok_finish_reason": check_non_ok_finish_reason,
        "multi_block_response": multi_block_llm_responses,
        "stream_merge_invalid_json": check_stream_merge_validity,
        "null_or_missing_sse_delta": check_null_or_missing_sse_delta,
        "response_shape_anomaly": detect_response_shape_anomalies,
    }
    results: dict[str, list[dict[str, Any]]] = {}
    for name, fn in checks.items():
        flags = fn(exchanges)
        if flags:
            results[name] = flags
    return results
