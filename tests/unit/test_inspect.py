"""
Unit tests for agent_trace._inspect — the pattern-check library backing
`agent-trace inspect` and related CLI diagnostics.

Every check function here is a pure function over plain dicts shaped like
Fixture.all_exchanges()/trace.json's "spans" list, so no live HTTP/fixture/
LangGraph dependency is needed anywhere in this file.
"""

from __future__ import annotations

import json

from agent_trace import _inspect as ins


def _exchange(
    url: str = "https://api.openai.com/v1/chat/completions",
    method: str = "POST",
    sequence_num: int = 0,
    request_body: str = "{}",
    response_body: str = "{}",
    response_status: int | None = 200,
) -> dict[str, object]:
    return {
        "url": url,
        "method": method,
        "sequence_num": sequence_num,
        "request_body": request_body,
        "response_body": response_body,
        "response_status": response_status,
    }


def _span(
    name: str,
    status: str = "OK",
    attributes: dict[str, object] | None = None,
    events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "attributes": attributes or {},
        "events": events or [],
    }


def _exception_event(message: str, exc_type: str = "ValueError") -> dict[str, object]:
    return {
        "name": "exception",
        "attributes": {"exception.type": exc_type, "exception.message": message},
    }


# ---------------------------------------------------------------------------
# check_orphaned_tool_call_ids
# ---------------------------------------------------------------------------


class TestCheckOrphanedToolCallIds:
    def test_no_messages_no_flag(self) -> None:
        assert ins.check_orphaned_tool_call_ids([_exchange(request_body="{}")]) == []

    def test_responded_id_not_flagged(self) -> None:
        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "c1", "function": {}}]},
                    {"role": "tool", "tool_call_id": "c1"},
                ]
            }
        )
        assert ins.check_orphaned_tool_call_ids([_exchange(request_body=body)]) == []

    def test_orphaned_id_flagged(self) -> None:
        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "c1", "function": {}}]},
                ]
            }
        )
        flags = ins.check_orphaned_tool_call_ids([_exchange(request_body=body)])
        assert len(flags) == 1
        assert flags[0]["orphaned_ids"] == ["c1"]

    def test_malformed_body_not_raised(self) -> None:
        assert ins.check_orphaned_tool_call_ids([_exchange(request_body="not json")]) == []


# ---------------------------------------------------------------------------
# check_orphaned_responses_api_call_ids (#33895)
# ---------------------------------------------------------------------------


class TestCheckOrphanedResponsesApiCallIds:
    def test_no_input_no_flag(self) -> None:
        assert ins.check_orphaned_responses_api_call_ids([_exchange(request_body="{}")]) == []

    def test_paired_call_id_not_flagged(self) -> None:
        body = json.dumps(
            {
                "input": [
                    {"type": "function_call", "call_id": "call_1", "name": "get_weather"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
                ]
            }
        )
        assert ins.check_orphaned_responses_api_call_ids([_exchange(request_body=body)]) == []

    def test_orphaned_call_id_flagged(self) -> None:
        """The exact #33895 shape: a function_call with no matching
        function_call_output — "No call message found for call_*"."""
        body = json.dumps(
            {
                "input": [
                    {"type": "function_call", "call_id": "call_1", "name": "get_weather"},
                ]
            }
        )
        flags = ins.check_orphaned_responses_api_call_ids([_exchange(request_body=body)])
        assert len(flags) == 1
        assert flags[0]["orphaned_ids"] == ["call_1"]
        assert flags[0]["check"] == "orphaned_responses_api_call_ids"

    def test_chat_completions_shape_not_flagged(self) -> None:
        """A plain Chat Completions body has no top-level `input` list at
        all — must not be misinterpreted as an empty Responses API body."""
        body = json.dumps(
            {"messages": [{"role": "assistant", "tool_calls": [{"id": "c1"}]}]}
        )
        assert ins.check_orphaned_responses_api_call_ids([_exchange(request_body=body)]) == []

    def test_non_function_call_items_ignored(self) -> None:
        body = json.dumps(
            {
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {"type": "reasoning", "id": "r1"},
                ]
            }
        )
        assert ins.check_orphaned_responses_api_call_ids([_exchange(request_body=body)]) == []

    def test_malformed_body_not_raised(self) -> None:
        assert (
            ins.check_orphaned_responses_api_call_ids([_exchange(request_body="not json")])
            == []
        )


# ---------------------------------------------------------------------------
# check_tool_call_boundary_leak
# ---------------------------------------------------------------------------


class TestCheckToolCallBoundaryLeak:
    def test_no_marker_not_flagged(self) -> None:
        assert ins.check_tool_call_boundary_leak([_exchange(response_body="hello")]) == []

    def test_marker_flagged(self) -> None:
        flags = ins.check_tool_call_boundary_leak(
            [_exchange(response_body="blah to=functions.search blah")]
        )
        assert len(flags) == 1
        assert flags[0]["marker"] == "to=functions."


# ---------------------------------------------------------------------------
# check_malformed_tool_call_arguments
# ---------------------------------------------------------------------------


class TestCheckMalformedToolCallArguments:
    def test_valid_json_arguments_not_flagged(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "f", "arguments": '{"x": 1}'}}
                            ]
                        }
                    }
                ]
            }
        )
        assert ins.check_malformed_tool_call_arguments([_exchange(response_body=body)]) == []

    def test_invalid_json_arguments_flagged(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "f", "arguments": "{not json"}}
                            ]
                        }
                    }
                ]
            }
        )
        flags = ins.check_malformed_tool_call_arguments([_exchange(response_body=body)])
        assert len(flags) == 1
        assert flags[0]["tool_name"] == "f"


# ---------------------------------------------------------------------------
# check_null_content_with_tool_calls
# ---------------------------------------------------------------------------


class TestCheckNullContentWithToolCalls:
    def test_string_content_not_flagged(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": "hi", "tool_calls": [{}]}}]}
        )
        assert ins.check_null_content_with_tool_calls([_exchange(response_body=body)]) == []

    def test_none_content_not_flagged(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": None, "tool_calls": [{}]}}]})
        assert ins.check_null_content_with_tool_calls([_exchange(response_body=body)]) == []

    def test_list_content_flagged(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": [], "tool_calls": [{}]}}]})
        flags = ins.check_null_content_with_tool_calls([_exchange(response_body=body)])
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# check_endpoint_host_mismatch
# ---------------------------------------------------------------------------


class TestCheckEndpointHostMismatch:
    def test_matching_host_not_flagged(self) -> None:
        exchanges = [_exchange(url="https://api.openai.com/v1/chat")]
        assert ins.check_endpoint_host_mismatch(exchanges, "api.openai.com") == []

    def test_mismatched_host_flagged(self) -> None:
        exchanges = [_exchange(url="https://evil.example.com/v1/chat")]
        flags = ins.check_endpoint_host_mismatch(exchanges, "api.openai.com")
        assert len(flags) == 1
        assert flags[0]["actual_host"] == "evil.example.com"


# ---------------------------------------------------------------------------
# check_tools_with_response_format
# ---------------------------------------------------------------------------


class TestCheckToolsWithResponseFormat:
    def test_tools_only_not_flagged(self) -> None:
        body = json.dumps({"tools": [{}]})
        assert ins.check_tools_with_response_format([_exchange(request_body=body)]) == []

    def test_tools_and_response_format_flagged(self) -> None:
        body = json.dumps({"tools": [{}], "response_format": {"type": "json_schema"}})
        flags = ins.check_tools_with_response_format([_exchange(request_body=body)])
        assert len(flags) == 1

    def test_tools_and_response_model_flagged(self) -> None:
        body = json.dumps({"tools": [{}], "response_model": "Foo"})
        flags = ins.check_tools_with_response_format([_exchange(request_body=body)])
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# check_anthropic_thinking_in_tool_result
# ---------------------------------------------------------------------------


class TestCheckAnthropicThinkingInToolResult:
    def test_normal_tool_result_not_flagged(self) -> None:
        body = json.dumps(
            {
                "messages": [
                    {
                        "content": [
                            {"type": "tool_result", "content": [{"type": "text", "text": "ok"}]}
                        ]
                    }
                ]
            }
        )
        assert ins.check_anthropic_thinking_in_tool_result([_exchange(request_body=body)]) == []

    def test_thinking_block_in_tool_result_flagged(self) -> None:
        body = json.dumps(
            {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": [{"type": "thinking", "thinking": "..."}],
                            }
                        ]
                    }
                ]
            }
        )
        flags = ins.check_anthropic_thinking_in_tool_result([_exchange(request_body=body)])
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# check_empty_content_not_final
# ---------------------------------------------------------------------------


class TestCheckEmptyContentNotFinal:
    def test_empty_content_as_final_message_not_flagged(self) -> None:
        body = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": ""},
                ]
            }
        )
        assert ins.check_empty_content_not_final([_exchange(request_body=body)]) == []

    def test_empty_content_mid_array_flagged(self) -> None:
        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": "hi"},
                ]
            }
        )
        flags = ins.check_empty_content_not_final([_exchange(request_body=body)])
        assert len(flags) == 1
        assert flags[0]["message_index"] == 0

    def test_empty_list_content_flagged(self) -> None:
        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": []},
                    {"role": "user", "content": "hi"},
                ]
            }
        )
        flags = ins.check_empty_content_not_final([_exchange(request_body=body)])
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# check_action_name_not_registered
# ---------------------------------------------------------------------------


class TestCheckActionNameNotRegistered:
    def test_registered_action_not_flagged(self) -> None:
        exchanges = [_exchange(response_body="Thought: x\nAction: search\nAction Input: y")]
        assert ins.check_action_name_not_registered(exchanges, {"search"}) == []

    def test_unregistered_action_flagged(self) -> None:
        exchanges = [_exchange(response_body="Action: unknown_tool")]
        flags = ins.check_action_name_not_registered(exchanges, {"search"})
        assert len(flags) == 1
        assert flags[0]["action_name"] == "unknown_tool"

    def test_formatting_noise_stripped_before_comparison(self) -> None:
        exchanges = [_exchange(response_body="Action: `search`")]
        assert ins.check_action_name_not_registered(exchanges, {"search"}) == []


# ---------------------------------------------------------------------------
# check_json_schema_lookaround_or_anyof
# ---------------------------------------------------------------------------


class TestCheckJsonSchemaLookaroundOrAnyof:
    def test_plain_pattern_not_flagged(self) -> None:
        body = json.dumps({"schema": {"pattern": "^[a-z]+$"}})
        assert ins.check_json_schema_lookaround_or_anyof([_exchange(request_body=body)]) == []

    def test_lookaround_pattern_flagged(self) -> None:
        body = json.dumps({"schema": {"pattern": "(?=foo)bar"}})
        flags = ins.check_json_schema_lookaround_or_anyof([_exchange(request_body=body)])
        assert any(f["check"] == "json_schema_lookaround" for f in flags)

    def test_anyof_type_mismatch_under_strict_flagged(self) -> None:
        body = json.dumps(
            {
                "strict": True,
                "schema": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            }
        )
        flags = ins.check_json_schema_lookaround_or_anyof([_exchange(request_body=body)])
        assert any(f["check"] == "json_schema_anyof_type_mismatch" for f in flags)


# ---------------------------------------------------------------------------
# check_duplicate_json_blocks
# ---------------------------------------------------------------------------


class TestCheckDuplicateJsonBlocks:
    def test_unique_blocks_not_flagged(self) -> None:
        body = json.dumps({"input": [{"a": 1}, {"a": 2}, {"a": 3}]})
        assert ins.check_duplicate_json_blocks([_exchange(request_body=body)]) == []

    def test_repeated_blocks_flagged(self) -> None:
        body = json.dumps({"input": [{"a": 1}, {"a": 1}, {"a": 1}]})
        flags = ins.check_duplicate_json_blocks([_exchange(request_body=body)])
        assert len(flags) == 1
        assert flags[0]["total_repeats"] == 3


# ---------------------------------------------------------------------------
# check_missing_extra_kwarg
# ---------------------------------------------------------------------------


class TestCheckMissingExtraKwarg:
    def test_present_kwarg_not_flagged(self) -> None:
        body = json.dumps({"extra_body": {"chat_template_kwargs": {"thinking": True}}})
        exchanges = [_exchange(request_body=body)]
        assert (
            ins.check_missing_extra_kwarg(
                exchanges, "extra_body.chat_template_kwargs.thinking"
            )
            == []
        )

    def test_absent_kwarg_flagged(self) -> None:
        exchanges = [_exchange(request_body="{}")]
        flags = ins.check_missing_extra_kwarg(
            exchanges, "extra_body.chat_template_kwargs.thinking"
        )
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# check_tool_call_name_fuzzy_match / check_tool_call_name_dotted_compound
# ---------------------------------------------------------------------------


def _response_with_tool_call(name: str) -> str:
    return json.dumps(
        {"choices": [{"message": {"tool_calls": [{"function": {"name": name}}]}}]}
    )


class TestCheckToolCallNameFuzzyMatch:
    def test_registered_name_not_flagged(self) -> None:
        exchanges = [_exchange(response_body=_response_with_tool_call("occrra_information"))]
        assert ins.check_tool_call_name_fuzzy_match(exchanges, {"occrra_information"}) == []

    def test_near_miss_spelling_flagged(self) -> None:
        exchanges = [_exchange(response_body=_response_with_tool_call("occcra_information"))]
        flags = ins.check_tool_call_name_fuzzy_match(exchanges, {"occrra_information"})
        assert len(flags) == 1
        assert flags[0]["nearest_registered_name"] == "occrra_information"

    def test_completely_unrelated_name_not_flagged(self) -> None:
        exchanges = [_exchange(response_body=_response_with_tool_call("totally_different_xyz"))]
        assert ins.check_tool_call_name_fuzzy_match(exchanges, {"search"}, max_distance=3) == []


class TestCheckToolCallNameDottedCompound:
    def test_single_registered_name_not_flagged(self) -> None:
        exchanges = [_exchange(response_body=_response_with_tool_call("Tool_A"))]
        assert (
            ins.check_tool_call_name_dotted_compound(exchanges, {"Tool_A", "Tool_B"}) == []
        )

    def test_dotted_compound_of_two_registered_names_flagged(self) -> None:
        exchanges = [_exchange(response_body=_response_with_tool_call("Tool_A.Tool_B"))]
        flags = ins.check_tool_call_name_dotted_compound(exchanges, {"Tool_A", "Tool_B"})
        assert len(flags) == 1
        assert flags[0]["compound_parts"] == ["Tool_A", "Tool_B"]

    def test_dotted_name_with_unregistered_part_not_flagged(self) -> None:
        exchanges = [_exchange(response_body=_response_with_tool_call("Tool_A.Unknown"))]
        assert (
            ins.check_tool_call_name_dotted_compound(exchanges, {"Tool_A", "Tool_B"}) == []
        )


# ---------------------------------------------------------------------------
# check_missing_tool_call_id
# ---------------------------------------------------------------------------


class TestCheckMissingToolCallId:
    def test_id_present_not_flagged(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"tool_calls": [{"id": "c1", "function": {}}]}}]}
        )
        assert ins.check_missing_tool_call_id([_exchange(response_body=body)]) == []

    def test_id_absent_flagged(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"tool_calls": [{"function": {"name": "f"}}]}}]}
        )
        flags = ins.check_missing_tool_call_id([_exchange(response_body=body)])
        assert len(flags) == 1

    def test_id_null_flagged(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"tool_calls": [{"id": None, "function": {}}]}}]}
        )
        flags = ins.check_missing_tool_call_id([_exchange(response_body=body)])
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# check_get_post_field_mismatch
# ---------------------------------------------------------------------------


class TestCheckGetPostFieldMismatch:
    def test_matching_values_not_flagged(self) -> None:
        get_body = json.dumps({"id": "asst_1", "instructions": "be nice"})
        post_body = json.dumps({"assistant_id": "asst_1", "instructions": "be nice"})
        exchanges = [
            _exchange(method="GET", sequence_num=0, response_body=get_body),
            _exchange(method="POST", sequence_num=1, request_body=post_body),
        ]
        flags = ins.check_get_post_field_mismatch(
            exchanges, "instructions", post_id_field="assistant_id"
        )
        assert flags == []

    def test_stale_value_flagged(self) -> None:
        get_body = json.dumps({"id": "asst_1", "instructions": "be nice"})
        post_body = json.dumps({"assistant_id": "asst_1", "instructions": "be rude"})
        exchanges = [
            _exchange(method="GET", sequence_num=0, response_body=get_body),
            _exchange(method="POST", sequence_num=1, request_body=post_body),
        ]
        flags = ins.check_get_post_field_mismatch(
            exchanges, "instructions", post_id_field="assistant_id"
        )
        assert len(flags) == 1
        assert flags[0]["get_value"] == "be nice"
        assert flags[0]["post_value"] == "be rude"

    def test_unrelated_resource_id_not_compared(self) -> None:
        get_body = json.dumps({"id": "asst_1", "instructions": "be nice"})
        post_body = json.dumps({"assistant_id": "asst_2", "instructions": "different"})
        exchanges = [
            _exchange(method="GET", sequence_num=0, response_body=get_body),
            _exchange(method="POST", sequence_num=1, request_body=post_body),
        ]
        assert ins.check_get_post_field_mismatch(exchanges, "instructions") == []


# ---------------------------------------------------------------------------
# check_duplicate_concurrent_tool_calls (#6882)
# ---------------------------------------------------------------------------


class TestCheckDuplicateConcurrentToolCalls:
    def test_single_tool_call_not_flagged(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "run_team"}},
                            ]
                        }
                    }
                ]
            }
        )
        assert ins.check_duplicate_concurrent_tool_calls([_exchange(response_body=body)]) == []

    def test_two_distinct_tools_not_flagged(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "tool_a"}},
                                {"id": "c2", "function": {"name": "tool_b"}},
                            ]
                        }
                    }
                ]
            }
        )
        assert ins.check_duplicate_concurrent_tool_calls([_exchange(response_body=body)]) == []

    def test_same_tool_called_twice_flagged(self) -> None:
        """The exact #6882 shape: parallel_tool_calls=True calling the same
        (non-reentrant) team/tool twice in one assistant turn."""
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "run_team"}},
                                {"id": "c2", "function": {"name": "run_team"}},
                            ]
                        }
                    }
                ]
            }
        )
        flags = ins.check_duplicate_concurrent_tool_calls([_exchange(response_body=body)])
        assert len(flags) == 1
        assert flags[0]["duplicated_tool_counts"] == {"run_team": 2}

    def test_three_calls_two_duplicated_one_distinct(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "run_team"}},
                                {"id": "c2", "function": {"name": "run_team"}},
                                {"id": "c3", "function": {"name": "other_tool"}},
                            ]
                        }
                    }
                ]
            }
        )
        flags = ins.check_duplicate_concurrent_tool_calls([_exchange(response_body=body)])
        assert len(flags) == 1
        assert flags[0]["duplicated_tool_counts"] == {"run_team": 2}

    def test_no_tool_calls_not_flagged(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": "hi"}}]})
        assert ins.check_duplicate_concurrent_tool_calls([_exchange(response_body=body)]) == []

    def test_malformed_body_not_raised(self) -> None:
        assert (
            ins.check_duplicate_concurrent_tool_calls([_exchange(response_body="not json")])
            == []
        )


# ---------------------------------------------------------------------------
# check_all_tool_calls_no_terminal_response (#3097)
# ---------------------------------------------------------------------------


def _tool_call_only_body() -> str:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "c1", "function": {"name": "search"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )


def _terminal_body(text: str = "done") -> str:
    return json.dumps(
        {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    )


class TestCheckAllToolCallsNoTerminalResponse:
    def test_below_min_exchanges_not_flagged(self) -> None:
        exchanges = [_exchange(response_body=_tool_call_only_body())]
        assert ins.check_all_tool_calls_no_terminal_response(exchanges) == []

    def test_all_tool_call_only_flagged(self) -> None:
        """The exact #3097 shape: an infinite tool-call loop that never
        resolved, no terminal response ever recorded."""
        exchanges = [
            _exchange(response_body=_tool_call_only_body()),
            _exchange(response_body=_tool_call_only_body()),
            _exchange(response_body=_tool_call_only_body()),
        ]
        flags = ins.check_all_tool_calls_no_terminal_response(exchanges)
        assert len(flags) == 1
        assert flags[0]["tool_call_only_exchange_count"] == 3

    def test_terminal_response_present_not_flagged(self) -> None:
        exchanges = [
            _exchange(response_body=_tool_call_only_body()),
            _exchange(response_body=_tool_call_only_body()),
            _exchange(response_body=_terminal_body()),
        ]
        assert ins.check_all_tool_calls_no_terminal_response(exchanges) == []

    def test_non_llm_exchanges_ignored(self) -> None:
        exchanges = [
            _exchange(response_body=json.dumps({"result": "ok"})),
            _exchange(response_body=json.dumps({"result": "ok"})),
        ]
        assert ins.check_all_tool_calls_no_terminal_response(exchanges) == []

    def test_custom_min_exchanges_threshold(self) -> None:
        exchanges = [
            _exchange(response_body=_tool_call_only_body()),
            _exchange(response_body=_tool_call_only_body()),
        ]
        assert (
            ins.check_all_tool_calls_no_terminal_response(exchanges, min_exchanges=5)
            == []
        )

    def test_malformed_body_ignored_not_raised(self) -> None:
        exchanges = [
            _exchange(response_body="not json"),
            _exchange(response_body=_tool_call_only_body()),
        ]
        # Only one qualifying exchange after the malformed one is skipped —
        # below the default min_exchanges=2 threshold.
        assert ins.check_all_tool_calls_no_terminal_response(exchanges) == []


# ---------------------------------------------------------------------------
# check_markdown_fenced_json_response (#4509)
# ---------------------------------------------------------------------------


class TestCheckMarkdownFencedJsonResponse:
    def test_plain_content_not_flagged(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": '{"a": 1}'}}]})
        assert ins.check_markdown_fenced_json_response([_exchange(response_body=body)]) == []

    def test_openai_shape_json_fence_flagged(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": '```json\n{"a": 1}\n```'}}]}
        )
        flags = ins.check_markdown_fenced_json_response([_exchange(response_body=body)])
        assert len(flags) == 1
        assert flags[0]["check"] == "markdown_fenced_json_response"

    def test_openai_shape_bare_fence_flagged(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": '```\n{"a": 1}\n```'}}]})
        flags = ins.check_markdown_fenced_json_response([_exchange(response_body=body)])
        assert len(flags) == 1

    def test_gemini_shape_json_fence_flagged(self) -> None:
        """The exact #4509 shape: Gemini wraps structured JSON output in a
        markdown code fence even when a pure-JSON response was requested."""
        body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '```json\n{"result": "ok"}\n```'}]
                        }
                    }
                ]
            }
        )
        flags = ins.check_markdown_fenced_json_response([_exchange(response_body=body)])
        assert len(flags) == 1

    def test_gemini_shape_plain_text_not_flagged(self) -> None:
        body = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": '{"result": "ok"}'}]}}]}
        )
        assert ins.check_markdown_fenced_json_response([_exchange(response_body=body)]) == []

    def test_no_content_not_flagged(self) -> None:
        body = json.dumps({"choices": [{"message": {}}]})
        assert ins.check_markdown_fenced_json_response([_exchange(response_body=body)]) == []

    def test_malformed_body_not_raised(self) -> None:
        assert (
            ins.check_markdown_fenced_json_response([_exchange(response_body="not json")])
            == []
        )


# ---------------------------------------------------------------------------
# flag_4xx_5xx_exchanges
# ---------------------------------------------------------------------------


class TestFlag4xx5xxExchanges:
    def test_200_not_flagged(self) -> None:
        assert ins.flag_4xx_5xx_exchanges([_exchange(response_status=200)]) == []

    def test_400_flagged(self) -> None:
        flags = ins.flag_4xx_5xx_exchanges([_exchange(response_status=400)])
        assert len(flags) == 1
        assert flags[0]["status"] == 400

    def test_500_flagged(self) -> None:
        flags = ins.flag_4xx_5xx_exchanges([_exchange(response_status=500)])
        assert len(flags) == 1

    def test_none_status_not_flagged(self) -> None:
        assert ins.flag_4xx_5xx_exchanges([_exchange(response_status=None)]) == []


# ---------------------------------------------------------------------------
# check_tool_calling_disabled
# ---------------------------------------------------------------------------


class TestCheckToolCallingDisabled:
    def test_tools_without_disabling_not_flagged(self) -> None:
        body = json.dumps({"tools": [{}], "tool_choice": "auto"})
        assert ins.check_tool_calling_disabled([_exchange(request_body=body)]) == []

    def test_gemini_mode_none_flagged(self) -> None:
        body = json.dumps(
            {"tools": [{}], "tool_config": {"function_calling_config": {"mode": "NONE"}}}
        )
        flags = ins.check_tool_calling_disabled([_exchange(request_body=body)])
        assert any(f["provider"] == "gemini" for f in flags)

    def test_openai_tool_choice_none_flagged(self) -> None:
        body = json.dumps({"tools": [{}], "tool_choice": "none"})
        flags = ins.check_tool_calling_disabled([_exchange(request_body=body)])
        assert any(f["provider"] == "openai/anthropic" for f in flags)

    def test_no_tools_declared_not_flagged(self) -> None:
        body = json.dumps({"tool_choice": "none"})
        assert ins.check_tool_calling_disabled([_exchange(request_body=body)]) == []


# ---------------------------------------------------------------------------
# match_known_error_patterns
# ---------------------------------------------------------------------------


class TestMatchKnownErrorPatterns:
    def test_ok_span_not_scanned(self) -> None:
        spans = [_span("llm:x", status="OK")]
        assert ins.match_known_error_patterns(spans) == []

    def test_unsupported_content_block_matched(self) -> None:
        spans = [
            _span(
                "llm:x",
                status="ERROR",
                events=[_exception_event("Unsupported content block type: foo")],
            )
        ]
        flags = ins.match_known_error_patterns(spans)
        assert len(flags) == 1
        assert flags[0]["pattern"] == "cross_provider_content_block_mismatch"

    def test_unmatched_message_not_flagged(self) -> None:
        spans = [_span("llm:x", status="ERROR", events=[_exception_event("some other error")])]
        assert ins.match_known_error_patterns(spans) == []


# ---------------------------------------------------------------------------
# check_reserved_kwarg_collision
# ---------------------------------------------------------------------------


class TestCheckReservedKwargCollision:
    def test_no_collision_not_flagged(self) -> None:
        spans = [
            _span(
                "tool:x",
                status="ERROR",
                attributes={"tool.input_str": '{"a": 1}'},
                events=[
                    _exception_event(
                        "f() missing 1 required positional argument: 'a'", "TypeError"
                    )
                ],
            )
        ]
        assert ins.check_reserved_kwarg_collision(spans) == []

    def test_reserved_kwarg_collision_flagged(self) -> None:
        spans = [
            _span(
                "tool:x",
                status="ERROR",
                attributes={"tool.input_str": '{"config": {...}}'},
                events=[
                    _exception_event(
                        "f() missing 1 required positional argument: 'config'", "TypeError"
                    )
                ],
            )
        ]
        flags = ins.check_reserved_kwarg_collision(spans)
        assert len(flags) == 1
        assert flags[0]["reserved_kwarg"] == "config"

    def test_reserved_name_missing_but_absent_from_input_not_flagged(self) -> None:
        spans = [
            _span(
                "tool:x",
                status="ERROR",
                attributes={"tool.input_str": '{"other": 1}'},
                events=[
                    _exception_event(
                        "f() missing 1 required positional argument: 'config'", "TypeError"
                    )
                ],
            )
        ]
        assert ins.check_reserved_kwarg_collision(spans) == []


# ---------------------------------------------------------------------------
# multi_block_llm_responses
# ---------------------------------------------------------------------------


class TestMultiBlockLlmResponses:
    def test_single_block_not_flagged(self) -> None:
        body = json.dumps({"output": [{"type": "message", "phase": "final_answer"}]})
        assert ins.multi_block_llm_responses([_exchange(response_body=body)]) == []

    def test_multi_phase_blocks_flagged(self) -> None:
        body = json.dumps(
            {
                "output": [
                    {"type": "message", "phase": "commentary"},
                    {"type": "message", "phase": "final_answer"},
                ]
            }
        )
        flags = ins.multi_block_llm_responses([_exchange(response_body=body)])
        assert len(flags) == 1
        assert flags[0]["block_count"] == 2


# ---------------------------------------------------------------------------
# check_stream_merge_validity
# ---------------------------------------------------------------------------


class TestCheckStreamMergeValidity:
    def test_non_sse_exchange_not_flagged(self) -> None:
        assert ins.check_stream_merge_validity([_exchange(response_body="plain text")]) == []

    def test_valid_merged_tool_call_not_flagged(self) -> None:
        body = (
            'data: {"choices": [{"delta": {"tool_calls": '
            '[{"index": 0, "function": {"name": "f", "arguments": "{\\"x\\": "}}]}}]}\n'
            'data: {"choices": [{"delta": {"tool_calls": '
            '[{"index": 0, "function": {"arguments": "1}"}}]}}]}\n'
            "data: [DONE]\n"
        )
        exchange = {
            **_exchange(response_body=body),
            "response_headers": {"Content-Type": "text/event-stream"},
        }
        assert ins.check_stream_merge_validity([exchange]) == []

    def test_invalid_merged_tool_call_arguments_flagged(self) -> None:
        body = (
            'data: {"choices": [{"delta": {"tool_calls": '
            '[{"index": 0, "function": {"name": "f", "arguments": "{not"}}]}}]}\n'
            "data: [DONE]\n"
        )
        exchange = {
            **_exchange(response_body=body),
            "response_headers": {"Content-Type": "text/event-stream"},
        }
        flags = ins.check_stream_merge_validity([exchange])
        assert len(flags) == 1
        assert flags[0]["tool_name"] == "f"


# ---------------------------------------------------------------------------
# detect_response_shape_anomalies
# ---------------------------------------------------------------------------


class TestDetectResponseShapeAnomalies:
    def test_consistent_single_shape_not_flagged(self) -> None:
        body = json.dumps({"id": "1", "choices": []})
        exchanges = [_exchange(response_body=body), _exchange(response_body=body)]
        assert ins.detect_response_shape_anomalies(exchanges) == []

    def test_inconsistent_shape_across_calls_flagged(self) -> None:
        exchanges = [
            _exchange(response_body=json.dumps({"id": "1", "choices": []})),
            _exchange(response_body=json.dumps({"id": "1", "provider": {"x": 1}})),
        ]
        flags = ins.detect_response_shape_anomalies(exchanges)
        assert any(f["check"] == "inconsistent_response_shape" for f in flags)

    def test_null_field_with_populated_nested_flagged(self) -> None:
        body = json.dumps({"usage": None, "provider": {"name": "x"}})
        flags = ins.detect_response_shape_anomalies([_exchange(response_body=body)])
        assert any(f["check"] == "null_field_with_populated_nested" for f in flags)


# ---------------------------------------------------------------------------
# field_present_on_wire_absent_downstream
# ---------------------------------------------------------------------------


class TestFieldPresentOnWireAbsentDownstream:
    def test_field_absent_from_wire_not_flagged(self) -> None:
        exchanges = [_exchange(response_body=json.dumps({"id": "1"}))]
        spans: list[dict[str, object]] = []
        assert ins.field_present_on_wire_absent_downstream(exchanges, spans, "usage") == []

    def test_field_present_on_wire_and_downstream_not_flagged(self) -> None:
        exchanges = [_exchange(response_body=json.dumps({"usage": {"total_tokens": 5}}))]
        spans = [_span("llm:x", attributes={"llm.usage.total_tokens": 5})]
        assert ins.field_present_on_wire_absent_downstream(exchanges, spans, "usage") == []

    def test_field_present_on_wire_absent_downstream_flagged(self) -> None:
        exchanges = [_exchange(response_body=json.dumps({"usage": {"total_tokens": 5}}))]
        spans = [_span("llm:x", attributes={})]
        flags = ins.field_present_on_wire_absent_downstream(exchanges, spans, "usage")
        assert len(flags) == 1


# ---------------------------------------------------------------------------
# find_near_duplicate_sibling_content
# ---------------------------------------------------------------------------


class TestFindNearDuplicateSiblingContent:
    def test_dissimilar_content_not_flagged(self) -> None:
        spans = [
            _span("node:a", attributes={}),
            {
                "name": "tool:x",
                "status": "OK",
                "parent_id": "n1",
                "attributes": {"tool.output": "the weather is sunny"},
                "events": [],
            },
            {
                "name": "llm:y",
                "status": "OK",
                "parent_id": "n1",
                "attributes": {"llm.content": "completely unrelated text about cats"},
                "events": [],
            },
        ]
        assert ins.find_near_duplicate_sibling_content(spans) == []

    def test_near_identical_content_flagged(self) -> None:
        spans = [
            {
                "name": "tool:x",
                "status": "OK",
                "parent_id": "n1",
                "attributes": {"tool.output": "the weather today is sunny and warm"},
                "events": [],
            },
            {
                "name": "llm:y",
                "status": "OK",
                "parent_id": "n1",
                "attributes": {"llm.content": "the weather today is sunny and warm"},
                "events": [],
            },
        ]
        flags = ins.find_near_duplicate_sibling_content(spans)
        assert len(flags) == 1
        assert flags[0]["similarity"] == 1.0


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_identical_requests_hash_equal(self) -> None:
        h1 = ins.content_hash("POST", "https://a", '{"x": 1}')
        h2 = ins.content_hash("post", "https://a", '{"x": 1}')
        assert h1 == h2

    def test_different_bodies_hash_differently(self) -> None:
        h1 = ins.content_hash("POST", "https://a", '{"x": 1}')
        h2 = ins.content_hash("POST", "https://a", '{"x": 2}')
        assert h1 != h2


# ---------------------------------------------------------------------------
# run_all_exchange_checks
# ---------------------------------------------------------------------------


class TestRunAllExchangeChecks:
    def test_empty_exchanges_returns_empty_dict(self) -> None:
        assert ins.run_all_exchange_checks([]) == {}

    def test_only_triggered_checks_included(self) -> None:
        exchanges = [_exchange(response_status=500)]
        results = ins.run_all_exchange_checks(exchanges)
        assert "http_error_status" in results
        assert "orphaned_tool_call_ids" not in results
