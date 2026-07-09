"""
Unit tests for agent_trace.integrations.llama_index.

These tests do NOT require llama-index-core to be installed:
  - The import-guard / friendly-error path is tested by forcing
    `import llama_index` to fail (matching how the real package's absence
    behaves — `sys.modules["llama_index"] = None` reproduces the exact
    ModuleNotFoundError Python raises when a package truly isn't installed).
  - `_span_name`, `_truncate`, and `_apply_event` are pure/duck-typed helpers
    that never import llama_index themselves, so they're tested directly
    against lightweight fake event objects.

Full round-trip tests against the real Dispatcher/BaseSpanHandler/
BaseEventHandler classes live in tests/integration/test_llama_index.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_trace import Tracer
from agent_trace.core.span import SpanStatus
from agent_trace.integrations.llama_index import (
    _apply_event,
    _span_name,
    _truncate,
)

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


@pytest.fixture()
def llama_index_absent(monkeypatch: pytest.MonkeyPatch):
    """Force `import llama_index` (and submodules) to fail like it's not installed.

    Also resets the module-level lazily-built handler-class cache
    (`_SpanHandlerClass`/`_EventHandlerClass` in
    ``agent_trace.integrations.llama_index``) so this test exercises the real
    guard behavior regardless of whether some *other* test in the same
    process already imported real llama-index-core and populated that cache
    — without the reset, `_get_handler_classes()` would short-circuit on the
    cached classes and never re-enter `_require_llama_index()`, making this
    test's outcome depend on suite execution order instead of testing the
    guard itself.
    """
    import agent_trace.integrations.llama_index as li_module

    for name in list(sys.modules):
        if name == "llama_index" or name.startswith("llama_index."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "llama_index", None)
    monkeypatch.setattr(li_module, "_SpanHandlerClass", None, raising=False)
    monkeypatch.setattr(li_module, "_EventHandlerClass", None, raising=False)
    yield


class TestImportGuard:
    def test_require_llama_index_raises_with_hint(self, llama_index_absent) -> None:
        from agent_trace.integrations.llama_index import _require_llama_index

        with pytest.raises(ImportError, match="pip install llama-index-core"):
            _require_llama_index()

    def test_tracer_construction_raises_when_absent(
        self, tmp_path: Path, llama_index_absent
    ) -> None:
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-absent") as trace, pytest.raises(ImportError):
            LlamaIndexTracer(tracer=t, trace=trace)


# ---------------------------------------------------------------------------
# _span_name
# ---------------------------------------------------------------------------


class TestSpanName:
    def test_strips_uuid4_suffix_from_instance_method_id(self) -> None:
        id_ = "MockLLM.chat-7521028a-b975-42ec-8387-ea352e9cf73f"
        assert _span_name(id_) == "MockLLM.chat"

    def test_strips_uuid4_suffix_from_function_qualname_id(self) -> None:
        id_ = "my_module.my_func-6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        assert _span_name(id_) == "my_module.my_func"

    def test_id_without_uuid_suffix_is_returned_unchanged(self) -> None:
        # Defensive: if the id doesn't match the expected shape, don't mangle it.
        assert _span_name("just_a_plain_id") == "just_a_plain_id"

    def test_uppercase_hex_uuid_is_also_stripped(self) -> None:
        id_ = "Foo.bar-6BA7B810-9DAD-11D1-80B4-00C04FD430C8"
        assert _span_name(id_) == "Foo.bar"


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_value_is_returned_as_is(self) -> None:
        assert _truncate("hello") == "hello"

    def test_long_value_is_truncated_with_marker(self) -> None:
        long_value = "x" * 5000
        result = _truncate(long_value)
        assert len(result) < len(long_value)
        assert result.endswith("...<truncated>")

    def test_non_string_value_is_stringified(self) -> None:
        assert _truncate(42) == "42"


# ---------------------------------------------------------------------------
# _apply_event — pure dispatch logic against fake (duck-typed) events
# ---------------------------------------------------------------------------


def _open_span(tmp_path: Path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace("apply-event-test") as trace:
        span = t.start_span("test-span")
        return t, trace, span


def _fake_event(class_name: str, **fields: object) -> SimpleNamespace:
    ns = SimpleNamespace(**fields)
    ns.class_name = lambda: class_name
    return ns


class TestApplyEventLLMChat:
    def test_chat_start_records_message_count_and_model(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        message = SimpleNamespace(role="user", content="hello there")
        event = _fake_event(
            "LLMChatStartEvent",
            messages=[message],
            model_dict={"model": "gpt-4o-mini"},
        )
        _apply_event(span, event)
        assert span.attributes["llm.messages_count"] == 1
        assert span.attributes["llm.model"] == "gpt-4o-mini"
        assert span.attributes["llm.last_message_role"] == "user"
        assert span.attributes["llm.last_message_content"] == "hello there"

    def test_chat_start_falls_back_to_model_name_key(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event(
            "LLMChatStartEvent", messages=[], model_dict={"model_name": "claude-3"}
        )
        _apply_event(span, event)
        assert span.attributes["llm.model"] == "claude-3"

    def test_chat_start_with_no_messages_does_not_crash(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event("LLMChatStartEvent", messages=None, model_dict=None)
        _apply_event(span, event)  # must not raise
        assert span.attributes["llm.messages_count"] == 0

    def test_chat_end_records_response_content_and_tool_calls_flag(
        self, tmp_path: Path
    ) -> None:
        _, _, span = _open_span(tmp_path)
        message = SimpleNamespace(
            content="final answer", additional_kwargs={"tool_calls": [{"id": "1"}]}
        )
        response = SimpleNamespace(message=message)
        event = _fake_event("LLMChatEndEvent", response=response)
        _apply_event(span, event)
        assert span.attributes["llm.response_content"] == "final answer"
        assert span.attributes["llm.has_tool_calls"] is True

    def test_chat_end_without_tool_calls_sets_false(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        message = SimpleNamespace(content="ok", additional_kwargs={})
        response = SimpleNamespace(message=message)
        event = _fake_event("LLMChatEndEvent", response=response)
        _apply_event(span, event)
        assert span.attributes["llm.has_tool_calls"] is False

    def test_chat_end_with_none_response_does_not_crash(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event("LLMChatEndEvent", response=None)
        _apply_event(span, event)  # must not raise
        assert "llm.response_content" not in span.attributes


class TestApplyEventCompletion:
    def test_completion_start_records_prompt(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event("LLMCompletionStartEvent", prompt="summarize this")
        _apply_event(span, event)
        assert span.attributes["llm.prompt"] == "summarize this"

    def test_completion_end_records_response_text(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        response = SimpleNamespace(text="the summary")
        event = _fake_event("LLMCompletionEndEvent", response=response)
        _apply_event(span, event)
        assert span.attributes["llm.response_content"] == "the summary"


class TestApplyEventTool:
    def test_agent_tool_call_adds_event_with_name_and_arguments(
        self, tmp_path: Path
    ) -> None:
        _, _, span = _open_span(tmp_path)
        tool = SimpleNamespace(name="web_search")
        event = _fake_event(
            "AgentToolCallEvent", tool=tool, arguments='{"query": "agent-trace"}'
        )
        _apply_event(span, event)
        assert len(span.events) == 1
        tool_event = span.events[0]
        assert tool_event.name == "tool_call"
        assert tool_event.attributes["tool.name"] == "web_search"
        assert tool_event.attributes["tool.arguments"] == '{"query": "agent-trace"}'

    def test_agent_tool_call_with_no_tool_uses_unknown(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event("AgentToolCallEvent", tool=None, arguments="{}")
        _apply_event(span, event)
        assert span.events[0].attributes["tool.name"] == "unknown"


class TestApplyEventAgentStep:
    def test_run_step_start_records_task_id_and_input(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event(
            "AgentRunStepStartEvent", task_id="task-123", input="what's the weather?"
        )
        _apply_event(span, event)
        assert span.attributes["agent.task_id"] == "task-123"
        assert span.attributes["agent.step_input"] == "what's the weather?"

    def test_run_step_end_records_step_output(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event("AgentRunStepEndEvent", step_output="it's sunny")
        _apply_event(span, event)
        assert span.attributes["agent.step_output"] == "it's sunny"


class TestApplyEventException:
    def test_exception_event_records_exception_and_marks_error(
        self, tmp_path: Path
    ) -> None:
        _, _, span = _open_span(tmp_path)
        exc = ValueError("stale streaming delta mistaken for tool call")
        event = _fake_event("ExceptionEvent", exception=exc)
        _apply_event(span, event)
        assert span.status == SpanStatus.ERROR
        assert any(e.name == "exception" for e in span.events)
        exc_event = next(e for e in span.events if e.name == "exception")
        assert exc_event.attributes["exception.type"] == "ValueError"

    def test_exception_event_with_non_exception_value_is_ignored(
        self, tmp_path: Path
    ) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event("ExceptionEvent", exception="not actually an exception")
        _apply_event(span, event)  # must not raise
        assert span.status != SpanStatus.ERROR


class TestApplyEventUnknown:
    def test_unrecognized_event_class_is_a_noop(self, tmp_path: Path) -> None:
        _, _, span = _open_span(tmp_path)
        event = _fake_event("SomeFutureEventType", foo="bar")
        _apply_event(span, event)  # must not raise
        assert span.attributes == {}
        assert span.events == []
