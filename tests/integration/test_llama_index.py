"""
Integration tests for the llama_index integration.

These tests require a real llama-index-core installation but do NOT require
live LLM API calls — they use llama_index's own `MockLLM` and a plain
`FunctionTool` (arbitrary Python functions), so they run with zero
credentials.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("llama_index.core", reason="llama-index-core not installed")


def _add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _boom(a: int) -> int:
    """Always raises."""
    raise RuntimeError("intentional failure")


@pytest.mark.integration
class TestLlamaIndexIntegration:
    def test_context_manager_captures_llm_chat_span(self, tmp_path: Path) -> None:
        """LlamaIndexTracer must produce a span for a MockLLM.chat() call."""
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.llms import MockLLM

        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-chat-test") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                llm = MockLLM()
                llm.chat([ChatMessage(role="user", content="hi")])

        span_names = [s.name for s in trace.spans]
        assert any("chat" in name.lower() for name in span_names), (
            f"Expected a span for MockLLM.chat, got: {span_names}"
        )

    def test_nested_llm_call_produces_parent_child_span_tree(
        self, tmp_path: Path
    ) -> None:
        """MockLLM.chat() internally calls MockLLM.complete(); the resulting
        spans must be nested (complete's span is a child of chat's span),
        confirming dispatcher parent_span_id propagation is wired correctly."""
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.llms import MockLLM

        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-nesting-test") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                llm = MockLLM()
                llm.chat([ChatMessage(role="user", content="hi")])

        chat_span = next(s for s in trace.spans if "chat" in s.name.lower())
        complete_span = next(s for s in trace.spans if "complete" in s.name.lower())
        assert complete_span.parent_id == chat_span.span_id

    def test_all_spans_closed_and_ok_on_clean_run(self, tmp_path: Path) -> None:
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.llms import MockLLM

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-clean-run") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                llm = MockLLM()
                llm.chat([ChatMessage(role="user", content="hi")])

        assert trace.spans, "Expected at least one span"
        for span in trace.spans:
            assert span.end_time is not None, f"{span.name} was left open"
            assert span.status == SpanStatus.OK, f"{span.name} was not OK"

    def test_llm_chat_start_event_enriches_span(self, tmp_path: Path) -> None:
        """The LLMChatStartEvent must land on the chat span with message data."""
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.llms import MockLLM

        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-event-test") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                llm = MockLLM()
                llm.chat([ChatMessage(role="user", content="hello world")])

        chat_span = next(s for s in trace.spans if "chat" in s.name.lower())
        assert chat_span.attributes.get("llm.messages_count") == 1
        assert chat_span.attributes.get("llm.last_message_role") == "MessageRole.USER"
        assert chat_span.attributes.get("llm.last_message_content") == "hello world"
        # LLMChatEndEvent must also have landed on the same span.
        assert "llm.response_content" in chat_span.attributes
        assert chat_span.attributes.get("llm.has_tool_calls") is False

    def test_tool_call_produces_error_span_on_exception(self, tmp_path: Path) -> None:
        """A FunctionTool that raises must produce an ERROR span, not a silently
        dropped one — mirrors the LangGraph on_tool_error contract."""
        from llama_index.core.tools import FunctionTool

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-tool-error") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                tool = FunctionTool.from_defaults(fn=_boom)
                with pytest.raises(RuntimeError, match="intentional failure"):
                    tool.call(a=1)

        error_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        assert error_spans, f"Expected an ERROR span. Spans: {trace.spans}"
        exc_events = [
            e for span in error_spans for e in span.events if e.name == "exception"
        ]
        assert exc_events, "Expected an 'exception' SpanEvent on the error span"
        assert exc_events[0].attributes["exception.message"] == "intentional failure"

    def test_successful_tool_call_produces_ok_span(self, tmp_path: Path) -> None:
        from llama_index.core.tools import FunctionTool

        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-tool-ok") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                tool = FunctionTool.from_defaults(fn=_add)
                result = tool.call(a=1, b=2)

        assert result.raw_output == 3
        tool_spans = [s for s in trace.spans if "FunctionTool" in s.name]
        assert tool_spans
        assert tool_spans[0].status == SpanStatus.OK

    def test_span_carries_llama_index_class_attribute(self, tmp_path: Path) -> None:
        from llama_index.core.tools import FunctionTool

        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-attr-test") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                tool = FunctionTool.from_defaults(fn=_add)
                tool.call(a=1, b=2)

        tool_span = next(s for s in trace.spans if "FunctionTool" in s.name)
        assert tool_span.attributes.get("llama_index.class") == "FunctionTool"
        assert "llama_index.span_id" in tool_span.attributes

    # ------------------------------------------------------------------
    # Install / uninstall lifecycle
    # ------------------------------------------------------------------

    def test_uninstall_stops_capturing_spans(self, tmp_path: Path) -> None:
        """After the context manager exits, further llama_index calls must
        not add spans to the (now-closed) trace — no leaked global handler."""
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.llms import MockLLM

        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        llm = MockLLM()
        with t.start_trace("li-uninstall-test") as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                llm.chat([ChatMessage(role="user", content="inside")])
            span_count_after_exit = len(trace.spans)

            # Outside the `with` block: the tracer must be uninstalled, so this
            # call must not add any further spans to the trace.
            llm.chat([ChatMessage(role="user", content="outside")])
            assert len(trace.spans) == span_count_after_exit

    def test_manual_install_uninstall_round_trip(self, tmp_path: Path) -> None:
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.llms import MockLLM

        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-manual-install") as trace:
            li_tracer = LlamaIndexTracer(tracer=t, trace=trace)
            li_tracer.install()
            try:
                MockLLM().chat([ChatMessage(role="user", content="hi")])
            finally:
                li_tracer.uninstall()

            count_after_uninstall = len(trace.spans)
            assert count_after_uninstall > 0
            MockLLM().chat([ChatMessage(role="user", content="hi again")])
            assert len(trace.spans) == count_after_uninstall

    def test_uninstall_without_install_is_a_noop(self, tmp_path: Path) -> None:
        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("li-noop-uninstall") as trace:
            li_tracer = LlamaIndexTracer(tracer=t, trace=trace)
            li_tracer.uninstall()  # must not raise

    def test_two_tracers_do_not_cross_contaminate_after_one_uninstalls(
        self, tmp_path: Path
    ) -> None:
        """Installing/uninstalling one LlamaIndexTracer must not disturb a
        second one that remains installed on the same (root) dispatcher."""
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.llms import MockLLM

        from agent_trace import Tracer
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t1 = Tracer(trace_dir=tmp_path)
        t2 = Tracer(trace_dir=tmp_path)
        with (
            t1.start_trace("li-multi-1") as trace1,
            t2.start_trace("li-multi-2") as trace2,
        ):
            tracer1 = LlamaIndexTracer(tracer=t1, trace=trace1)
            tracer2 = LlamaIndexTracer(tracer=t2, trace=trace2)
            tracer1.install()
            tracer2.install()
            try:
                MockLLM().chat([ChatMessage(role="user", content="both active")])
                assert len(trace1.spans) > 0
                assert len(trace2.spans) > 0

                tracer1.uninstall()
                count1 = len(trace1.spans)
                count2 = len(trace2.spans)
                MockLLM().chat([ChatMessage(role="user", content="only t2 active")])
                assert len(trace1.spans) == count1, "tracer1 must stay uninstalled"
                assert len(trace2.spans) > count2, "tracer2 must still be capturing"
            finally:
                tracer2.uninstall()

    # ------------------------------------------------------------------
    # Replay compatibility
    # ------------------------------------------------------------------

    def test_replay_context_allows_llama_index_pure_python_tool(
        self, tmp_path: Path
    ) -> None:
        """Record then replay a FunctionTool call. No HTTP calls are made
        (pure Python function), so AGENT_TRACE_NETWORK_GUARD=1 is satisfied
        automatically."""
        from llama_index.core.tools import FunctionTool

        from agent_trace import Tracer, replay
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace(
            "li-replay-record", record=True, run_id="li-replay-run"
        ) as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                tool = FunctionTool.from_defaults(fn=_add)
                tool.call(a=2, b=3)

        assert len(trace.spans) >= 1

        with replay("li-replay-run", trace_dir=tmp_path):
            tool = FunctionTool.from_defaults(fn=_add)
            result = tool.call(a=2, b=3)

        assert result.raw_output == 5

    # ------------------------------------------------------------------
    # HTTP-exchange-to-originating-span correlation (#13449)
    # ------------------------------------------------------------------

    def test_tool_http_call_tagged_with_originating_span_id(
        self, tmp_path: Path
    ) -> None:
        """A FunctionTool making a real HTTP call must have that exchange
        recoverable via Fixture.exchanges_for_correlation_id(span_id) — the
        exact "attribute this HTTP exchange to the tool/step that made it"
        gap #13449 flagged (no span_id/node column on http_exchanges)."""
        import httpx
        from llama_index.core.tools import FunctionTool

        from agent_trace import Tracer
        from agent_trace._replay.fixture import Fixture
        from agent_trace.integrations.llama_index import LlamaIndexTracer

        def _call_api(city: str) -> str:
            client = httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"city": city})
                )
            )
            response = client.get(f"https://api.example.com/weather/{city}")
            return response.text

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace(
            "li-correlation-test", record=True, run_id="li-correlation-run"
        ) as trace:
            with LlamaIndexTracer(tracer=t, trace=trace):
                tool = FunctionTool.from_defaults(fn=_call_api)
                tool.call(city="Boston")

        tool_span = next(s for s in trace.spans if "FunctionTool" in s.name)

        with Fixture(tmp_path / "li-correlation-run" / "fixture.db") as fixture:
            exchanges = fixture.exchanges_for_correlation_id(tool_span.span_id)

        assert len(exchanges) == 1
        assert exchanges[0]["url"] == "https://api.example.com/weather/Boston"
