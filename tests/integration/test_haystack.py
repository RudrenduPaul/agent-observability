"""
Integration tests for the Haystack tracer.

These tests require a real ``haystack-ai`` installation (Haystack 2.x) but do
NOT require live LLM API calls — the pipelines use pure-Python ``@component``
classes only, exercised through the real ``Pipeline.run()`` /
``PipelineBase._create_component_span`` execution path.

Run with: uv run pytest tests/integration/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("haystack", reason="haystack-ai not installed")

import haystack.tracing
from haystack import Pipeline, component


@component
class _Doubler:
    """Pure-Python component: doubles an int. No network calls."""

    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, int]:
        return {"value": value * 2}


@component
class _Adder:
    """Pure-Python component: adds one. No network calls."""

    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, int]:
        return {"value": value + 1}


@component
class _Failer:
    """Pure-Python component that always raises."""

    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, int]:
        raise ValueError("intentional component failure")


@pytest.fixture(autouse=True)
def _disable_tracing_after_each_test():
    """Haystack's tracer registration is process-global — always reset it."""
    yield
    haystack.tracing.disable_tracing()


@pytest.mark.integration
class TestHaystackTracerIntegration:
    def test_pipeline_run_produces_pipeline_and_component_spans(
        self, tmp_path: Path
    ) -> None:
        from agent_trace import Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        pipeline = Pipeline()
        pipeline.add_component("doubler", _Doubler())

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("haystack-basic") as trace:
            haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
            result = pipeline.run({"doubler": {"value": 21}})

        assert result["doubler"]["value"] == 42

        span_names = [s.name for s in trace.spans]
        assert "haystack.pipeline.run" in span_names
        assert "haystack.component.run" in span_names

    def test_component_span_is_child_of_pipeline_span(self, tmp_path: Path) -> None:
        from agent_trace import Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        pipeline = Pipeline()
        pipeline.add_component("doubler", _Doubler())

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("haystack-parenting") as trace:
            haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
            pipeline.run({"doubler": {"value": 1}})

        pipeline_span = next(s for s in trace.spans if s.name == "haystack.pipeline.run")
        component_span = next(s for s in trace.spans if s.name == "haystack.component.run")
        assert component_span.parent_id == pipeline_span.span_id

    def test_multi_component_pipeline_produces_one_span_per_component(
        self, tmp_path: Path
    ) -> None:
        from agent_trace import Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        pipeline = Pipeline()
        pipeline.add_component("doubler", _Doubler())
        pipeline.add_component("adder", _Adder())
        pipeline.connect("doubler.value", "adder.value")

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("haystack-multi") as trace:
            haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
            result = pipeline.run({"doubler": {"value": 10}})

        assert result["adder"]["value"] == 21  # (10 * 2) + 1

        component_spans = [s for s in trace.spans if s.name == "haystack.component.run"]
        assert len(component_spans) == 2
        names = {s.attributes.get("haystack.component.name") for s in component_spans}
        assert names == {"doubler", "adder"}

    def test_component_span_carries_name_and_type_tags(self, tmp_path: Path) -> None:
        from agent_trace import Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        pipeline = Pipeline()
        pipeline.add_component("doubler", _Doubler())

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("haystack-tags") as trace:
            haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
            pipeline.run({"doubler": {"value": 5}})

        component_span = next(s for s in trace.spans if s.name == "haystack.component.run")
        assert component_span.attributes.get("haystack.component.name") == "doubler"
        assert component_span.attributes.get("haystack.component.type") == "_Doubler"

    def test_content_tracing_enabled_captures_real_arguments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact capability #4574 needs: actual received component args.

        Haystack gates raw input/output content behind its own
        HAYSTACK_CONTENT_TRACING_ENABLED env var, read by the module-level
        ProxyTracer singleton at Span.set_content_tag() call time.
        """
        monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
        haystack.tracing.tracer.is_content_tracing_enabled = True
        try:
            from agent_trace import Tracer
            from agent_trace.integrations.haystack import HaystackTracer

            pipeline = Pipeline()
            pipeline.add_component("doubler", _Doubler())

            t = Tracer(trace_dir=tmp_path)
            with t.start_trace("haystack-content") as trace:
                haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
                pipeline.run({"doubler": {"value": 21}})

            component_span = next(
                s for s in trace.spans if s.name == "haystack.component.run"
            )
            # Real received arguments, not just their types.
            assert "21" in str(component_span.attributes.get("haystack.component.input"))
            assert "42" in str(component_span.attributes.get("haystack.component.output"))
        finally:
            haystack.tracing.tracer.is_content_tracing_enabled = False

    def test_failing_component_produces_error_span(self, tmp_path: Path) -> None:
        from agent_trace import SpanStatus, Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        pipeline = Pipeline()
        pipeline.add_component("failer", _Failer())

        t = Tracer(trace_dir=tmp_path)
        with pytest.raises(Exception, match="intentional component failure"):
            with t.start_trace("haystack-error") as trace:
                haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
                pipeline.run({"failer": {"value": 1}})

        error_spans = [s for s in trace.spans if s.status == SpanStatus.ERROR]
        assert error_spans, (
            f"Expected at least one ERROR span. "
            f"Spans: {[(s.name, s.status) for s in trace.spans]}"
        )
        pipeline_span = next(s for s in trace.spans if s.name == "haystack.pipeline.run")
        assert pipeline_span.status == SpanStatus.ERROR

    def test_all_spans_closed_after_clean_run(self, tmp_path: Path) -> None:
        from agent_trace import Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        pipeline = Pipeline()
        pipeline.add_component("doubler", _Doubler())

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("haystack-close") as trace:
            haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
            pipeline.run({"doubler": {"value": 1}})

        unclosed = [s for s in trace.spans if s.end_time is None]
        assert unclosed == [], f"Spans left open: {[s.name for s in unclosed]}"

    def test_current_span_returns_innermost_open_span(self, tmp_path: Path) -> None:
        from agent_trace import Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("haystack-current-span") as trace:
            h_tracer = HaystackTracer(tracer=t, trace=trace)
            assert h_tracer.current_span() is None
            with h_tracer.trace("outer") as outer_span:
                assert h_tracer.current_span().raw_span() is outer_span.raw_span()
                with h_tracer.trace("inner") as inner_span:
                    assert h_tracer.current_span().raw_span() is inner_span.raw_span()
                assert h_tracer.current_span().raw_span() is outer_span.raw_span()
            assert h_tracer.current_span() is None

    def test_disable_tracing_stops_capture(self, tmp_path: Path) -> None:
        """After disable_tracing(), a subsequent pipeline.run() must not add spans."""
        from agent_trace import Tracer
        from agent_trace.integrations.haystack import HaystackTracer

        pipeline = Pipeline()
        pipeline.add_component("doubler", _Doubler())

        t = Tracer(trace_dir=tmp_path)
        with t.start_trace("haystack-disable") as trace:
            haystack.tracing.enable_tracing(HaystackTracer(tracer=t, trace=trace))
            pipeline.run({"doubler": {"value": 1}})
            haystack.tracing.disable_tracing()
            span_count_after_disable = len(trace.spans)
            pipeline.run({"doubler": {"value": 2}})
            assert len(trace.spans) == span_count_after_disable
