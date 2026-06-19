"""
Unit tests for agent_trace.core.trace — Trace.
"""

from __future__ import annotations

import uuid

from agent_trace.core.span import Span
from agent_trace.core.trace import Trace


def _make_span(
    name: str, span_id: str, parent_id: str | None = None, trace_id: str = "trace-000"
) -> Span:
    s = Span(name=name, span_id=span_id, trace_id=trace_id, parent_id=parent_id)
    s.end()
    return s


class TestTraceDefaults:
    def test_trace_id_is_valid_uuid(self) -> None:
        trace = Trace()
        uuid.UUID(trace.trace_id)

    def test_run_id_is_valid_uuid(self) -> None:
        trace = Trace()
        uuid.UUID(trace.run_id)

    def test_spans_is_empty_list(self) -> None:
        trace = Trace()
        assert trace.spans == []

    def test_metadata_is_empty_dict(self) -> None:
        trace = Trace()
        assert trace.metadata == {}

    def test_trace_id_and_run_id_differ(self) -> None:
        trace = Trace()
        # By default they are independently generated
        assert trace.trace_id != trace.run_id


class TestAddSpan:
    def test_add_span_appends(self) -> None:
        trace = Trace(trace_id="t-001")
        span = _make_span("root", "s-001", trace_id="t-001")
        trace.add_span(span)
        assert len(trace.spans) == 1
        assert trace.spans[0] is span

    def test_add_span_multiple(self) -> None:
        trace = Trace(trace_id="t-001")
        for i in range(5):
            trace.add_span(_make_span(f"span-{i}", f"s-{i:03d}", trace_id="t-001"))
        assert len(trace.spans) == 5

    def test_add_span_corrects_mismatched_trace_id(self) -> None:
        trace = Trace(trace_id="correct-trace-id")
        span = Span(name="orphan", span_id="s-orphan", trace_id="wrong-trace-id")
        trace.add_span(span)
        # add_span silently corrects the trace_id
        assert span.trace_id == "correct-trace-id"
        assert trace.spans[0].trace_id == "correct-trace-id"

    def test_add_span_matching_trace_id_unchanged(self) -> None:
        trace = Trace(trace_id="t-001")
        span = _make_span("root", "s-001", trace_id="t-001")
        trace.add_span(span)
        assert span.trace_id == "t-001"


class TestGetSpan:
    def test_get_span_returns_correct_span(self) -> None:
        trace = Trace(trace_id="t-001")
        span = _make_span("target", "s-target", trace_id="t-001")
        trace.add_span(_make_span("other", "s-other", trace_id="t-001"))
        trace.add_span(span)
        found = trace.get_span("s-target")
        assert found is span

    def test_get_span_returns_none_if_missing(self) -> None:
        trace = Trace()
        assert trace.get_span("nonexistent") is None

    def test_get_span_empty_trace(self) -> None:
        trace = Trace()
        assert trace.get_span("any") is None


class TestRootSpans:
    def test_root_spans_returns_spans_without_parent(self) -> None:
        trace = Trace(trace_id="t-001")
        root = _make_span("root", "s-root", parent_id=None, trace_id="t-001")
        child = _make_span("child", "s-child", parent_id="s-root", trace_id="t-001")
        trace.add_span(root)
        trace.add_span(child)
        roots = trace.root_spans()
        assert len(roots) == 1
        assert roots[0] is root

    def test_root_spans_empty_trace(self) -> None:
        trace = Trace()
        assert trace.root_spans() == []

    def test_root_spans_all_roots(self) -> None:
        trace = Trace(trace_id="t-001")
        for i in range(3):
            trace.add_span(
                _make_span(f"root-{i}", f"s-{i}", parent_id=None, trace_id="t-001")
            )
        assert len(trace.root_spans()) == 3

    def test_root_spans_no_roots_when_all_have_parents(self) -> None:
        trace = Trace(trace_id="t-001")
        trace.add_span(
            _make_span("child", "s-child", parent_id="ghost-parent", trace_id="t-001")
        )
        assert trace.root_spans() == []


class TestChildrenOf:
    def test_children_of_returns_direct_children(self) -> None:
        trace = Trace(trace_id="t-001")
        root = _make_span("root", "s-root", trace_id="t-001")
        child1 = _make_span("child1", "s-c1", parent_id="s-root", trace_id="t-001")
        child2 = _make_span("child2", "s-c2", parent_id="s-root", trace_id="t-001")
        grandchild = _make_span("gc", "s-gc", parent_id="s-c1", trace_id="t-001")
        trace.add_span(root)
        trace.add_span(child1)
        trace.add_span(child2)
        trace.add_span(grandchild)

        children = trace.children_of("s-root")
        assert len(children) == 2
        assert child1 in children
        assert child2 in children
        # grandchild is NOT a direct child of root
        assert grandchild not in children

    def test_children_of_unknown_span(self) -> None:
        trace = Trace()
        assert trace.children_of("unknown") == []

    def test_children_of_leaf_span(self) -> None:
        trace = Trace(trace_id="t-001")
        leaf = _make_span("leaf", "s-leaf", trace_id="t-001")
        trace.add_span(leaf)
        assert trace.children_of("s-leaf") == []


class TestTraceSerialization:
    def test_to_dict_contains_expected_keys(self) -> None:
        trace = Trace()
        d = trace.to_dict()
        assert set(d.keys()) == {"trace_id", "run_id", "spans", "metadata"}

    def test_to_dict_spans_preserved_in_order(self) -> None:
        trace = Trace(trace_id="t-001")
        for i in range(3):
            trace.add_span(_make_span(f"span-{i}", f"s-{i}", trace_id="t-001"))
        d = trace.to_dict()
        names = [s["name"] for s in d["spans"]]
        assert names == ["span-0", "span-1", "span-2"]

    def test_round_trip_all_spans_preserved(self) -> None:
        trace = Trace(trace_id="t-roundtrip", run_id="run-001")
        trace.metadata["key"] = "value"
        for i in range(3):
            s = _make_span(f"span-{i}", f"s-{i:03d}", trace_id="t-roundtrip")
            trace.add_span(s)

        d = trace.to_dict()
        restored = Trace.from_dict(d)

        assert restored.trace_id == trace.trace_id
        assert restored.run_id == trace.run_id
        assert len(restored.spans) == 3
        for orig, rest in zip(trace.spans, restored.spans):
            assert rest.span_id == orig.span_id
            assert rest.name == orig.name

    def test_round_trip_metadata_preserved(self) -> None:
        trace = Trace(trace_id="t-meta")
        trace.metadata["env"] = "test"
        trace.metadata["version"] = "1"
        restored = Trace.from_dict(trace.to_dict())
        assert restored.metadata == trace.metadata

    def test_from_dict_missing_optional_fields_use_defaults(self) -> None:
        minimal = {
            "trace_id": str(uuid.uuid4()),
            "run_id": str(uuid.uuid4()),
        }
        trace = Trace.from_dict(minimal)
        assert trace.spans == []
        assert trace.metadata == {}

    def test_empty_trace_serializes(self) -> None:
        trace = Trace()
        d = trace.to_dict()
        assert d["spans"] == []
        restored = Trace.from_dict(d)
        assert restored.spans == []
        assert restored.root_spans() == []
