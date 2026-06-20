#!/usr/bin/env bash
# Run with: bash scripts/create-launch-issues.sh
# Creates 19 remaining good-first-issues (#7 already exists).
# Requires: gh auth login with write:issues scope.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "Creating 19 good-first-issues for agent-trace launch..."

gh issue create \
  --title "Add Trace.failed_spans property" \
  --label "good first issue" \
  --body "## What to build

Add a \`failed_spans\` property to \`Trace\` that returns all spans with \`SpanStatus.ERROR\`. Return \`None\`-safe: empty list if no errors.

## Where the code is

\`src/agent_trace/core/trace.py\` — add after the \`root_spans\` property (~line 66). Import \`SpanStatus\` from \`agent_trace.core.span\`.

## What the test should check

\`\`\`python
# tests/unit/test_core_trace.py
def test_failed_spans_filters_correctly(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"errs\") as trace:
        ok = t.start_span(\"ok\"); ok.end(SpanStatus.OK)
        err = t.start_span(\"err\"); err.end(SpanStatus.ERROR)
    assert len(trace.failed_spans) == 1
    assert trace.failed_spans[0].name == \"err\"

def test_failed_spans_empty_when_all_ok(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"clean\") as trace:
        s = t.start_span(\"s\"); s.end()
    assert trace.failed_spans == []
\`\`\`

## Estimated effort

20 minutes. One-liner list comprehension."

gh issue create \
  --title "Add Tracer.clear_plugins() method" \
  --label "good first issue" \
  --body "## What to build

Add a \`clear_plugins()\` method to \`Tracer\` that removes all registered plugins in one call. Useful in tests to reset state between runs.

## Where the code is

\`src/agent_trace/__init__.py\` — add after \`remove_plugin()\`.

## What the test should check

\`\`\`python
# tests/unit/test_plugins.py
def test_clear_plugins(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    t.add_plugin(RecordingPlugin())
    t.add_plugin(RecordingPlugin())
    t.clear_plugins()
    assert t._plugins == []

def test_clear_plugins_noop_when_empty(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    t.clear_plugins()  # must not raise
\`\`\`

## Estimated effort

15 minutes. One-liner \`self._plugins.clear()\`."

gh issue create \
  --title "Add Tracer.active_trace property" \
  --label "good first issue" \
  --body "## What to build

Add an \`active_trace\` property to \`Tracer\` that returns the current \`Trace\` from the context variable, or \`None\` if no trace is active.

## Where the code is

\`src/agent_trace/__init__.py\` — the context variable is already maintained for span parenting; expose it via a property. Look for \`_trace_var\` or similar ContextVar.

## What the test should check

\`\`\`python
# tests/unit/test_tracer.py
def test_active_trace_none_outside(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    assert t.active_trace is None

def test_active_trace_inside_context(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"check\") as trace:
        assert t.active_trace is trace
    assert t.active_trace is None
\`\`\`

## Estimated effort

20 minutes. Reads from an existing ContextVar."

gh issue create \
  --title "Add on_span_error hook to SpanPlugin protocol" \
  --label "good first issue" \
  --body "## What to build

Add an \`on_span_error(span, exc)\` hook to the \`SpanPlugin\` protocol and \`PluginBase\` base class. This hook fires when \`span.end(SpanStatus.ERROR)\` is called, giving plugins a dedicated error-observation entry point.

## Where the code is

1. \`src/agent_trace/plugins/base.py\` — add \`on_span_error(self, span: Span, exc: BaseException | None) -> None\` to \`SpanPlugin\` and \`PluginBase\`
2. \`src/agent_trace/__init__.py\` — call \`on_span_error\` inside the \`_plugin_end\` closure when \`status == SpanStatus.ERROR\`

## What the test should check

\`\`\`python
# tests/unit/test_plugins.py
def test_on_span_error_called_on_error_status(tmp_path):
    errors = []
    class ErrPlugin(PluginBase):
        def on_span_error(self, span, exc):
            errors.append(span.name)
    t = Tracer(trace_dir=tmp_path)
    t.add_plugin(ErrPlugin())
    with t.start_trace(\"e\"):
        s = t.start_span(\"fail\"); s.end(SpanStatus.ERROR)
    assert errors == [\"fail\"]

def test_on_span_error_not_called_on_ok(tmp_path):
    errors = []
    class ErrPlugin(PluginBase):
        def on_span_error(self, span, exc): errors.append(1)
    t = Tracer(trace_dir=tmp_path)
    t.add_plugin(ErrPlugin())
    with t.start_trace(\"ok\"):
        s = t.start_span(\"s\"); s.end()
    assert errors == []
\`\`\`

## Estimated effort

45 minutes. Protocol extension + hook dispatch."

gh issue create \
  --title "Add FixtureNotFoundError with diagnostic message" \
  --label "good first issue" \
  --body "## What to build

When the replay engine cannot find a fixture for a given \`trace_id\`, it currently raises a generic \`KeyError\`. Replace this with a \`FixtureNotFoundError\` that includes the fixture path and trace_id in the message.

## Where the code is

1. \`src/agent_trace/core/exceptions.py\` — define \`class FixtureNotFoundError(LookupError): pass\`
2. \`src/agent_trace/_replay/engine.py\` — catch the \`KeyError\` / missing fixture case and raise \`FixtureNotFoundError(f\"No fixture for trace_id={trace_id!r} in {fixture_path}\")\`
3. \`src/agent_trace/__init__.py\` — export \`FixtureNotFoundError\` from the public API

## What the test should check

\`\`\`python
# tests/unit/test_replay_engine.py
def test_fixture_not_found_error(tmp_path):
    from agent_trace import FixtureNotFoundError
    # create engine pointing at empty fixture dir
    with pytest.raises(FixtureNotFoundError, match=\"No fixture\"):
        with replay(trace_id=\"nonexistent\", fixture_dir=tmp_path):
            pass
\`\`\`

## Estimated effort

30 minutes. New exception class + raise site + export."

gh issue create \
  --title "Support AGENT_TRACE_FIXTURE_DIR environment variable" \
  --label "good first issue" \
  --body "## What to build

Allow users to set \`AGENT_TRACE_FIXTURE_DIR=/path/to/fixtures\` as an environment variable so they don't have to pass \`fixture_dir\` explicitly in every \`record()\` / \`replay()\` call.

## Where the code is

\`src/agent_trace/_replay/engine.py\` — in \`record()\` and \`replay()\`, if \`fixture_dir\` argument is \`None\`, fall back to \`os.environ.get(\"AGENT_TRACE_FIXTURE_DIR\")\` before defaulting to the current directory.

## What the test should check

\`\`\`python
# tests/unit/test_replay_engine.py
def test_fixture_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv(\"AGENT_TRACE_FIXTURE_DIR\", str(tmp_path))
    # record without explicit fixture_dir — should use env var
    with record():  # no fixture_dir arg
        tracer.start_trace(\"env-test\")
    assert any(tmp_path.glob(\"*.db\"))
\`\`\`

## Estimated effort

30 minutes. \`os.environ.get\` + two fallback sites."

gh issue create \
  --title "Add --version flag to the CLI" \
  --label "good first issue" \
  --body "## What to build

\`python -m agent_trace --version\` should print the installed package version and exit 0. Example output: \`agent-trace 0.1.0\`

## Where the code is

\`src/agent_trace/_cli.py\` — the CLI entry point. Use Python's \`importlib.metadata.version(\"agent-trace\")\` to read the version at runtime (no hardcoding).

## What the test should check

\`\`\`python
# tests/unit/test_cli.py
import subprocess, sys
def test_cli_version():
    result = subprocess.run(
        [sys.executable, \"-m\", \"agent_trace\", \"--version\"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert \"agent-trace\" in result.stdout
    # version string is semver-like
    import re
    assert re.search(r\"\\d+\\.\\d+\\.\\d+\", result.stdout)
\`\`\`

## Estimated effort

15 minutes. One \`argparse\` flag."

gh issue create \
  --title "Add pytest plugin / fixture for Tracer" \
  --label "good first issue" \
  --body "## What to build

Publish a \`tracer\` pytest fixture via a plugin entry point so users can write:

\`\`\`python
def test_my_agent(tracer):
    with tracer.start_trace(\"test\"):
        ...
\`\`\`

without manually constructing \`Tracer(trace_dir=tmp_path)\` in every test.

## Where the code is

1. Create \`src/agent_trace/pytest_plugin.py\` — define \`@pytest.fixture def tracer(tmp_path): return Tracer(trace_dir=tmp_path)\`
2. \`pyproject.toml\` — register it under \`[project.entry-points.\"pytest11\"]\`: \`agent_trace = \"agent_trace.pytest_plugin\"\`

## What the test should check

\`\`\`python
# tests/unit/test_pytest_plugin.py
# The fixture itself is auto-available; just use it:
def test_tracer_fixture_is_tracer_instance(tracer):
    from agent_trace import Tracer
    assert isinstance(tracer, Tracer)

def test_tracer_fixture_trace_dir_exists(tracer):
    assert tracer._trace_dir.exists()
\`\`\`

## Estimated effort

45 minutes. \`pytest11\` entry point + fixture function."

gh issue create \
  --title "Add Span.child_count property" \
  --label "good first issue" \
  --body "## What to build

Add a \`child_count\` property to \`Span\` that returns the number of direct child spans within its trace. This requires access to the parent \`Trace\` object — pass it at construction time or look it up from context.

## Where the code is

\`src/agent_trace/core/span.py\` — \`Span\` class. Also look at \`src/agent_trace/core/trace.py:children_of()\` which already computes children.

**Hint:** The simplest approach is a method on \`Trace\` rather than \`Span\`:
\`trace.children_of(span.span_id)\` already works — document that instead of adding a property that requires back-references.

Consider opening a discussion on the PR about which API feels cleaner before implementing.

## What the test should check

\`\`\`python
# tests/unit/test_core_trace.py
def test_children_of_count(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"c\") as trace:
        parent = t.start_span(\"parent\")
        child1 = t.start_span(\"c1\", parent_id=parent.span_id)
        child2 = t.start_span(\"c2\", parent_id=parent.span_id)
        child1.end(); child2.end(); parent.end()
    assert len(trace.children_of(parent.span_id)) == 2
\`\`\`

## Estimated effort

20–45 minutes depending on chosen approach."

gh issue create \
  --title "Add Trace.to_flamegraph_json() for Perfetto visualization" \
  --label "good first issue" \
  --body "## What to build

Add \`Trace.to_flamegraph_json()\` that returns a JSON dict in Perfetto's trace format so users can drag-drop the file into \`ui.perfetto.dev\` and see a flamegraph of their agent run.

Perfetto JSON format: \`{\"traceEvents\": [{\"name\": ..., \"ph\": \"X\", \"ts\": ..., \"dur\": ..., \"pid\": 1, \"tid\": 1}]}\`

## Where the code is

\`src/agent_trace/core/trace.py\` — add \`to_flamegraph_json(self) -> dict[str, Any]\`. Convert each span's \`start_time\` (seconds) to microseconds for Perfetto's \`ts\` and \`dur\` fields.

## What the test should check

\`\`\`python
# tests/unit/test_core_trace.py
def test_to_flamegraph_json_schema(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"fg\") as trace:
        s = t.start_span(\"s\"); s.end()
    fg = trace.to_flamegraph_json()
    assert \"traceEvents\" in fg
    event = fg[\"traceEvents\"][0]
    assert {\"name\", \"ph\", \"ts\", \"dur\", \"pid\", \"tid\"} <= set(event.keys())
    assert event[\"ph\"] == \"X\"  # complete event type
\`\`\`

## Estimated effort

1.5–2 hours. Involves timestamp conversion and understanding the Perfetto JSON schema."

gh issue create \
  --title "Add JSON Schema for the trace fixture format" \
  --label "good first issue" \
  --body "## What to build

Add \`schemas/trace-v1.json\` — a JSON Schema (draft-07) that describes the \`trace.json\` file format written by \`exporters/file.py\`. Use it in tests to validate fixture output.

## Where the code is

1. Read \`src/agent_trace/exporters/file.py\` and \`src/agent_trace/core/trace.py:to_dict()\` to understand the output shape.
2. Create \`schemas/trace-v1.json\` matching that shape.
3. In tests, validate existing trace fixture output against the schema using \`jsonschema\`.

## What the test should check

\`\`\`python
# tests/unit/test_exporters_file.py
import jsonschema, json
from pathlib import Path
SCHEMA = json.loads(Path(\"schemas/trace-v1.json\").read_text())

def test_trace_json_validates_against_schema(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"schema-test\"):
        s = t.start_span(\"s\"); s.end()
    trace_file = next(tmp_path.glob(\"*.json\"))
    data = json.loads(trace_file.read_text())
    jsonschema.validate(data, SCHEMA)  # raises if invalid
\`\`\`

## Estimated effort

1 hour. Requires reading existing output shape and authoring the schema."

gh issue create \
  --title "Add SpanStatus.CANCELLED for timeout/abort scenarios" \
  --label "good first issue" \
  --body "## What to build

Add a \`CANCELLED\` value to the \`SpanStatus\` enum for spans that were aborted (timeout, user cancellation, graceful shutdown). It should be serializable to/from JSON and displayed distinctly in the stdout exporter.

## Where the code is

\`src/agent_trace/core/span.py\` — \`class SpanStatus(str, Enum)\`. Add \`CANCELLED = \"cancelled\"\`.

Also update:
- \`src/agent_trace/exporters/stdout.py\` — add a display symbol for CANCELLED (e.g. ⊘)
- \`src/agent_trace/core/span.py:to_dict()\` / \`from_dict()\` — already handled by enum value if you use \`SpanStatus(data[\"status\"])\`

## What the test should check

\`\`\`python
# tests/unit/test_core_span.py
def test_span_cancelled_status_serializes(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"c\"):
        s = t.start_span(\"s\")
        s.end(SpanStatus.CANCELLED)
    d = s.to_dict()
    assert d[\"status\"] == \"cancelled\"
    s2 = Span.from_dict(d)
    assert s2.status == SpanStatus.CANCELLED
\`\`\`

## Estimated effort

20 minutes. Enum addition + stdout exporter update."

gh issue create \
  --title "Write example: record+replay with the requests library" \
  --label "good first issue" \
  --body "## What to build

Create \`examples/04-requests-record-replay/\` showing how to record an agent run that uses the \`requests\` library (not httpx) and replay it offline.

Most users' existing code uses \`requests\`, not \`httpx\`. This example closes the gap.

## Where the code is

- \`src/agent_trace/interceptor/requests_patch.py\` — the \`requests\` adapter (read this first)
- \`examples/01-basic-trace/\` — use this as a structural template

## What the example should contain

1. \`examples/04-requests-record-replay/README.md\` — 3-step quickstart
2. \`examples/04-requests-record-replay/record.py\` — makes a real HTTP call via \`requests\`, records to fixture
3. \`examples/04-requests-record-replay/replay.py\` — replays from fixture, asserts no network call
4. \`examples/04-requests-record-replay/requirements.txt\`

## What the test should check

\`\`\`bash
# examples/04-requests-record-replay/test_example.sh
cd examples/04-requests-record-replay
pip install -r requirements.txt
python record.py   # should produce fixture.db
AGENT_TRACE_NETWORK_GUARD=1 python replay.py  # must not raise
\`\`\`

## Estimated effort

1 hour. Mostly writing example code and a short README."

gh issue create \
  --title "Add OTLP HTTP exporter (currently only gRPC supported)" \
  --label "good first issue" \
  --body "## What to build

Add \`OtlpHttpExporter\` to \`src/agent_trace/exporters/otlp.py\` that sends spans via HTTP/JSON to an OTLP-compatible collector (Jaeger, Grafana Tempo, etc.).

The gRPC exporter requires protobuf compilation which breaks in some environments. HTTP+JSON is dependency-free for most setups.

## Where the code is

\`src/agent_trace/exporters/otlp.py\` — read the existing \`OtlpExporter\` (gRPC). Model the new class after it but use \`httpx.post\` with the OTLP/JSON payload format.

OTLP/JSON spec: https://opentelemetry.io/docs/specs/otlp/#otlphttp

## What the test should check

\`\`\`python
# tests/unit/test_exporters_otlp.py
from unittest.mock import patch, MagicMock

def test_otlp_http_exporter_posts_json(tmp_path):
    from agent_trace.exporters.otlp import OtlpHttpExporter
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"http\") as trace:
        s = t.start_span(\"s\"); s.end()
    with patch(\"httpx.post\") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        OtlpHttpExporter(endpoint=\"http://localhost:4318\").export(trace)
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert \"resourceSpans\" in kwargs.get(\"json\", {})
\`\`\`

## Estimated effort

2 hours. Requires reading the OTLP JSON spec and mapping agent-trace's Span model to it."

gh issue create \
  --title "Add Tracer.plugin_count property" \
  --label "good first issue" \
  --body "## What to build

Add a \`plugin_count\` property to \`Tracer\` that returns the number of currently registered plugins. Useful for assertions in tests and for debugging plugin registration issues.

## Where the code is

\`src/agent_trace/__init__.py\` — add the property after \`clear_plugins()\` (or wherever \`_plugins\` is managed).

## What the test should check

\`\`\`python
# tests/unit/test_plugins.py
def test_plugin_count_increments(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    assert t.plugin_count == 0
    t.add_plugin(RecordingPlugin())
    assert t.plugin_count == 1
    t.add_plugin(RecordingPlugin())
    assert t.plugin_count == 2

def test_plugin_count_decrements_on_remove(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    p = RecordingPlugin()
    t.add_plugin(p)
    t.remove_plugin(p)
    assert t.plugin_count == 0
\`\`\`

## Estimated effort

10 minutes. \`return len(self._plugins)\`."

gh issue create \
  --title "Add Tracer.context_depth() method" \
  --label "good first issue" \
  --body "## What to build

Add \`Tracer.context_depth()\` that returns the nesting depth of the current context: \`0\` outside any trace, \`1\` inside a trace but no span, \`2\` inside a span, etc. Useful for debugging instrumentation setup.

## Where the code is

\`src/agent_trace/__init__.py\` — inspect the current ContextVar values for trace and span context.

## What the test should check

\`\`\`python
# tests/unit/test_tracer.py
def test_context_depth_outside(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    assert t.context_depth() == 0

def test_context_depth_in_trace(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"d\"):
        assert t.context_depth() == 1

def test_context_depth_in_span(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"d\"):
        t.start_span(\"s\")
        assert t.context_depth() == 2
\`\`\`

## Estimated effort

30 minutes. Requires understanding how ContextVar nesting works in the codebase."

gh issue create \
  --title "Write example: multi-tool agent with failure injection and replay debugging" \
  --label "good first issue" \
  --body "## What to build

Create \`examples/05-failure-injection/\` showing how to record a multi-tool agent run, inject a failure into the fixture, and replay it to observe how the agent handles errors.

This demonstrates the core value proposition: reproduce failures offline without re-running the LLM.

## Where the code is

- \`examples/02-langgraph-failure-replay/\` — use as structural template
- \`src/agent_trace/_replay/fixture.py\` — understand how to mutate fixtures

## What the example should contain

1. \`examples/05-failure-injection/agent.py\` — a mock 3-tool agent (no real LLM needed; use fixture)
2. \`examples/05-failure-injection/inject.py\` — script that modifies fixture to simulate a tool returning an error
3. \`examples/05-failure-injection/replay.py\` — replays the modified fixture and prints ERROR spans
4. \`examples/05-failure-injection/README.md\` — explains the workflow in 4 steps

## What the test should check

The README should include a \`# Test\` section showing the expected terminal output after running replay.py (ERR span should appear in output).

## Estimated effort

1.5–2 hours. Most effort is in the mock agent and clear README writing."

gh issue create \
  --title "Add Span.set_tag(key, value) alias for set_attribute" \
  --label "good first issue" \
  --body "## What to build

Add \`Span.set_tag(key, value)\` as a convenience alias for \`Span.set_attribute(key, value)\`. Many developers coming from OpenTracing / Datadog APM use \`set_tag\` terminology.

## Where the code is

\`src/agent_trace/core/span.py\` — \`Span.set_attribute()\` is at line ~111. Add \`set_tag = set_attribute\` as a class-level alias after the method, or define a thin wrapper.

For mypy \`--strict\` compliance, use a thin wrapper rather than a direct alias:

\`\`\`python
def set_tag(self, key: str, value: _AttrValue) -> None:
    self.set_attribute(key, value)
\`\`\`

## What the test should check

\`\`\`python
# tests/unit/test_core_span.py
def test_set_tag_is_set_attribute_alias(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"tag\"):
        s = t.start_span(\"s\")
        s.set_tag(\"model\", \"gpt-4o\")
        s.set_attribute(\"tokens\", 42)
        s.end()
    assert s.attributes[\"model\"] == \"gpt-4o\"
    assert s.attributes[\"tokens\"] == 42
\`\`\`

## Estimated effort

15 minutes. One-liner addition + one test."

gh issue create \
  --title "Add trace.metadata[\"duration_ms\"] to saved trace.json" \
  --label "good first issue" \
  --body "## What to build

When the file exporter writes \`trace.json\`, it should include \`duration_ms\` at the top level of the JSON so downstream tools (dashboards, scripts) don't have to compute it from span timestamps.

## Where the code is

\`src/agent_trace/core/trace.py:to_dict()\` — add \`\"duration_ms\": self.duration_ms\` to the returned dict. (This depends on issue #7 \`Trace.duration_ms\` being implemented first, or you can compute it inline.)

## What the test should check

\`\`\`python
# tests/unit/test_exporters_file.py
import json
def test_trace_json_includes_duration_ms(tmp_path):
    t = Tracer(trace_dir=tmp_path)
    with t.start_trace(\"dur\"):
        s = t.start_span(\"s\"); s.end()
    trace_file = next(tmp_path.glob(\"*.json\"))
    data = json.loads(trace_file.read_text())
    assert \"duration_ms\" in data
    assert isinstance(data[\"duration_ms\"], (float, int, type(None)))
\`\`\`

## Estimated effort

20 minutes. Depends on issue #7 or compute inline."

gh issue create \
  --title "Add write example: OpenAI Agents SDK record+replay with tool calls" \
  --label "good first issue" \
  --body "## What to build

Expand \`examples/03-ci-pipeline/\` or create \`examples/06-openai-agents-sdk/\` showing a real-world OpenAI Agents SDK workflow that records tool calls to a fixture and replays them in CI without spending tokens.

## Where the code is

- \`src/agent_trace/integrations/openai_agents.py\` — the OpenAI Agents integration (read first)
- \`examples/02-langgraph-failure-replay/\` — structural template

## What the example should contain

1. \`agent.py\` — a 2-tool agent using the OpenAI Agents SDK (use mock responses so no real API key needed in CI)
2. \`record.py\` — runs once with real API to produce fixture
3. \`replay.py\` — runs in CI with \`AGENT_TRACE_NETWORK_GUARD=1\`
4. \`README.md\` — explains the workflow; includes a note that record.py requires \`OPENAI_API_KEY\`

## What the test should check

\`\`\`bash
AGENT_TRACE_NETWORK_GUARD=1 python replay.py
# should exit 0 and print at least 2 tool-call spans
\`\`\`

## Estimated effort

1.5 hours. Primary effort is in writing the realistic example agent."

echo "Done. 19 issues created."
