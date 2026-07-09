# Changelog

All notable changes to agent-trace are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- gRPC transport interception (`pip install agent-trace[grpc]`) for LLM SDKs
  that default to gRPC instead of REST (e.g. Vertex AI's mTLS-authenticated
  path). Records/replays unary-unary and unary-stream gRPC calls the same
  way `RecordingTransport`/`ReplayTransport` do for httpx; wired into
  `Tracer._install_recording_transport`/`_uninstall_recording_transport` and
  the replay engine alongside the existing httpx/requests patches. Covers
  both sync `grpc` and async `grpc.aio` (unary-unary only for aio).

## [0.1.0] - 2026-06-19

### Added
- Core span and trace data model (`Span`, `Trace`, `SpanStatus`)
- Clock abstraction (`core/clock.py`) enabling deterministic replay
- HTTP transport interception for `httpx` and `requests`
- SQLite-backed fixture store (`replay/fixture.py`) for record/replay
- Replay engine with `AGENT_TRACE_NETWORK_GUARD` enforcement
- `@tracer.instrument(record=True)` decorator API
- `tracer.span()` context manager for manual span creation
- LangGraph callback handler integration (`pip install agent-trace[langgraph]`)
- OpenAI Agents SDK hook integration (`pip install agent-trace[openai-agents]`)
- OTLP exporter for Jaeger/Grafana Tempo (`pip install agent-trace[otlp]`)
- `stdout` and `file` exporters
- Three benchmark scripts: overhead, fidelity, ingestion
- CI pipeline with ruff, mypy --strict, pytest, Trivy, OpenSSF Scorecard
- Apache 2.0 license
