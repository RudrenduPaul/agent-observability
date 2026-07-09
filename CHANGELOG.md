# Changelog

All notable changes to agent-trace are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- HTTP transport interception for `aiohttp.ClientSession` (`pip install agent-trace[aiohttp]`), closing a silent recording gap for LLM traffic routed through aiohttp-based clients (e.g. LiteLLM's default async transport)

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
