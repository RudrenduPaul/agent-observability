# Changelog

All notable changes to agent-trace are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.6] - 2026-07-20

### Added
- `--json` flag on `list`, `inspect`, `diff`, and `run` (`show` already
  returned JSON by default). Closes the gap between what the agent-native
  positioning claimed and what the CLI actually did — previously only `show`
  returned machine-parseable output. `run --json` prints agent-trace's own
  status as one final JSON line on stdout with the child process's own
  output passed through untouched ahead of it (status lines move to stderr
  in this mode, since the child's own output can't itself be made
  structured).

### Security
- Enabling Dependabot security-advisory alerts (previously disabled) surfaced
  5 real vulnerabilities: 3 in `mcp` (high), 1 in `json-repair` (high, via the
  `crewai` extra), 1 in `chromadb` (critical, pre-auth code injection, also
  via `crewai`). Bumped `mcp` to `>=1.28.1` (patched) and `crewai` to its
  latest release — this patches all 3 `mcp` alerts. `json-repair` and
  `chromadb` remain vulnerable: both are transitive pins inside `crewai`
  itself (confirmed by attempting to force a patched `json-repair` version,
  which `uv` reports as unsatisfiable against `crewai`'s own requirements),
  and `chromadb`'s advisory currently has no upstream patched version at
  all. The critical chromadb CVE requires an exposed ChromaDB *server* with
  `trust_remote_code=true` — this package never runs a ChromaDB server, only
  pulls it in transitively, which lowers real-world exploitability but
  doesn't close the alert. Tracked as upstream-blocked, not fixed.

## [0.1.5] - 2026-07-20

### Fixed
- `agent-trace version` printed `0.1.0` regardless of the actually-installed
  version (0.1.4 at the time), from two independently hardcoded version
  constants (`_cli.py`'s `_VERSION`, `__init__.py`'s `__version__`). Both now
  resolve via `importlib.metadata.version()` against the installed package,
  so they can't drift from `pyproject.toml` again.
- 39 more broken `pip install agent-trace` / `uv add agent-trace` commands
  across 27 `.py` files (`examples/*/example.py`, `demos/`, `tests/`,
  `src/agent_trace/integrations/langchain_core.py`) and 5
  `examples/*/README.md` files using `uv add` or a quoted
  `"agent-trace[extra]"` form the prior `.md`-only sweep's grep pattern
  didn't match.
- 3 relative image links in README.md (`docs/assets/...`, `docs/demo.gif`,
  `docs/usage.gif`) — broken on the live PyPI project page, which (unlike
  GitHub) does not rewrite relative markdown paths. Now absolute
  raw.githubusercontent.com URLs.
- `release.yml`'s Trusted Publishing setup comment pointed at the wrong,
  nonexistent PyPI project slug (`agent-trace` instead of
  `agent-observability-trace-cli`).
- README's Security section claimed SLSA Level 2 provenance, Sigstore
  signing on every release, SBOM attached to every release, and secret
  scanning auto-enabling on going public. None of these are currently true
  (verified via `gh release view`, the GitHub API's `security_and_analysis`
  endpoint, and a repo-wide grep for any provenance-attestation mechanism).
  Rewritten to state the actual current state.
- `docs/concepts.md` described an httpx patch mechanism the source's own
  comments say was deliberately replaced, claimed `httpx.AsyncClient` isn't
  intercepted (it is), and documented a 2-table fixture schema when the real
  one has 4 (`ws_frames`/`mcp_frames` were undocumented). Rewritten against
  current source.

## [0.1.4] - 2026-07-20

### Changed
- `Development Status` classifier bumped from Alpha to Beta.
- README rewritten for launch: pain-first hero, benchmark numbers moved above
  the fold, a verified 30-second CLI quickstart, a CI cost calculator, a
  dedicated "why not LangSmith" comparison section, a real-failures list, and
  a supported-frameworks callout near install.
- CI now passes on Ubuntu, macOS, and Windows via a new
  `cross-platform-tests` job (previously Ubuntu-only).

### Added
- `examples/03-ci-pipeline/fixture.db`: a real fixture recorded against
  httpbin.org, committed so contributors can run the CI-replay example
  without any API keys.

### Fixed
- `examples/03-ci-pipeline/example.py` and `test_with_fixture.py` read
  `fixture.exchange_count()` after the `replay()` context had already closed
  the database connection, raising `sqlite3.ProgrammingError` on every run.
- Every `pip install`/`uv add` command in `docs/getting-started.md`,
  `docs/integrations/`, and all 20 `examples/*/README.md` files referenced
  `agent-trace` as the distribution name, which 404s on PyPI (`agent-trace`
  is only the CLI command / import module name). All now install the real
  package, `agent-observability-trace-cli`.
- `npm/README.md` linked to the comparison section's old anchor
  (`#how-agent-observability-compares`), broken since the README rename
  above, and still claimed CI ran on Ubuntu only.
- `release.yml`'s SBOM step used an outdated `cyclonedx-py` CLI flag syntax
  (`--format`/`--output-file` instead of `--of`/`-o`), silently failing on
  every prior tagged release before the signing/GitHub-release steps ran.

## [0.1.2] - 2026-07-16

### Added
- gRPC transport interception (`pip install agent-trace[grpc]`) for LLM SDKs
  that default to gRPC instead of REST (e.g. Vertex AI's mTLS-authenticated
  path). Records/replays unary-unary and unary-stream gRPC calls the same
  way `RecordingTransport`/`ReplayTransport` do for httpx; wired into
  `Tracer._install_recording_transport`/`_uninstall_recording_transport` and
  the replay engine alongside the existing httpx/requests patches. Covers
  both sync `grpc` and async `grpc.aio` (unary-unary only for aio).
- HTTP transport interception for `aiohttp.ClientSession` (`pip install agent-trace[aiohttp]`), closing a silent recording gap for LLM traffic routed through aiohttp-based clients (e.g. LiteLLM's default async transport)

### Fixed
- PyPI distribution renamed to `agent-observability-trace` to match the name actually registered on PyPI (import path is unchanged: `import agent_trace`)
- `Author` field on PyPI linked to Rudrendu's personal email under Sourav's displayed name; authors are now name-only with GitHub profile links in `project.urls`

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
