# Contributing to agent-trace

Thank you for your interest in contributing. This guide covers everything you need to get started, from setting up a local dev environment to getting your PR merged quickly.

## Development Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) (install via `curl -Lf https://astral.sh/uv/install.sh | sh`)
- Git

### First-time setup

```bash
git clone https://github.com/RudrenduPaul/agent-observability.git
cd agent-observability
uv sync --extra dev
pre-commit install
```

### Verify your setup

```bash
uv run pytest tests/unit/ -q
```

All tests should pass before you make any changes. If they don't, open an issue.

---

## Engineering Standards

Every PR must pass the full quality gate before it can merge. Run these locally before pushing:

```bash
# Lint and format
uv run ruff check src/ tests/ benchmarks/
uv run ruff format --check src/ tests/ benchmarks/

# Type checking — zero errors required
uv run mypy src/ --strict

# Unit tests with coverage gate
uv run pytest tests/unit/ --cov=src/ --cov-fail-under=80 -v
```

### Style conventions

- **Formatter:** `ruff format` (88-char line length, no manual overrides)
- **Linter:** `ruff check` with the ruleset in `pyproject.toml` — do not add `# noqa` without a comment explaining why
- **Types:** All public functions must be fully typed. Internal helpers too, unless the type annotation would be genuinely unreadable
- **Docstrings:** Public API functions need a one-line summary. Private helpers do not require docstrings

### What makes a PR merge quickly

1. **One concern per PR.** A PR that fixes a bug and refactors three unrelated modules will be asked to split. Reviewers review the whole diff — keep it small and reviewable.
2. **Tests for every behavior change.** If you add a feature, add a unit test. If you fix a bug, add a regression test that would have caught it.
3. **CHANGELOG.md entry.** Every user-facing change goes under `## [Unreleased]`. Internal refactors and CI changes do not need an entry.
4. **Passing CI before requesting review.** The CI matrix runs on Python 3.10–3.13. Fix failures before pinging for review.
5. **Clear PR description.** Fill in the PR template. Reviewers should not have to ask "what does this do?" after reading your description.

---

## Integration Tests

Integration tests hit real external APIs (OpenAI, Anthropic, etc.) and are never mocked. They are gated behind the `@pytest.mark.integration` marker and excluded from the standard CI run.

**Rules:**
- Integration tests live in `tests/integration/`
- Every integration test must be decorated with `@pytest.mark.integration`
- Do not mock API calls in integration tests — that is what unit tests with `respx` are for
- Integration tests must be self-contained: they must set up and tear down any state they create
- Secrets are injected via environment variables, never hardcoded

**Running integration tests locally:**

```bash
export OPENAI_API_KEY=sk-...
uv run pytest tests/integration/ -m integration -v
```

**CI policy:** Integration tests do not run in CI by default. They run in a separate scheduled workflow against maintainer-controlled secrets. If your PR requires integration test coverage, mention it in the PR description.

---

## Benchmark PRs

If your change touches any of the following, it is a benchmark-sensitive PR and requires a benchmark comparison:

- `src/agent_trace/core/` — span/trace data model
- `src/agent_trace/transport/` — HTTP interceptors
- `src/agent_trace/replay/` — fixture store or replay engine
- Any change to serialization or deserialization paths

**How to run benchmarks and attach the output:**

```bash
# First run on main to establish baseline
git checkout main
uv run pytest benchmarks/ --benchmark-save=baseline

# Apply your changes, then compare
git checkout your-branch
uv run pytest benchmarks/ --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

Paste the comparison table into your PR description under "Benchmark impact". PRs that regress mean latency by more than 10% will not merge without a compelling justification.

---

## Response SLAs

We aim to respond within these windows on business days (Pacific Time):

| Type | Target |
|------|--------|
| Bug reports | 24 hours |
| Feature requests | 72 hours |
| Pull requests | 72 hours |

Response means "acknowledged, triaged, or reviewed" — not necessarily resolved. If your issue or PR is approaching these windows with no response, a single polite ping in the thread is welcome.

---

## AI-Assisted Contributions

AI tools (Copilot, Claude, GPT-4, etc.) are permitted and can speed up boilerplate-heavy work. However:

- **You are responsible for every line you submit.** "The AI wrote it" is not a defense for a security issue, incorrect logic, or a type error.
- Read and understand every generated line before submitting.
- Indicate in your PR description if the implementation is AI-assisted. This is for the reviewer's awareness, not a negative mark.
- Do not submit AI-generated tests that only test the happy path. Tests must cover edge cases and failure modes.

---

## Good First Issues

Looking for somewhere to start? Filter the issue tracker by these labels:

- [`good first issue`](https://github.com/RudrenduPaul/agent-observability/labels/good%20first%20issue) — small, well-scoped tasks with clear acceptance criteria
- [`docs`](https://github.com/RudrenduPaul/agent-observability/labels/docs) — documentation improvements, typo fixes, example additions
- [`help wanted`](https://github.com/RudrenduPaul/agent-observability/labels/help%20wanted) — medium-complexity issues where a contributor's perspective would be valuable

If you plan to work on an issue, leave a comment so we can assign it and avoid duplication.

---

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0. See `LICENSE` for details.
