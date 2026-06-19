# Example 03 — CI Pipeline

This example shows the recommended pattern for using agent-trace in CI:
record once, commit the fixture, replay in every test run.

## The pattern

```
Developer laptop           CI (GitHub Actions / CircleCI / etc.)
─────────────────          ──────────────────────────────────────
python example.py record   pytest examples/03-ci-pipeline/ -v
  → writes fixture.db
  → commit fixture.db
                           ↑ reads fixture.db from repo
                           ↑ no API calls, no API key needed
```

## Files in this example

| File | Purpose |
|------|---------|
| `example.py` | The agent + record/test CLI |
| `test_with_fixture.py` | pytest tests that run against the fixture |
| `fixture.db` | Recorded HTTP exchanges (committed to the repo after recording) |

## Step 1 — Record (run once on your laptop)

```bash
pip install agent-trace httpx

# Record a real run. This calls httpbin.org (no API key needed for this demo).
python examples/03-ci-pipeline/example.py record
```

Output:

```
Recording to: examples/03-ci-pipeline/fixture.db
Document: agent-trace is an observability tool for AI agents...

Stored 1 HTTP exchange(s) in fixture

Result: {'classification': 'long', 'confidence': 0.356, 'api_response_size': 412}

Fixture written to: examples/03-ci-pipeline/fixture.db
Commit this file to your repo so CI can use it.

Run the test with:
  pytest examples/03-ci-pipeline/ -v
```

## Step 2 — Commit the fixture

```bash
git add examples/03-ci-pipeline/fixture.db
git commit -m "Add CI fixture for document classifier"
```

The fixture contains full HTTP request and response bodies. Do not commit
fixtures recorded against production API keys. Use a test key, or scrub the
fixture after recording (see `docs/concepts.md` for the SQLite schema).

## Step 3 — Run tests in CI (and locally)

```bash
pytest examples/03-ci-pipeline/ -v
```

Output:

```
examples/03-ci-pipeline/test_with_fixture.py::test_agent_responds_correctly PASSED
examples/03-ci-pipeline/test_with_fixture.py::test_replay_is_fast PASSED
examples/03-ci-pipeline/test_with_fixture.py::test_fixture_exchange_count_matches_expectation PASSED
```

All three tests pass without any network access. Set `AGENT_TRACE_NETWORK_GUARD=1`
(already configured in `pyproject.toml`'s `[tool.pytest.ini_options]`) to ensure
any accidental live call fails loudly rather than silently.

## GitHub Actions snippet

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install agent-trace httpx pytest

      - name: Run tests
        env:
          AGENT_TRACE_NETWORK_GUARD: "1"
        run: pytest examples/03-ci-pipeline/ -v
```

No `OPENAI_API_KEY` secret needed in CI — the fixture handles all HTTP responses.

## When to re-record

Re-record the fixture when:

- The agent's prompt or logic changes in a way that would produce a different
  API request (different URL, different request body structure).
- The API response schema changes and your agent code depends on the new fields.
- You want to test a different code path (add a new fixture for each scenario).

Do not re-record just to get a "fresher" LLM response — the value of the
fixture is that it pins the exact response, making tests deterministic.
