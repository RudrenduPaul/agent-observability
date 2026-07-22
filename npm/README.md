# agent-observability-trace-cli (npm wrapper)

An npm-installable wrapper for [`agent-trace`](https://github.com/RudrenduPaul/agent-observability), the deterministic record/replay tool for LLM agents. This package does not reimplement `agent-trace`. It puts the real `agent-trace` command on your PATH through `npm install`/`npx`, then forwards every argument straight through to the actual Python CLI.

## What agent-trace does

`agent-trace` records every HTTP request and response your agent makes, verbatim, to a local SQLite fixture, then replays those exact bytes later with zero network calls. It patches `httpx.Client`, `httpx.AsyncClient`, and `requests.Session` at the transport layer, so it works with any Python HTTP client, including the clients used internally by the OpenAI Python SDK and the Anthropic SDK. The project's own benchmark suite (`benchmarks/test_overhead.py`, `benchmarks/test_replay_vs_live.py`, `benchmarks/test_fidelity.py` in the main repo) reports 0.011% recording overhead, 0.93ms mean replay latency, and byte-for-byte replay fidelity, at $0 API cost per replay.

## Why this wrapper exists

`agent-trace` is a Python tool. If your team, CI pipeline, or agent harness already reaches for `npx` to run CLIs, this wrapper skips the "how do I invoke a Python tool from a Node-first workflow" step. It is a thin exec-and-forward shim, nothing more: the actual record/replay engine, interceptors, and CLI logic all live in the `agent_trace` Python package.

## Prerequisites

- Python 3.10 or newer
- The [`agent-observability-trace-cli`](https://pypi.org/project/agent-observability-trace-cli/) PyPI package installed on the same machine (its console command is `agent-trace`):

```bash
pip install agent-observability-trace-cli
# or
uv add agent-observability-trace-cli
# or, for an isolated global install
pipx install agent-observability-trace-cli
```

Without that package installed, this wrapper prints install instructions and exits non-zero. It never installs anything on your behalf.

## Install

```bash
npm install -g agent-observability-trace-cli
# or run without installing
npx agent-observability-trace-cli version
```

## Usage

Once installed, `agent-trace` on your PATH is this wrapper, and every subcommand is forwarded unchanged to the real CLI:

```bash
agent-trace version
agent-trace list
agent-trace show run_abc123def456
agent-trace show run_abc123def456 --errors-only
agent-trace replay run_abc123def456
agent-trace inspect run_abc123def456
agent-trace diff run_a run_b
agent-trace run -- langgraph dev
```

A recording/replay round trip in Python, using the underlying library directly:

```python
from agent_trace import tracer
import httpx

@tracer.instrument(record=True)
def fetch_data(query: str) -> dict:
    with tracer.span("http-call") as span:
        resp = httpx.get("https://httpbin.org/get", params={"q": query})
        span.set_attribute("http.status_code", resp.status_code)
        return resp.json()

result = fetch_data("hello")
# Trace and fixture saved to ~/.agent-trace/runs/run_<id>/
```

Full command reference — every subcommand's flags, defaults, and `--json` support — is in the [main repository's CLI reference](https://github.com/RudrenduPaul/agent-observability#cli-reference). The record/replay model and framework integrations (LangGraph, OpenAI Agents SDK, CrewAI, and more) are also documented in the [main repository](https://github.com/RudrenduPaul/agent-observability).

## How it works

The `agent-trace` bin script this package installs does two things, in order:

1. Looks for a real `agent-trace` executable on your PATH (the console script `pip`/`uv`/`pipx` installs) and execs it with your arguments.
2. If that isn't found, falls back to invoking the `agent_trace` Python module directly through `python3`/`python`.

If neither is available, it prints the install instructions above and exits with a non-zero status.

## How it compares

Most observability tools for LLM agents, including LangSmith, Langfuse, Helicone, and OpenLLMetry, are observe-only: they show you a trace of what happened, but reproducing a failure still means re-running the full agent against live APIs. `agent-trace` additionally lets you reproduce that exact run offline, deterministically, without touching the live API.

The closest built-in comparison is LangSmith's `LANGSMITH_TEST_CACHE` (VCR-style cassettes via `langsmith[vcr]`). It's Python plus LangChain only, captures HTTP calls to `api.openai.com` specifically rather than any HTTP client, doesn't record full wire-level bytes, and requires a LangSmith account. `agent-trace` works with any Python HTTP client, records full request and response bytes locally, and needs no account or hosted service.

The full capability table against LangSmith, Langfuse, Helicone, and OpenLLMetry lives in the [main repository's README](https://github.com/RudrenduPaul/agent-observability#why-not-just-use-langsmith-langfuse-or-helicone).

## Known limitations

This wrapper only forwards arguments; the limitations below belong to `agent-trace` itself (documented in full in the main repo's README):

- Recording and replay happen inside the Python process you import `agent_trace` into. It cannot observe or replay calls made by a third-party hosted service you don't run yourself, only your own process's outbound calls.
- gRPC coverage is partial: unary-unary and sync unary-stream calls (used by Gemini/Vertex AI) are captured and replayed, but client-streaming, bidirectional-streaming, and any `grpc.aio` streaming call are not.
- Capture starts once a fully-constructed HTTP request object reaches the interceptor. Exceptions raised earlier, while an SDK is still serializing a tool schema or building headers, produce zero fixture rows unless a wired-in framework integration's own error callback catches them first.

## FAQ

**Does this package reimplement `agent-trace` in JavaScript?**

No. It is a thin wrapper. The `bin/agent-trace.js` script it installs execs the real `agent-trace` console script if it finds one on your PATH, or falls back to invoking the `agent_trace` Python module through `python3`/`python`. All record/replay logic, HTTP interceptors, and CLI commands live in the Python package.

**Why would I install a Node wrapper for a Python tool?**

If your team already standardizes on `npx`/`npm` to run CLIs, whether in local scripts, CI steps, or agent tooling, this lets `agent-trace` slot into that same invocation pattern without a separate "activate a Python environment first" step. You still need the Python package installed; this wrapper does not remove that dependency.

**What do I need installed before this works?**

Python 3.10 or newer, plus the `agent-observability-trace-cli` PyPI package (`pip install agent-observability-trace-cli`, `uv add agent-observability-trace-cli`, or `pipx install agent-observability-trace-cli`). This npm package alone does nothing useful without it.

**What happens if I run `agent-trace` through this wrapper without the Python package installed?**

The wrapper tries the real console script first, then falls back to a direct Python module import. If both fail, it prints the exact `pip`/`uv`/`pipx` install commands above and exits with a non-zero status. It never installs anything automatically.

**Which commands does this wrapper support?**

All of them, unmodified. Every argument you pass to `agent-trace` through this wrapper is forwarded verbatim to the real CLI, so `agent-trace version`, `agent-trace list`, `agent-trace show <run-id>`, `agent-trace replay <run-id>`, `agent-trace inspect <run-id>`, `agent-trace diff <run-a> <run-b>`, and `agent-trace run -- <command>` all work exactly as documented in the main repository.

**How is this different from LangSmith's tracing/caching?**

LangSmith's `LANGSMITH_TEST_CACHE` needs LangChain and a LangSmith account, and only captures calls to `api.openai.com`. `agent-trace` works with any Python HTTP client (`httpx`, `requests`, and the transports used inside the OpenAI and Anthropic SDKs), records full request/response bytes locally, and needs no account. See the comparison table in the main repository for the full breakdown against LangSmith, Langfuse, Helicone, and OpenLLMetry.

**Does this wrapper work on Windows?**

The wrapper itself is plain Node.js and has no OS-specific code. The underlying `agent-trace` Python package's automated CI passes on Ubuntu, macOS, and Windows, documented in the main repository.

**Is it safe to commit recorded fixtures to version control?**

Not by default, and this is a property of `agent-trace` itself, not this wrapper. Fixture files can contain full HTTP request and response bodies, including API keys and prompt contents. See the main repository's README for guidance on `.gitignore` entries and redacting secrets before committing a fixture.

**Is this free to use commercially?**

Yes. Both this npm package and the underlying `agent-observability-trace-cli` PyPI package are Apache 2.0 licensed, which permits commercial use, modification, and redistribution, subject to the license's attribution and notice terms.

## License

Apache-2.0
