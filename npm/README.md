# agent-trace-cli

An npm-installable wrapper for [`agent-trace`](https://github.com/RudrenduPaul/agent-observability), deterministic record/replay for LLM agents. This package does not reimplement the tool. It puts the real `agent-trace` command on your PATH through `npx`/`npm install`, then forwards straight through to the Python CLI.

## Why this exists

`agent-trace` is a Python tool. If your team already reaches for `npx` to run one-off CLIs, this wrapper lets you do that without a separate "how do I run this Python thing" step, as long as Python and the `agent-trace` package are available.

## Prerequisites

- Python 3.10+
- The `agent-observability-trace` package installed and on the same machine (its console command is `agent-trace`):

```bash
pip install agent-observability-trace
# or
uv add agent-observability-trace
# or, for an isolated global install
pipx install agent-observability-trace
```

## Install

```bash
npm install -g agent-trace-cli
# or run without installing
npx agent-trace-cli version
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

Full command reference, the record/replay model, and framework integrations (LangGraph, OpenAI Agents SDK, CrewAI, and more) are documented in the [main repository](https://github.com/RudrenduPaul/agent-observability).

## How it works

The `agent-trace` bin script this package installs does two things, in order:

1. Looks for a real `agent-trace` executable on your PATH (the console script `pip`/`uv`/`pipx` installs) and execs it with your arguments.
2. If that isn't found, falls back to invoking the `agent_trace` Python module directly through `python3`/`python`.

If neither is available, it prints the install instructions above and exits with a non-zero status. Nothing is installed automatically on your behalf.

## License

Apache-2.0
