# Why we built agent-trace

*Pin this as the first GitHub Discussion under "Announcements".*

---

We kept hitting the same wall debugging LLM agents.

An agent run fails. You look at the logs. The logs show a tool call returned something unexpected. You re-run the agent to reproduce it — and the LLM generates a different tool selection. The failure is gone. Ten minutes later it's back but in a different node. You can't pin it down because the LLM is nondeterministic and every re-run costs tokens and takes 20–90 seconds.

The standard advice is "add more logging." We did. It didn't solve the core problem: **there was no way to rewind to the exact moment of failure and step through it offline.**

## What we built

agent-trace gives you deterministic record/replay for LLM agents.

During a record pass, every HTTP request your agent makes — every LLM call, every tool API call — is captured to a local SQLite fixture file. During a replay pass, the HTTP transport is swapped for a cache reader. No network calls. No token spend. No nondeterminism. You get the exact same inputs to every node in your agent graph, with zero LLM involvement.

This means:
- **Reproduce any failure offline**, from a single `.db` file checked into your repo
- **Run agent tests in CI** without an API key or hitting rate limits
- **Inject failures** by editing the fixture — make a tool return an error, see how your agent handles it
- **Benchmark your agent graph** independently of API latency

## What "deterministic" means (and doesn't)

Same fixture → same inputs to each agent node + same tool responses. That's it. We don't intercept the LLM's token sampling. We skip the LLM entirely during replay by serving the recorded response from cache. If you need to test a different LLM response, edit the fixture.

## The stack

- Transport interception: `httpx` + `requests` adapter (works with OpenAI, Anthropic, any HTTP-based LLM SDK)
- Storage: SQLite fixture (one file per trace, portable, inspectable with any SQLite viewer)
- Integrations: LangGraph and OpenAI Agents SDK today; more on the roadmap
- Export: OTLP, file, stdout
- License: Apache 2.0 core

## How to contribute

If you've ever lost 45 minutes to a nondeterministic agent failure, you understand the problem. We built this to solve it for ourselves, and we think it solves it for you too.

Good first issues are labeled [`good first issue`](../../issues?q=is%3Aopen+label%3A%22good+first+issue%22) — each one includes exactly where the code is, what the test should check, and an honest effort estimate.

If you find a bug, open an issue. If you have a question, start a Discussion. If you have a use case we haven't thought of, we want to hear it.

---

*Built by Rudrendu Paul. Apache 2.0.*
