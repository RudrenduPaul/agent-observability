# Show HN Draft

**Best window:** Tuesday–Thursday, 9–11am EST (per launch checklist)

---

**Title:**

> Show HN: agent-trace – deterministic record/replay for LLM agents

---

**Body (≤ 250 words for the submission text):**

> I built agent-trace to solve a debugging problem that cost me hours every week: when an LLM agent fails, you can't reproduce the failure because the LLM is nondeterministic. Re-running the agent costs tokens, hits rate limits, and often produces a different output anyway.
>
> agent-trace works by intercepting HTTP at the transport layer. During a record pass, every LLM request/response pair is serialized to a SQLite fixture file. During a replay pass, the transport is swapped for a cache reader — no network calls, no token spend, no LLM involvement. The inputs to every node in your agent graph are identical to the original run.
>
> This gives you:
> - Offline reproduction of any failure from a single .db file
> - Agent tests in CI without an API key
> - Failure injection by editing the fixture
> - Performance benchmarks independent of API latency
>
> Works with LangGraph and OpenAI Agents SDK today. Transport interception covers both httpx and requests, so it works with the OpenAI Python SDK, Anthropic SDK, and most other LLM clients.
>
> Apache 2.0. 287 tests, 95% coverage.
>
> GitHub: https://github.com/RudrenduPaul/agent-trace

---

**Pre-launch checklist for HN day:**

- [ ] Post between 9–11am EST Tuesday–Thursday
- [ ] Have 4+ people post link in LangChain Discord #show-and-tell within 15 min of going live
- [ ] Respond to every HN comment within 2 hours
- [ ] If < 50 upvotes at 48h mark: pause and diagnose before next content push
- [ ] Cross-post to r/MachineLearning, r/LangChain after HN clears

---

**Talking points if asked in comments:**

- **vs LangSmith**: LangSmith is a hosted SaaS dashboard. agent-trace is local-first and open-source. Replay works offline; LangSmith doesn't support offline replay.
- **vs Langfuse**: Same — Langfuse is observability dashboards. agent-trace's unique angle is the deterministic replay engine, not dashboards.
- **vs mocking**: Mocking lies. Your mock and the real API diverge. agent-trace records the real response, so replay is the real response. This is the difference between a unit test that passes and a production failure nobody caught.
- **Forkability**: Yes, the Apache 2.0 core can be forked. The moat is in the hosted tier (trace history, team dashboards, SSO) and in the integration maintenance burden — we keep the LangGraph and OpenAI Agents SDK integrations green as those APIs change.
