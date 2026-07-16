# Example 04 — Record & replay an AWS Bedrock call (boto3)

Demonstrates the botocore/boto3 HTTP interceptor: agent-trace captures AWS
SDK traffic (Bedrock, SageMaker, or any other boto3-based service) the same
way it already captures `httpx`/`requests` traffic — no extra client wiring,
just wrap the call in `tracer.start_trace(record=True)`.

## Why this matters

boto3-based LLM providers — such as crewAI's native AWS Bedrock completion
provider (`crewai.llms.providers.bedrock.completion`), which constructs its
client via `boto3.session.Session(...)` directly — never touch
`httpx.Client` or `requests.Session`. Before this interceptor, that traffic
was invisible to agent-trace at the HTTP layer no matter which framework
integration was in play.

## What it demonstrates

- `tracer.start_trace(name, record=True)` capturing a real boto3 client call
  with zero extra configuration
- `agent_trace.replay(run_id)` serving that call back offline, at zero AWS
  cost, with the exact recorded response
- That `response["body"]` (a `botocore.response.StreamingBody`) is still
  correctly readable after recording — the interceptor eagerly drains the
  body to persist it, then hands the caller a fresh in-memory stream so
  `.read()` keeps working transparently

## How to run

This example needs no AWS account and makes no real AWS calls: it stands up
a local loopback HTTP server that mimics the shape of the Bedrock Runtime
`InvokeModel` API and points a real boto3 client at it via `endpoint_url`.
Everything else — request signing, response parsing, `StreamingBody`
handling — is the real botocore code path; only the network endpoint is
fake.

```bash
# From the repo root, with the botocore extra installed:
pip install -e ".[botocore,dev]"

# Record a call:
uv run python examples/04-bedrock-record-replay/example.py record

# Replay it offline (no network calls at all):
uv run python examples/04-bedrock-record-replay/example.py replay
```

## What the output looks like

```
Recording a Bedrock InvokeModel call via boto3...
Model said: (fake model) You said: 'What is agent-trace?'

Captured 1 AWS SDK exchange to: ~/.agent-trace/runs/example-04-bedrock/fixture.db
Replay it offline any time with:
    uv run python examples/04-bedrock-record-replay/example.py replay
```

```
Replaying the Bedrock call from fixture.db — zero network calls...
Model said (replayed, offline, zero cost): (fake model) You said: 'What is agent-trace?'
```

## Known limitation

For AWS event-stream operations (`InvokeModelWithResponseStream`,
`ConverseStream`), the interceptor still eagerly drains the full stream
before returning, the same non-buffering tradeoff `httpx_hook.py` makes for
SSE responses today. True incremental pass-through capture for streaming
responses is a known follow-up, not yet implemented.
