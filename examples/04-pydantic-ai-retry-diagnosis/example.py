"""
pydantic-ai retry-attempt diagnosis example.

Demonstrates the pydantic-ai framework integration
(``agent_trace.integrations.pydantic_ai``): every model request and tool
call in an ``Agent`` run is captured as its own span, and — unlike the
generic httpx interceptor, which only sees anonymous raw HTTP rows — each
span is tagged with *why* it happened: a fresh turn, a retry forced by a
failing ``@agent.output_validator`` (mirrors real-world issues like
pydantic-ai#5508's ~1/3 intermittent-failure retries and #4919's duplicated
retry payloads), or a retry forced by a tool implementation raising
``ModelRetry``.

Runs entirely offline by default (``pydantic_ai.models.test.TestModel`` — no
API key, no network) so it always produces the same span tree. Pass
``--model openai:gpt-4o-mini`` (with ``OPENAI_API_KEY`` set) to run the same
scenario against a real provider — agent-trace's existing httpx interceptor
records/replays that traffic exactly as it does for the other examples.

Prerequisites:
    pip install "agent-observability-trace-cli[pydantic-ai]"

Run:
    python examples/04-pydantic-ai-retry-diagnosis/example.py
    python examples/04-pydantic-ai-retry-diagnosis/example.py --model openai:gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys

try:
    from pydantic_ai import Agent, ModelRetry
except ImportError:
    sys.exit(
        'pydantic-ai is not installed.\nRun: pip install "agent-observability-trace-cli[pydantic-ai]"'
    )

from agent_trace import Tracer
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.pydantic_ai import run_traced

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

# A validator that rejects the model's first answer, forcing pydantic-ai to
# retry the model call with a RetryPromptPart explaining what was wrong —
# exactly the "was this exchange a retry, and of what" question a developer
# cannot answer from raw HTTP bodies alone.
_validator_attempts = {"count": 0}


def _reset_counters() -> None:
    _validator_attempts["count"] = 0
    _tool_attempts["count"] = 0


_tool_attempts = {"count": 0}


def build_agent(model: str) -> Agent[None, int]:
    agent: Agent[None, int] = Agent(
        model, output_type=int, name="retry-demo-agent", retries=3
    )

    @agent.tool_plain
    def flaky_lookup(query: str) -> str:
        """A tool that fails once with ModelRetry before succeeding."""
        _tool_attempts["count"] += 1
        if _tool_attempts["count"] < 2:
            raise ModelRetry(f"lookup for {query!r} timed out, try again")
        return "42"

    @agent.output_validator
    def must_pass_on_second_look(output: int) -> int:
        """Reject the first output unconditionally to force one retry —
        stands in for a real validation rule (schema/business-logic check)
        that occasionally rejects a model's first answer.
        """
        _validator_attempts["count"] += 1
        if _validator_attempts["count"] < 2:
            raise ModelRetry("output failed validation, try again")
        return output

    return agent


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------


async def main_async(model: str) -> None:
    _reset_counters()
    agent = build_agent(model)

    t = Tracer()
    with t.start_trace("pydantic-ai-retry-diagnosis", record=True) as trace:
        try:
            result = await run_traced(
                agent,
                "Look up the answer using flaky_lookup, then return it as an integer.",
                tracer=t,
                trace=trace,
            )
            print(f"Result: {result.output}\n")
        except Exception as exc:
            print(f"Run failed with: {type(exc).__name__}: {exc}\n")

    print(f"Run ID: {trace.run_id}\n")
    print("Span tree (note llm.is_retry / llm.retry_tool_name / tool.retried):\n")
    StdoutExporter().export(trace)

    retry_spans = [s for s in trace.spans if s.attributes.get("llm.is_retry")]
    retried_tools = [s for s in trace.spans if s.attributes.get("tool.retried")]
    print(f"\n{len(retry_spans)} model-call span(s) tagged as retries.")
    print(f"{len(retried_tools)} tool span(s) tagged as retried (ModelRetry).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="test",
        help=(
            "pydantic-ai model identifier. Defaults to 'test' "
            "(TestModel — offline, deterministic, no API key needed)."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.model))


if __name__ == "__main__":
    main()
