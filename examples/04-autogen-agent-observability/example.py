"""
AutoGen (autogen-agentchat/autogen-ext v0.4+/v0.7.x) observability example.

Demonstrates agent_trace.integrations.autogen end-to-end with zero API key
and zero network I/O:

- instrument_agent() -- agent-attributed spans, tool-call events, token
  usage, and exception-to-span attribution for an AssistantAgent tool-call
  turn (using autogen-ext's own ReplayChatCompletionClient test double, so
  no real LLM call is made).
- instrument_code_executor() -- exit-code/stdout+stderr capture for a real
  local Python subprocess run through LocalCommandLineCodeExecutor.

Prerequisites:
    pip install agent-trace[autogen]

Run:
    python examples/04-autogen-agent-observability/example.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken
    from autogen_core.code_executor import CodeBlock
    from autogen_core.models import ModelInfo
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    from autogen_ext.models.replay import ReplayChatCompletionClient
except ImportError:
    sys.exit(
        "autogen-agentchat / autogen-ext are not installed.\n"
        "Run: pip install agent-trace[autogen]"
    )

from agent_trace import tracer
from agent_trace.core.trace import Trace
from agent_trace.exporters.stdout import StdoutExporter
from agent_trace.integrations.autogen import instrument_agent, instrument_code_executor


def search_docs(query: str) -> str:
    """Search internal documentation for *query*."""
    return f"3 results found for {query!r}"


async def run_agent_with_tool_call() -> str:
    """Run an AssistantAgent through a tool-call turn with zero API cost.

    ReplayChatCompletionClient is autogen-ext's own test double: it returns
    pre-recorded CreateResult objects instead of calling a real model, so
    the whole turn -- including the tool-call loop -- runs with zero
    network I/O and zero API key required.
    """
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, RequestUsage

    responses = [
        CreateResult(
            finish_reason="function_calls",
            content=[
                FunctionCall(
                    id="call_1",
                    arguments='{"query": "refund policy"}',
                    name="search_docs",
                )
            ],
            usage=RequestUsage(prompt_tokens=42, completion_tokens=12),
            cached=False,
        ),
        CreateResult(
            finish_reason="stop",
            content="I found 3 documents about the refund policy.",
            usage=RequestUsage(prompt_tokens=58, completion_tokens=10),
            cached=False,
        ),
    ]
    model_client = ReplayChatCompletionClient(
        responses,
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            family="unknown",
            structured_output=False,
        ),
    )

    with tracer.start_trace("autogen-example", run_id="autogen-example-run") as trace:
        agent = AssistantAgent(
            "support_agent",
            model_client=model_client,
            tools=[search_docs],
            reflect_on_tool_use=True,
        )
        instrument_agent(agent, tracer=tracer, trace=trace)

        response = await agent.on_messages(
            [TextMessage(content="What is the refund policy?", source="user")],
            CancellationToken(),
        )

        executor_dir = tempfile.mkdtemp(prefix="agent-trace-autogen-")
        executor = LocalCommandLineCodeExecutor(work_dir=executor_dir)
        instrument_code_executor(executor, tracer=tracer, trace=trace)
        await executor.execute_code_blocks(
            [
                CodeBlock(
                    code="print('3 refund-policy docs indexed')", language="python"
                )
            ],
            CancellationToken(),
        )

    print_span_tree(trace.run_id)
    return str(response.chat_message.content)


def print_span_tree(run_id: str) -> None:
    trace_path = Path.home() / ".agent-trace" / "runs" / run_id / "trace.json"
    if not trace_path.exists():
        print(f"  (no trace.json found at {trace_path})")
        return
    trace_obj = Trace.from_dict(json.loads(trace_path.read_text()))
    StdoutExporter().export(trace_obj)


def main() -> None:
    print(
        "Running an AssistantAgent tool-call turn (zero API cost via "
        "ReplayChatCompletionClient) plus a real local code execution...\n"
    )
    result = asyncio.run(run_agent_with_tool_call())
    print(f"\nFinal agent response: {result}")


if __name__ == "__main__":
    main()
