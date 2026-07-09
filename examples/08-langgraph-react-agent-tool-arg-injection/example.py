"""
create_react_agent + post_model_hook + InjectedState/InjectedStore — #4841.

Reproduces the exact failure shape behind
https://github.com/langchain-ai/langgraph/issues/4841: a prebuilt
``create_react_agent`` graph configured with a ``post_model_hook`` routes
tool calls through LangGraph's own ``should_continue``/
``post_model_hook_router`` conditional-edge dispatch before
``ToolNode._inject_tool_args()`` resolves ``InjectedState``/``InjectedStore``
arguments. When a tool parameter *looks* like it should have come from graph
state (same name as a real state field) but was never annotated
``Annotated[..., InjectedState(...)]``, the LLM has to invent a value for it
at call time — exactly the "the LLM is populating the state field and
hallucinating the value" diagnosis a LangGraph maintainer made by hand in
that issue.

No API key required — the "model" is a plain Python stand-in that always
returns the same two parallel tool calls, deterministically.

What this example demonstrates about agent-trace's capture:

1. **`branch:dispatch` spans** for both of `create_react_agent`'s own
   internal routers — `should_continue` (whether to keep tool-calling or
   stop) and `post_model_hook_router` (LangGraph's own router inserted
   after a `post_model_hook`) — something LangGraph itself builds with
   `trace=False` and that was previously invisible to agent-trace on the
   success path entirely.
2. **`tool_inject:<name>` spans** showing exactly which argument names
   `ToolNode._inject_tool_args()` actually resolved for each tool call —
   `lookup_user_pref` shows `tool.injection_ran=True` with
   `user_id,store` injected from real graph state/the graph's own
   `BaseStore`; `hallucinated_lookup` shows `tool.injection_ran=False` —
   nothing was injected, because nothing was ever annotated to be.
3. **`find_tool_params_shaped_like_state()`** flags `hallucinated_lookup`'s
   `user_id` parameter automatically at tracer-construction time (recorded
   onto `trace.metadata["tool_state_shaped_params"]`) — the schema-level
   check that turns the maintainer's by-hand diagnosis into something
   agent-trace itself catches, before the agent is ever run.

Run:
    python examples/08-langgraph-react-agent-tool-arg-injection/example.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Annotated, Any

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.prebuilt import InjectedState, InjectedStore, create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.store.base import BaseStore
    from langgraph.store.memory import InMemoryStore
except ImportError:
    sys.exit("langgraph is not installed.\nRun: pip install agent-trace[langgraph]")

from agent_trace import Tracer
from agent_trace.integrations.langgraph import (
    LangGraphTracer,
    find_tool_params_shaped_like_state,
)

# create_react_agent emits a deprecation warning on current LangGraph
# versions (moved to langchain.agents) — irrelevant noise for this example;
# create_react_agent is still what issue #4841 itself uses.
warnings.filterwarnings("ignore", message=".*create_react_agent.*")

# ---------------------------------------------------------------------------
# Fake chat model — no API key, fully deterministic
# ---------------------------------------------------------------------------

_call_count = {"n": 0}


class FakeChatModel(BaseChatModel):
    """First turn: two parallel tool calls, one correctly using state that
    would be injected, one that only *looks* like it should be. Second
    turn (after both ToolMessages come back): a plain final answer, ending
    the should_continue loop."""

    @property
    def _llm_type(self) -> str:
        return "fake-react-agent-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        _call_count["n"] += 1
        if _call_count["n"] == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_user_pref",
                        "args": {"query": "theme"},
                        "id": "call_lookup_1",
                    },
                    {
                        "name": "hallucinated_lookup",
                        # The model has to invent user_id itself — nothing
                        # marks this parameter as coming from graph state.
                        "args": {"query": "theme", "user_id": "hallucinated-user-999"},
                        "id": "call_hallucinated_1",
                    },
                ],
            )
        else:
            msg = AIMessage(content="Your theme preference is dark-mode.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


# ---------------------------------------------------------------------------
# Tools — one correctly injected, one shaped like state but not marked
# ---------------------------------------------------------------------------


@tool
def lookup_user_pref(
    query: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Correctly injected: user_id comes from real graph state, store from
    the graph's own compiled BaseStore — the LLM never sees or invents
    either."""
    return f"pref for {user_id}: dark-mode (query={query})"


@tool
def hallucinated_lookup(query: str, user_id: str) -> str:
    """BUG (the #4841 shape): user_id shares its name with graph state's
    own 'user_id' field but was never annotated InjectedState — the LLM
    has to invent a value for it at call time."""
    return f"pref for {user_id} (hallucinated)"


def post_hook(state: dict[str, Any]) -> dict[str, Any]:
    """No-op post_model_hook — present purely to exercise
    create_react_agent's post_model_hook wiring (and the
    post_model_hook_router conditional edge LangGraph inserts because of
    it), the exact configuration #4841 uses."""
    return {}


class ReactAgentState(AgentState):
    user_id: str


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def build_agent() -> Any:
    store = InMemoryStore()
    return create_react_agent(
        model=FakeChatModel(),
        tools=[lookup_user_pref, hallucinated_lookup],
        post_model_hook=post_hook,
        state_schema=ReactAgentState,
        store=store,
    )


def main() -> None:
    agent = build_agent()

    # find_tool_params_shaped_like_state() also works standalone, before
    # any invocation — a static check of the graph's own wiring.
    findings = find_tool_params_shaped_like_state(agent)
    print("Schema-level check (find_tool_params_shaped_like_state):")
    for f in findings:
        print(f"  node={f['node']!r} tool={f['tool']!r} param={f['param']!r}")
    if not findings:
        print("  (none found)")

    t = Tracer(trace_dir=Path.home() / ".agent-trace" / "runs")
    with t.start_trace("langgraph-react-agent-tool-arg-injection") as trace:
        cb = LangGraphTracer(tracer=t, trace=trace, graph=agent)
        agent.invoke(
            {
                "messages": [HumanMessage(content="What's my theme preference?")],
                "user_id": "real-user-42",
            },
            config={"callbacks": [cb]},
        )

    print(f"\ntrace.metadata['tool_state_shaped_params']: "
          f"{trace.metadata.get('tool_state_shaped_params')}")

    print("\n--- branch:dispatch spans (routers exercised) ---")
    for s in trace.spans:
        if s.name == "branch:dispatch":
            print(f"  router={s.attributes.get('branch.router_name')!r}  "
                  f"status={s.status.value}")

    print("\n--- tool_inject:<name> spans (InjectedState/InjectedStore resolution) ---")
    for s in trace.spans:
        if s.name.startswith("tool_inject:"):
            print(
                f"  tool={s.attributes.get('tool.name')!r}  "
                f"injection_ran={s.attributes.get('tool.injection_ran')}  "
                f"injected_arg_keys={s.attributes.get('tool.injected_arg_keys', '')!r}"
            )

    print(f"\nRun ID: {trace.run_id}")


if __name__ == "__main__":
    main()
