# Example 08 — create_react_agent + post_model_hook + InjectedState/InjectedStore (#4841)

Reproduces the exact failure shape behind
[langgraph#4841](https://github.com/langchain-ai/langgraph/issues/4841): a
prebuilt `create_react_agent` graph configured with a `post_model_hook`
routes tool calls through LangGraph's own `should_continue`/
`post_model_hook_router` conditional-edge dispatch before
`ToolNode._inject_tool_args()` resolves `InjectedState`/`InjectedStore`
arguments. When a tool parameter *looks* like it should come from graph
state (same name as a real state field) but was never annotated
`Annotated[..., InjectedState(...)]`, the LLM has to invent a value for it —
exactly the "the LLM is populating the state field and hallucinating the
value" diagnosis a LangGraph maintainer made by hand in the related
[#3266](https://github.com/langchain-ai/langgraph/issues/3266) thread.

No API key required — the "model" is a plain Python stand-in that always
returns the same two parallel tool calls, deterministically.

## Prerequisites

```bash
pip install agent-observability-trace-cli[langgraph]
```

## The two tools

- `lookup_user_pref(query, user_id: Annotated[str, InjectedState("user_id")], store: Annotated[BaseStore, InjectedStore()])`
  — correctly injected: `user_id` comes from real graph state, `store` from
  the graph's own compiled `BaseStore`. The LLM never sees or invents
  either — its own model-facing schema only has `query`.
- `hallucinated_lookup(query, user_id)` — the bug: `user_id` shares its name
  with graph state's own `user_id` field, but nothing marks it injected, so
  it stays in the model-facing schema and the LLM fills it in itself.

## What this demonstrates about agent-trace's capture

1. **`branch:dispatch` spans for both of `create_react_agent`'s internal
   routers** — `should_continue` (keep tool-calling or stop) and
   `post_model_hook_router` (LangGraph's own router inserted after a
   `post_model_hook`). Both are built by LangGraph itself with
   `trace=False`; previously only a *failing* dispatch produced any span at
   all — a successful one (the overwhelmingly common case) left zero
   evidence of which router ran.
2. **`tool_inject:<name>` spans** showing exactly which argument names
   `ToolNode._inject_tool_args()` actually resolved, for every tool call:

   ```
   tool='lookup_user_pref'    injection_ran=True   injected_arg_keys='store,user_id'
   tool='hallucinated_lookup' injection_ran=False  injected_arg_keys=''
   ```

   Before this fix, this step was invisible to agent-trace entirely
   (confirmed via repo-wide grep: zero hits for `InjectedState`,
   `InjectedStore`, or `inject_tool_args`) — a `ValidationError` raised
   later inside a tool call looked identical whether the cause was a
   routing bug that skipped injection, a caller passing a malformed tool
   schema, or genuinely malformed model output.
3. **`find_tool_params_shaped_like_state()`** flags `hallucinated_lookup`'s
   `user_id` parameter automatically — both standalone (callable directly
   against a compiled graph, no run required) and wired into
   `LangGraphTracer(graph=...)`, which records the finding onto
   `trace.metadata["tool_state_shaped_params"]` at construction time.

## Run

```bash
python examples/08-langgraph-react-agent-tool-arg-injection/example.py
```

Expected output (abridged):

```
Schema-level check (find_tool_params_shaped_like_state):
  node='tools' tool='hallucinated_lookup' param='user_id'

trace.metadata['tool_state_shaped_params']: [{"node": "tools", "tool": "hallucinated_lookup", "param": "user_id"}]

--- branch:dispatch spans (routers exercised) ---
  router='should_continue'  status=OK
  router='post_model_hook_router'  status=OK
  router='should_continue'  status=OK
  router='post_model_hook_router'  status=OK

--- tool_inject:<name> spans (InjectedState/InjectedStore resolution) ---
  tool='lookup_user_pref'  injection_ran=True  injected_arg_keys='store,user_id'
  tool='hallucinated_lookup'  injection_ran=False  injected_arg_keys=''
```

## See also

- `examples/06-langgraph-handoff-parallel-tools/` — a different multi-agent
  routing failure (`Command(graph=Command.PARENT, ...)` handoffs racing
  parallel tool calls), also captured via the callback layer rather than
  HTTP replay.
- `docs/integrations/langgraph.md` — full LangGraph integration guide.
