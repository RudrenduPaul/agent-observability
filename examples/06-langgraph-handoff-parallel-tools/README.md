# Example 06 — Multi-Agent Handoff + Parallel-Tool-Call Race (#5277)

Reproduces the exact failure class behind
[langgraph#5277](https://github.com/langchain-ai/langgraph/issues/5277):
"Agent handoff with parallel calls error with openai."

No API key required — the crash is 100% free and deterministic (confirmed
in the original investigation of this issue: zero HTTP exchanges occur
before the failure).

## Prerequisites

```bash
pip install agent-trace[langgraph]
```

## The race

A `flight_assistant` agent returns two parallel tool calls in one turn:

- `book_flight` — a normal tool
- `transfer_to_hotel_assistant` — a handoff tool returning
  `Command(graph=Command.PARENT, ...)`, LangGraph's own documented
  multi-agent handoff pattern

The handoff's parent-graph jump reaches `hotel_assistant` before
`book_flight`'s `ToolMessage` has been merged into shared state. When
`hotel_assistant`'s own `call_model` step runs, its message history contains
an `AIMessage` with two tool calls but only one matching `ToolMessage`.
LangGraph's own `_validate_chat_history()` rejects it locally:

```
ValueError: Found AIMessages with tool_calls that do not have a
corresponding ToolMessage. [...] (ErrorCode.INVALID_CHAT_HISTORY)
```

The original reporter self-diagnosed this exact mechanism unaided in the
GitHub thread; a LangChain maintainer confirmed it's expected behavior with
parallel tool calls enabled on a handoff-capable agent ("parallel tool
calling should be disabled since it's easy to get into an ill-defined
state").

## What this demonstrates about agent-trace's capture

1. **Zero HTTP exchanges recorded before the crash.** The failure is
   entirely client-side — the HTTP interceptor
   (`src/agent_trace/interceptor/httpx_hook.py`) has nothing to see here.
   All the evidence comes from the LangGraph callback layer
   (`LangGraphTracer`, `src/agent_trace/integrations/langgraph.py`), not
   HTTP replay.
2. **The `ParentCommand` handoff jump closes OK, not ERROR.** Three spans
   (`node:flight_assistant`, the root `node:LangGraph`, `node:tools`) carry
   LangGraph's internal `ParentCommand` control-flow signal — a successful
   handoff, not a failure — and close `OK` with `langgraph.handoff=true`.
   Before this fix, these were indistinguishable from a genuine crash in the
   trace (`status=ERROR` either way).
3. **The genuine failure is classified automatically.** Every span carrying
   the real `ValueError` gets `error.origin=application` and
   `error.known_pattern=langgraph_invalid_chat_history` — turning "read the
   raw trace.json and recognize the pattern yourself" (what the reporter did
   unaided) into a one-line pointer already on the span.

## Run

```bash
python examples/06-langgraph-handoff-parallel-tools/example.py
```

Expected output (abridged):

```
Crashed as expected: ValueError: Found AIMessages with tool_calls that do
not have a corresponding ToolMessage. ...

HTTP exchanges recorded before the crash: 0
(zero — the failure is entirely client-side; nothing ever reached the wire)

--- Span tree ---
Trace: langgraph-handoff-parallel-tools-race  run_<id>
└── node:LangGraph  ERROR
    ├── node:flight_assistant  OK
    │   └── node:LangGraph  OK
    │       ├── node:call_model  OK
    │       │   └── llm:FlightFakeChatModel  OK
    │       └── node:tools  OK
    └── node:hotel_assistant  ERROR
        └── node:hotel_assistant  ERROR
            └── node:agent  ERROR
                └── node:call_model  ERROR

3 span(s) correctly closed OK for the ParentCommand handoff:
  node:flight_assistant  (langgraph.control_flow_signal=ParentCommand)
  node:LangGraph  (langgraph.control_flow_signal=ParentCommand)
  node:tools  (langgraph.control_flow_signal=ParentCommand)

5 span(s) closed ERROR for the genuine failure:
  node:LangGraph       origin=application  known_pattern=langgraph_invalid_chat_history
  node:hotel_assistant origin=application  known_pattern=langgraph_invalid_chat_history
  node:hotel_assistant origin=application  known_pattern=langgraph_invalid_chat_history
  node:agent           origin=application  known_pattern=langgraph_invalid_chat_history
  node:call_model      origin=application  known_pattern=langgraph_invalid_chat_history
```

## See also

- `examples/05-parallel-command-parent-routing/` — the related #7129
  silent-drop shape (a distinct failure: parallel `Command.PARENT` updates
  to the *same* channel, one surviving silently, rather than a loud
  `INVALID_CHAT_HISTORY` crash)
- `docs/integrations/langgraph.md` — full LangGraph integration guide
