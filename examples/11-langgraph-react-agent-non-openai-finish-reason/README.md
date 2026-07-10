# Example 11 — `create_react_agent` + a Non-OpenAI Provider's `finish_reason` Anomaly

Closes the gap flagged against [issue #6574](https://github.com/langchain-ai/langgraph/issues/6574):
no existing example demonstrates `create_react_agent` against a provider
whose `finish_reason` vocabulary diverges from OpenAI's well-known
`stop`/`tool_calls`/`length` set.

No API key required — `FakeGeminiChatModel` is a plain `BaseChatModel`
stand-in with no network calls.

## The anomaly

Gemini can return `finish_reason="MALFORMED_FUNCTION_CALL"` alongside an
`AIMessage` that still has `tool_calls` populated — the model proposed a
tool call, but Gemini's own validation flagged the generated arguments as
malformed. Most ReAct-agent loops only check `response.tool_calls`, so this
dispatches exactly like a clean, validated tool call with nothing to
indicate a provider-side red flag was raised.

## What this shows

`LangGraphTracer`'s `on_llm_end` capture
(`src/agent_trace/integrations/langgraph.py::_extract_finish_reason`/
`_record_llm_end_data`) persists `llm.finish_reason` and
`llm.has_tool_calls` together on every LLM span — including this one — so
a developer inspecting the trace can see the co-occurrence: a tool call was
dispatched (`llm.has_tool_calls=True`) despite a finish reason that isn't
`stop`/`tool_calls` (`llm.finish_reason="MALFORMED_FUNCTION_CALL"`).

## Run

```bash
pip install agent-trace[langgraph]
python examples/11-langgraph-react-agent-non-openai-finish-reason/example.py
```
