"""
agent-trace framework integrations.

Each submodule wraps a specific agent framework's callback/event-hook
surface to emit agent-trace spans (e.g. ``langgraph.py`` for LangGraph,
``openai_agents.py`` for the OpenAI Agents SDK, ``crewai.py`` for crewAI).

Submodules are exported lazily here (PEP 562 module ``__getattr__``) so that
``import agent_trace.integrations`` never fails regardless of which optional
framework SDKs happen to be installed — none of the submodules themselves
import their target SDK at module import time either (see langgraph.py /
openai_agents.py, which defer the real ``import langgraph`` / ``import
agents`` to call time inside the class/function that needs it). Accessing a
specific submodule (e.g. ``agent_trace.integrations.crewai`` or ``from
agent_trace.integrations import crewai``) triggers that submodule's own
import; only *calling* something inside it that actually needs the SDK
raises a helpful ``ImportError`` with an install hint if the SDK is
missing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_trace.integrations import (
        agno,
        autogen,
        crewai,
        google_genai,
        haystack,
        langgraph,
        langgraph_checkpoint,
        langgraph_state_diff,
        langgraph_stream_debug,
        llama_index,
        mcp,
        openai_agents,
        pydantic_ai,
        streaming,
    )

__all__ = [
    "agno",
    "autogen",
    "crewai",
    "google_genai",
    "haystack",
    "langgraph",
    "langgraph_checkpoint",
    "langgraph_state_diff",
    "langgraph_stream_debug",
    "llama_index",
    "mcp",
    "openai_agents",
    "pydantic_ai",
    "streaming",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"agent_trace.integrations.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
