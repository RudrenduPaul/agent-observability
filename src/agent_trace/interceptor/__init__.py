"""
agent-trace transport interceptors.

Each submodule patches one HTTP/RPC/WS client library (httpx, requests,
aiohttp, botocore/boto3, grpc, websockets, MCP stdio, ...) to record real
traffic into a ``Fixture`` and/or replay it offline — see
``httpx_hook.py``'s module docstring for the canonical Recording*/Replay*
Transport shape every other interceptor here mirrors.

Submodules are exported lazily here (PEP 562 module ``__getattr__``) so that
``import agent_trace.interceptor`` never fails regardless of which optional
client libraries happen to be installed. Most submodules (httpx_hook.py,
aiohttp_hook.py, botocore_hook.py, websocket_hook.py, stdio_hook.py) already
defer their own ``import <library>`` to call time and are always safely
importable; ``grpc_hook.py`` and ``requests_patch.py`` import their target
library eagerly (``grpc``/``requests``) since ``Tracer._patch_grpc`` /
``Tracer._patch_requests`` in ``agent_trace/__init__.py`` already wrap the
*submodule* import itself in ``try/except ImportError`` — lazy resolution
here means accessing e.g. ``agent_trace.interceptor.grpc_hook`` only fails
at that point (with the normal ``ImportError``), not merely by importing
this package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_trace.interceptor import (
        aiohttp_hook,
        botocore_hook,
        grpc_hook,
        httpx_hook,
        logging_hook,
        requests_patch,
        sse,
        stdio_hook,
        warnings_hook,
        websocket_hook,
    )

__all__ = [
    "aiohttp_hook",
    "botocore_hook",
    "grpc_hook",
    "httpx_hook",
    "logging_hook",
    "requests_patch",
    "sse",
    "stdio_hook",
    "warnings_hook",
    "websocket_hook",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"agent_trace.interceptor.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
