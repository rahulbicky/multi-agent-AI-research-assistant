"""
tools/mcp_tools.py
==================
MCP tool loader — refactored from langgraph_mcp_backend.py.

Changes from original:
  - MCP URL read from MCP_SERVER_URL env variable (default: http://127.0.0.1:8000/mcp)
  - Graceful failure: returns [] if the server is unreachable
  - Async execution handled via a dedicated background thread loop
  - Used only when MCP_ENABLED=true (default)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dedicated event loop for async MCP operations (same pattern as original)
# ---------------------------------------------------------------------------

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _run_async(coro):
    """Run a coroutine on the background event loop and return its result."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()


def submit_async_task(coro):
    """
    Schedule a coroutine on the backend event loop without blocking.
    Used by the Streamlit streaming code to kick off async graph execution.
    """
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------

_client: MultiServerMCPClient | None = None


def _get_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        _client = MultiServerMCPClient(
            {
                "my_mcp": {
                    "transport": "streamable_http",
                    "url": settings.mcp_server_url,
                }
            }
        )
    return _client


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------

_mcp_tools_cache: List[BaseTool] | None = None


def load_mcp_tools() -> List[BaseTool]:
    """
    Load tools from the configured MCP server.

    Returns an empty list if:
      - MCP_ENABLED is false
      - The server is unreachable
      - Any other error occurs

    The result is cached after the first successful load.
    """
    global _mcp_tools_cache

    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    if not settings.mcp_enabled:
        logger.info("MCP is disabled (MCP_ENABLED=false). Skipping tool load.")
        _mcp_tools_cache = []
        return _mcp_tools_cache

    try:
        client = _get_client()
        tools = _run_async(client.get_tools())
        _mcp_tools_cache = tools
        logger.info(
            "Loaded %d MCP tools from %s: %s",
            len(tools),
            settings.mcp_server_url,
            [t.name for t in tools],
        )
    except Exception as exc:
        logger.warning(
            "MCP server at %s is unavailable (%s). Continuing without MCP tools.",
            settings.mcp_server_url,
            exc,
        )
        _mcp_tools_cache = []

    return _mcp_tools_cache
