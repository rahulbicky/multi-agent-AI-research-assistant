"""
agents/general_agent.py
========================
General Chat Agent.

Handles casual conversation, greetings, general knowledge, and everything
that doesn't require specialised tools. Has access to web search and
calculator as optional fallbacks.

MCP tools are injected at graph build time if available.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

from core.config import settings
from core.context import trim_history
from core.state import AgentState
from tools.tools import GENERAL_TOOLS
from tools.mcp_tools import load_mcp_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = """You are a friendly, knowledgeable AI assistant.

Your role:
- Answer general questions clearly and helpfully
- Engage in conversation naturally
- Use web search when you need current or factual information
- Use the calculator for arithmetic

Be concise, warm, and accurate.
If a question is highly specialised (e.g. requires documents, research, or code),
let the user know they can ask more specifically.
"""

# ---------------------------------------------------------------------------
# LLM + tools
# ---------------------------------------------------------------------------

def _build_general_llm():
    all_tools = list(GENERAL_TOOLS) + list(load_mcp_tools())
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0.3,
        api_key=settings.groq_api_key,
    )
    if all_tools:
        return llm.bind_tools(all_tools), all_tools
    return llm, []


_llm_with_tools, _tools = _build_general_llm()


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def general_node(state: AgentState) -> AgentState:
    """Main LLM node — may answer directly or call a tool."""
    messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT), *trim_history(state["messages"])]
    try:
        response = _llm_with_tools.invoke(messages)
    except Exception as exc:
        logger.error("General agent LLM error: %s", exc)
        response = AIMessage(content="⚠️ I ran into an error answering that. Please try again.")
    return {**state, "messages": [response], "agent_name": "General"}


def general_tool_node_fn():
    """Return a configured ToolNode for the general agent's tools."""
    return ToolNode(_tools) if _tools else None


def get_general_tools():
    return _tools
