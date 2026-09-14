"""
agents/research_agent.py
=========================
Research Agent.

Specialises in web research, news, stock prices, financial data, and
fact-checked synthesis. Uses DuckDuckGo (free), Alpha Vantage (optional),
and calculator.

After tool results are collected, a synthesis pass produces a clean,
well-structured answer — reducing hallucination and unsupported claims.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from core.config import settings
from core.context import trim_history
from core.state import AgentState
from tools.tools import RESEARCH_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """You are a research assistant with web search access.

You MUST use the duckduckgo_search tool for ANY question about:
- Current office holders (Prime Minister, Chief Minister, President, Mayor, CEO, etc.)
- Political leadership, government appointments, election results
- Latest news, recent events
- Current prices (stocks, fuel, commodities)
- Recent sports results
- Any fact that may have changed since your training

DO NOT answer these from memory — your training data may be outdated.
ALWAYS call duckduckgo_search FIRST, then answer from the search results.

For calculations, use the calculator tool.
For stock prices, use get_stock_price tool.

After you receive tool results, use them to write a clear, accurate answer.
If the search results do not contain the answer, say so honestly.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a research analyst producing a final answer.

Tool results are already in the conversation. Each duckduckgo_search result
includes a title, link, and snippet. Use them to:
- Write a clear, direct answer to the user's question
- Cite specific facts from the search results
- End with a "**Sources:**" section listing the 2-4 most relevant results as
  markdown links, e.g. "- [Title](https://link)" — only include links that
  were actually returned by the search tool, never invent one
- Note if information seems outdated or conflicting
- Do NOT call any more tools
- Do NOT add information not present in the tool results
- If tool results are empty or unhelpful, say you could not find reliable
  information and omit the Sources section
"""

# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------

_research_llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,
    api_key=settings.groq_api_key,
).bind_tools(RESEARCH_TOOLS)

_synthesis_llm = ChatGroq(
    model=settings.groq_model,
    temperature=0.1,
    api_key=settings.groq_api_key,
)

# ---------------------------------------------------------------------------
# Tool node
# ---------------------------------------------------------------------------

research_tool_node = ToolNode(RESEARCH_TOOLS)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def research_node(state: AgentState) -> AgentState:
    """Main research node — decides whether to call tools or synthesise."""
    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT), *trim_history(state["messages"])]
    try:
        response = _research_llm.invoke(messages)
    except Exception as exc:
        logger.error("Research agent LLM error: %s", exc)
        response = AIMessage(
            content="⚠️ I ran into an error while starting research on that. Please try again."
        )
    return {**state, "messages": [response], "agent_name": "Research"}


def research_synthesis_node(state: AgentState) -> AgentState:
    """
    Synthesis node — runs after tool results are in the conversation.
    Produces a final, clean answer without calling additional tools.
    """
    messages = [SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT), *trim_history(state["messages"])]
    try:
        response = _synthesis_llm.invoke(messages)
    except Exception as exc:
        logger.error("Research synthesis LLM error: %s", exc)
        response = AIMessage(
            content="⚠️ I found some search results but ran into an error summarising them. Please try again."
        )
    return {**state, "messages": [response], "agent_name": "Research"}


def get_research_tools():
    return RESEARCH_TOOLS
