"""
agents/coding_agent.py
=======================
Coding Agent.

Specialises in:
- Code generation in any language
- Code debugging and error analysis
- Code review and improvement suggestions
- Algorithm and data structure explanations
- Technical implementation guidance

Has access to the calculator tool for numeric checks.
No external API required beyond Groq.
"""

from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from core.config import settings
from core.state import AgentState
from tools.tools import CODING_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CODING_SYSTEM_PROMPT = """You are an expert software engineer and coding assistant.

Your capabilities:
- Write clean, well-commented code in any programming language
- Debug and fix code errors — explain the root cause clearly
- Review code and suggest improvements (readability, performance, security)
- Explain algorithms, data structures, and design patterns
- Help with system design and architecture questions
- Convert code between languages
- Write unit tests

Your style:
- Always use proper code blocks with the correct language identifier
- Explain your code with concise inline comments
- For bugs: identify the root cause first, then provide the fix
- For new code: briefly explain the approach before the implementation
- Keep explanations clear enough for someone learning

Use the calculator tool for any arithmetic you need to verify.
"""

# ---------------------------------------------------------------------------
# LLM + tools
# ---------------------------------------------------------------------------

_coding_llm = ChatGroq(
    model=settings.groq_model,
    temperature=0.1,   # Slight creativity for code, but mostly deterministic
    api_key=settings.groq_api_key,
).bind_tools(CODING_TOOLS)

coding_tool_node = ToolNode(CODING_TOOLS)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def coding_node(state: AgentState) -> AgentState:
    """Coding agent node — generates, reviews, or debugs code."""
    messages = [SystemMessage(content=CODING_SYSTEM_PROMPT), *state["messages"]]
    response = _coding_llm.invoke(messages)
    return {**state, "messages": [response], "agent_name": "Coding"}


def get_coding_tools():
    return CODING_TOOLS
