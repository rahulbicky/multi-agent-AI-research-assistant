"""
agents/supervisor.py
====================
Supervisor / Router Agent.

Classifies the user's latest message into one of four routing targets:
  general | research | rag | coding

Uses ChatGroq for classification — no hardcoded keyword rules.
Falls back to "general" on any error so the system never crashes.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from core.config import settings
from core.context import recent_exchange_summary
from core.state import AgentState

logger = logging.getLogger(__name__)

VALID_ROUTES = {"general", "research", "rag", "coding"}

# ---------------------------------------------------------------------------
# Routing model — light, fast, temperature=0 for deterministic routing
# ---------------------------------------------------------------------------

_router_llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,
    api_key=settings.groq_api_key,
)

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """You are a routing agent for an AI assistant system.
Your ONLY job is to classify the user's message into exactly one of these categories:

  research  - REQUIRED for ANY question about:
               • current office holders (Prime Minister, Chief Minister, President, CEO, etc.)
               • current political positions or appointments
               • latest news, recent events, today's happenings
               • current prices (stocks, crypto, commodities, fuel)
               • current sports scores, standings, results
               • recent releases (movies, software, products)
               • current company information
               • any fact that may have changed recently
               • anything with words: current, latest, recent, now, today, new, 2024, 2025, 2026

  rag       - Questions about an uploaded document, PDF, or file.
               Signals: "the document", "the PDF", "the file", "according to",
               "in the report", "what does it say", "summarize the PDF".

  coding    - Programming, code generation, debugging, algorithms, technical
               implementation, any question involving writing or reading code.

  general   - ONLY for: casual conversation, greetings, opinions, timeless facts
               (e.g. "what is gravity", "explain recursion", "what is Python").
               Do NOT use this for anything that might need current information.

IMPORTANT RULES:
- When in doubt between 'general' and 'research', always choose 'research'.
- Questions about WHO currently holds ANY position ALWAYS go to 'research'.
- Questions about ANY country's government, politics, leadership go to 'research'.

Respond with EXACTLY ONE WORD — the category name. No punctuation, no explanation.
"""


def _extract_route(raw: str) -> Optional[str]:
    """
    Parse the router LLM's reply into one of VALID_ROUTES.

    Tolerant of extra punctuation/words the model may add (e.g. "research.",
    "Category: research") — looks for an exact match first, then falls back
    to searching for a valid route word anywhere in the reply.
    """
    cleaned = raw.strip().lower().strip(" .:!\"'`")
    if cleaned in VALID_ROUTES:
        return cleaned
    for route in VALID_ROUTES:
        if route in cleaned:
            return route
    return None


def supervisor_node(state: AgentState) -> AgentState:
    """
    Inspect the last human message and route to the correct agent.

    Sets state["next_agent"] to one of: general | research | rag | coding
    """
    messages = state.get("messages", [])

    # Find the last human message for classification
    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not last_human:
        logger.debug("Supervisor: no human message found, defaulting to general")
        return {**state, "next_agent": "general"}

    # A short slice of recent turns helps route follow-up questions
    # ("what about last year?") that only make sense with context.
    context = recent_exchange_summary(messages)
    prompt_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]
    if context:
        prompt_messages.append(
            SystemMessage(content=f"Recent conversation context (for reference only):\n{context}")
        )
    prompt_messages.append(HumanMessage(content=last_human))

    # Try once, and retry once more on an unparseable (not a failed-call) reply.
    for attempt in range(2):
        try:
            response = _router_llm.invoke(prompt_messages)
        except Exception as exc:
            logger.error("Supervisor classification error: %s — defaulting to general", exc)
            return {**state, "next_agent": "general"}

        route = _extract_route(str(response.content))
        if route:
            logger.info("Supervisor routed '%s...' → %s", last_human[:60], route)
            return {**state, "next_agent": route}

        logger.warning(
            "Supervisor returned unparseable route '%s' (attempt %d/2)",
            response.content, attempt + 1,
        )

    logger.warning("Supervisor could not classify '%s...', defaulting to general", last_human[:60])
    return {**state, "next_agent": "general"}


# ---------------------------------------------------------------------------
# Routing function (used as conditional edge in graph.py)
# ---------------------------------------------------------------------------

def route_to_agent(state: AgentState) -> str:
    """Return the next node name based on state["next_agent"]."""
    return state.get("next_agent", "general")
