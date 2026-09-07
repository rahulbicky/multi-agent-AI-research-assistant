"""
core/graph.py
=============
Assembles the full multi-agent LangGraph.

Architecture
------------

START
  │
  ▼
supervisor_node          ← classifies query, sets next_agent
  │
  ├── "general"  → general_node  ──► general_tools?  ──► general_node  ──► END
  │
  ├── "research" → research_node ──► research_tools ──► research_synthesis ──► END
  │
  ├── "rag"      → rag_node      ──► rag_tools?      ──► rag_node       ──► END
  │
  └── "coding"   → coding_node   ──► coding_tools?   ──► coding_node    ──► END

Each agent has its own tool node.  Tool calls loop back to the same agent node
(not to the supervisor) so the agent can reason over the tool result.

The research agent has an extra synthesis node to produce a clean final answer.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from agents.coding_agent import coding_node, coding_tool_node
from agents.general_agent import general_node, general_tool_node_fn
from agents.rag_agent import rag_node, rag_tool_node
from agents.research_agent import (
    research_node,
    research_synthesis_node,
    research_tool_node,
)
from agents.supervisor import route_to_agent, supervisor_node
from core.memory import get_checkpointer
from core.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom tools_condition wrappers
# ---------------------------------------------------------------------------
# We need per-agent tool routing.  We tag each agent's tool calls with a
# unique next-hop so the conditional edge can direct them correctly.
# ---------------------------------------------------------------------------

def _research_tools_condition(state: AgentState) -> str:
    """After research_node: go to tools, or synthesis if done."""
    from langchain_core.messages import AIMessage
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "research_tools"
    return "research_synthesis"


def _agent_tools_condition(tool_node_name: str):
    """
    Factory: returns a condition function that routes to `tool_node_name`
    when the last message has tool calls, otherwise to END.
    """
    def _condition(state: AgentState) -> str:
        from langchain_core.messages import AIMessage
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return tool_node_name
        return END
    return _condition


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    """Construct and compile the multi-agent StateGraph."""

    graph = StateGraph(AgentState)

    # ── Supervisor ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)

    # ── General agent ─────────────────────────────────────────────────────
    graph.add_node("general_node", general_node)
    general_tools_node = general_tool_node_fn()
    if general_tools_node:
        graph.add_node("general_tools", general_tools_node)

    # ── Research agent ────────────────────────────────────────────────────
    graph.add_node("research_node", research_node)
    graph.add_node("research_tools", research_tool_node)
    graph.add_node("research_synthesis", research_synthesis_node)

    # ── RAG agent ─────────────────────────────────────────────────────────
    graph.add_node("rag_node", rag_node)
    graph.add_node("rag_tools", rag_tool_node)

    # ── Coding agent ──────────────────────────────────────────────────────
    graph.add_node("coding_node", coding_node)
    graph.add_node("coding_tools", coding_tool_node)

    # ── Edges: START → supervisor ──────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Edges: supervisor → agents ────────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "general": "general_node",
            "research": "research_node",
            "rag": "rag_node",
            "coding": "coding_node",
        },
    )

    # ── Edges: General agent ──────────────────────────────────────────────
    if general_tools_node:
        graph.add_conditional_edges(
            "general_node",
            _agent_tools_condition("general_tools"),
            {"general_tools": "general_tools", END: END},
        )
        graph.add_edge("general_tools", "general_node")
    else:
        graph.add_edge("general_node", END)

    # ── Edges: Research agent ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "research_node",
        _research_tools_condition,
        {"research_tools": "research_tools", "research_synthesis": "research_synthesis"},
    )
    # After tools run, ALWAYS go to synthesis — prevents recursion loop
    graph.add_edge("research_tools", "research_synthesis")
    graph.add_edge("research_synthesis", END)

    # ── Edges: RAG agent ──────────────────────────────────────────────────
    graph.add_conditional_edges(
        "rag_node",
        _agent_tools_condition("rag_tools"),
        {"rag_tools": "rag_tools", END: END},
    )
    graph.add_edge("rag_tools", "rag_node")

    # ── Edges: Coding agent ───────────────────────────────────────────────
    graph.add_conditional_edges(
        "coding_node",
        _agent_tools_condition("coding_tools"),
        {"coding_tools": "coding_tools", END: END},
    )
    graph.add_edge("coding_tools", "coding_node")

    # ── Compile with SQLite checkpointer ──────────────────────────────────
    checkpointer = get_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Multi-agent graph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Singleton graph instance
# ---------------------------------------------------------------------------

chatbot = build_graph()
