"""
core/state.py
=============
Shared AgentState TypedDict used across the entire multi-agent graph.
"""

from __future__ import annotations

from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Single shared state that flows through every node in the multi-agent graph.

    Fields
    ------
    messages      : Full conversation history (human + AI + tool messages).
                    `add_messages` reducer appends new messages rather than
                    replacing the whole list — same pattern as existing code.
    next_agent    : Set by the supervisor to indicate which agent runs next.
                    One of: "general" | "research" | "rag" | "coding".
    user_id       : Unique identifier for the browser session / user.
                    Used together with thread_id to isolate state in SQLite.
    thread_id     : Conversation thread identifier (UUID string).
    agent_name    : Name of the agent that produced the last AI response.
                    Displayed as a badge in the UI.
    documents     : List of document filenames indexed for this session/thread.
                    Updated by the RAG agent after ingestion.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: Optional[str]
    user_id: Optional[str]
    thread_id: Optional[str]
    agent_name: Optional[str]
    documents: Optional[List[str]]
