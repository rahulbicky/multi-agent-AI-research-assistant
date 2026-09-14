"""
core/context.py
================
Conversation-context helpers shared by every agent node.

Keeps per-turn LLM prompts bounded on long-running threads without touching
the full history persisted by the SQLite checkpointer — `state["messages"]`
is never mutated, only the window handed to a given LLM call.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Default number of most-recent messages passed to an agent LLM call.
# Generous enough to preserve multi-turn context, small enough to keep
# prompts (and token costs) bounded on long threads.
DEFAULT_WINDOW = 20


def trim_history(messages: List[BaseMessage], max_messages: int = DEFAULT_WINDOW) -> List[BaseMessage]:
    """
    Return a bounded window of the most recent messages.

    Cuts only on a HumanMessage boundary so a tool-call/tool-result pair from
    an older turn is never split apart (which would produce an invalid
    message sequence for the LLM API).
    """
    if len(messages) <= max_messages:
        return messages

    start = len(messages) - max_messages
    for i in range(start, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break
    else:
        start = 0

    return messages[start:]


def recent_exchange_summary(messages: List[BaseMessage], turns: int = 2) -> str:
    """
    Build a short "User: ... / Assistant: ..." transcript of the last few
    turns, excluding the most recent message. Used by the supervisor to
    route follow-up questions ("what about next year?") correctly without
    sending the full history to the (cheap, fast) routing call.
    """
    if len(messages) <= 1:
        return ""

    window = messages[:-1][-(turns * 2):]
    lines: list[str] = []
    for msg in window:
        if isinstance(msg, HumanMessage) and msg.content:
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            lines.append(f"Assistant: {msg.content}")

    return "\n".join(lines)
