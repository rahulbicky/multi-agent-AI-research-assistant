"""
core/memory.py
==============
SQLite checkpointer factory with per-user isolation.

Each unique (user_id, thread_id) pair is stored as a separate LangGraph
checkpoint thread — complete isolation at the database level.

Migration path to PostgreSQL
------------------------------
Install `langgraph-checkpoint-postgres` and replace SqliteSaver with
AsyncPostgresSaver from that package. The rest of the code is identical.
"""

from __future__ import annotations

import logging
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single shared SQLite connection (thread-safe for read-heavy Streamlit use)
# ---------------------------------------------------------------------------

_connection: sqlite3.Connection | None = None
_checkpointer: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    """
    Return the singleton SqliteSaver.

    The connection is created once and reused across all calls (Streamlit
    re-runs the script on every interaction, so module-level singletons are
    the right pattern here).
    """
    global _connection, _checkpointer

    if _checkpointer is None:
        logger.info("Initialising SQLite checkpointer at %s", settings.db_path)
        _connection = sqlite3.connect(
            database=settings.db_path,
            check_same_thread=False,  # Streamlit uses multiple threads
        )
        _checkpointer = SqliteSaver(conn=_connection)

    return _checkpointer


def make_config(user_id: str, thread_id: str) -> dict:
    """
    Build a LangGraph run config that isolates this (user, thread) pair.

    The composite key ``{user_id}:{thread_id}`` is used as the LangGraph
    thread_id so that two different users with the same thread UUID never
    share state.
    """
    composite_thread_id = f"{user_id}:{thread_id}"
    return {
        "configurable": {"thread_id": composite_thread_id},
        "metadata": {"user_id": user_id, "thread_id": thread_id},
        "run_name": "chat_turn",
    }


def list_user_threads(user_id: str) -> list[str]:
    """
    Return all thread_ids that belong to a specific user.

    Only threads whose composite key starts with ``{user_id}:`` are returned.
    This prevents User A from seeing User B's conversations.
    """
    checkpointer = get_checkpointer()
    prefix = f"{user_id}:"
    threads: list[str] = []

    try:
        for checkpoint in checkpointer.list(None):
            composite = checkpoint.config["configurable"].get("thread_id", "")
            if composite.startswith(prefix):
                raw_thread_id = composite[len(prefix):]
                if raw_thread_id not in threads:
                    threads.append(raw_thread_id)
    except Exception as exc:
        logger.warning("Could not list threads for user %s: %s", user_id, exc)

    return threads
