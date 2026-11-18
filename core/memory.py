"""
core/memory.py
==============
SQLite checkpointer factory with per-user isolation.

Each unique (user_id, thread_id) pair is stored as a separate LangGraph
checkpoint thread — complete isolation at the database level.

Conversation management
------------------------
Thread titles are stored in a lightweight ``thread_titles`` table that this
module owns.  The LangGraph ``checkpoints`` / ``writes`` tables remain
untouched by application code — deletions use direct SQL to avoid
re-implementing LangGraph internals.

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

# Maximum allowed length for a conversation title (characters)
MAX_TITLE_LENGTH = 120


def _get_connection() -> sqlite3.Connection:
    """Return (and lazily initialise) the shared SQLite connection."""
    global _connection
    if _connection is None:
        get_checkpointer()  # initialises _connection as a side-effect
    assert _connection is not None
    return _connection


def _ensure_title_table(conn: sqlite3.Connection) -> None:
    """
    Create the ``thread_titles`` metadata table if it does not yet exist.

    Schema
    ------
    composite_thread_id  TEXT  PRIMARY KEY  — "{user_id}:{thread_id}"
    title                TEXT  NOT NULL     — human-readable conversation name
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_titles (
            composite_thread_id TEXT PRIMARY KEY,
            title               TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _ensure_settings_table(conn: sqlite3.Connection) -> None:
    """
    Create the ``user_settings`` table for persisting preferences like theme.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_settings (
            setting_key   TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL
        )
        """
    )
    conn.commit()


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
        # Ensure our auxiliary tables exist on first connection
        _ensure_title_table(_connection)
        _ensure_settings_table(_connection)

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


# ---------------------------------------------------------------------------
# Title management
# ---------------------------------------------------------------------------

def save_thread_title(user_id: str, thread_id: str, title: str) -> None:
    """
    Persist the display title for a (user_id, thread_id) pair.

    Validates:
    - ``title`` is non-empty after stripping whitespace.
    - ``title`` does not exceed MAX_TITLE_LENGTH characters.
    - Only the owner ``user_id`` can write this record (composite key
      embeds the user_id so another user's key would never match).
    """
    title = title.strip()
    if not title:
        raise ValueError("Conversation title cannot be empty.")
    if len(title) > MAX_TITLE_LENGTH:
        raise ValueError(
            f"Title too long ({len(title)} chars). Maximum is {MAX_TITLE_LENGTH}."
        )

    composite = f"{user_id}:{thread_id}"
    conn = _get_connection()
    _ensure_title_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO thread_titles (composite_thread_id, title) VALUES (?, ?)",
        (composite, title),
    )
    conn.commit()
    logger.info("Saved title for %s: %r", composite, title)


def get_thread_title(user_id: str, thread_id: str) -> str | None:
    """
    Return the persisted title for a (user_id, thread_id) pair, or None if
    no title has been saved yet.
    """
    composite = f"{user_id}:{thread_id}"
    conn = _get_connection()
    _ensure_title_table(conn)
    row = conn.execute(
        "SELECT title FROM thread_titles WHERE composite_thread_id = ?",
        (composite,),
    ).fetchone()
    return row[0] if row else None


def load_all_thread_titles(user_id: str) -> dict[str, str]:
    """
    Return a mapping of ``{thread_id: title}`` for every thread that has a
    persisted title for this user.
    """
    prefix = f"{user_id}:"
    conn = _get_connection()
    _ensure_title_table(conn)
    rows = conn.execute(
        "SELECT composite_thread_id, title FROM thread_titles WHERE composite_thread_id LIKE ?",
        (f"{prefix}%",),
    ).fetchall()
    result: dict[str, str] = {}
    for composite, title in rows:
        if composite.startswith(prefix):
            tid = composite[len(prefix):]
            result[tid] = title
    return result


# ---------------------------------------------------------------------------
# Thread deletion
# ---------------------------------------------------------------------------

def delete_thread(user_id: str, thread_id: str) -> bool:
    """
    Permanently remove all LangGraph checkpoint data and the title record for
    a (user_id, thread_id) pair from the SQLite database.

    Safety
    ------
    - The composite key ``{user_id}:{thread_id}`` is used for all DELETE
      statements so that one user can never accidentally delete another's data.
    - Deletions from ``checkpoints``, ``writes``, and ``thread_titles`` are
      all performed inside a single transaction; on failure the transaction is
      rolled back and no partial state is left.
    - If the thread does not exist the function returns False without error.

    Returns True on success, False if the thread did not exist.
    """
    composite = f"{user_id}:{thread_id}"
    conn = _get_connection()
    _ensure_title_table(conn)

    try:
        with conn:                      # transaction — auto-commit or rollback
            # Remove all checkpoints for this thread
            cur = conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (composite,)
            )
            deleted_checkpoints = cur.rowcount

            # Remove all write records for this thread
            conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (composite,)
            )

            # Remove the title record
            conn.execute(
                "DELETE FROM thread_titles WHERE composite_thread_id = ?",
                (composite,),
            )

        if deleted_checkpoints == 0:
            logger.warning(
                "delete_thread: no checkpoints found for %s — may already be deleted",
                composite,
            )
            return False

        logger.info(
            "Deleted thread %s: removed %d checkpoint row(s)",
            composite, deleted_checkpoints,
        )
        return True

    except sqlite3.Error as exc:
        logger.error("delete_thread failed for %s: %s", composite, exc)
        raise RuntimeError(f"Failed to delete conversation: {exc}") from exc


# ---------------------------------------------------------------------------
# Conversation Search
# ---------------------------------------------------------------------------

def search_user_threads(user_id: str, query: str) -> list[str]:
    """
    Search conversation titles and message content for a user.
    Returns a deduplicated list of matching raw thread_ids.
    """
    cleaned_query = (query or "").strip().lower()
    if not cleaned_query:
        return list_user_threads(user_id)

    prefix = f"{user_id}:"
    conn = _get_connection()
    _ensure_title_table(conn)
    matching_threads: list[str] = []

    # 1. Search in thread_titles table
    try:
        cur = conn.execute(
            "SELECT composite_thread_id FROM thread_titles WHERE composite_thread_id LIKE ? AND LOWER(title) LIKE ?",
            (f"{prefix}%", f"%{cleaned_query}%"),
        )
        for (comp,) in cur.fetchall():
            if comp.startswith(prefix):
                tid = comp[len(prefix):]
                if tid not in matching_threads:
                    matching_threads.append(tid)
    except sqlite3.Error as exc:
        logger.warning("Error searching thread titles: %s", exc)

    # 2. Search in writes table (channel='messages')
    try:
        q_bytes = f"%{cleaned_query}%".encode("utf-8")
        cur = conn.execute(
            "SELECT DISTINCT thread_id FROM writes WHERE thread_id LIKE ? AND channel='messages' AND value LIKE ?",
            (f"{prefix}%", q_bytes),
        )
        for (comp,) in cur.fetchall():
            if comp.startswith(prefix):
                tid = comp[len(prefix):]
                if tid not in matching_threads:
                    matching_threads.append(tid)
    except sqlite3.Error as exc:
        logger.warning("Error searching message writes: %s", exc)

    return matching_threads


# ---------------------------------------------------------------------------
# User Preferences / Settings (e.g. Theme)
# ---------------------------------------------------------------------------

def get_user_setting(user_id: str, key: str, default: str = "") -> str:
    """Retrieve a persisted user preference from SQLite."""
    conn = _get_connection()
    _ensure_settings_table(conn)
    composite = f"{user_id}:{key}"
    row = conn.execute(
        "SELECT setting_value FROM user_settings WHERE setting_key = ?",
        (composite,),
    ).fetchone()
    if row:
        return row[0]
    # Check global fallback setting
    row_global = conn.execute(
        "SELECT setting_value FROM user_settings WHERE setting_key = ?",
        (key,),
    ).fetchone()
    return row_global[0] if row_global else default


def set_user_setting(user_id: str, key: str, value: str) -> None:
    """Save a user preference persistently in SQLite."""
    conn = _get_connection()
    _ensure_settings_table(conn)
    composite = f"{user_id}:{key}"
    conn.execute(
        "INSERT OR REPLACE INTO user_settings (setting_key, setting_value) VALUES (?, ?)",
        (composite, str(value)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Checkpoint Rewinding (Regenerate / Edit Message)
# ---------------------------------------------------------------------------

def rewind_for_regeneration(user_id: str, thread_id: str) -> str | None:
    """
    Rewind the checkpoint state for a thread to regenerate the last AI response.
    Removes the last turn (the last HumanMessage and all subsequent AI/tool messages).
    Returns the content of the removed HumanMessage so the caller can re-stream it,
    or None if no HumanMessage was found.
    """
    from langchain_core.messages import HumanMessage, RemoveMessage
    from core.graph import chatbot

    config = make_config(user_id, thread_id)
    state = chatbot.get_state(config)
    msgs = state.values.get("messages", [])
    if not msgs:
        return None

    last_human_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        return None

    last_user_prompt = str(msgs[last_human_idx].content)
    to_remove = msgs[last_human_idx:]
    chatbot.update_state(
        config,
        {"messages": [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]},
    )
    return last_user_prompt


def rewind_for_edit(user_id: str, thread_id: str, user_message_index: int) -> bool:
    """
    Rewind the checkpoint state to allow editing a previous user message.
    `user_message_index` is the 0-based index of the HumanMessage among all HumanMessages.
    Removes that HumanMessage and all subsequent messages.
    Returns True on success, False if the target index is invalid.
    """
    from langchain_core.messages import HumanMessage, RemoveMessage
    from core.graph import chatbot

    config = make_config(user_id, thread_id)
    state = chatbot.get_state(config)
    msgs = state.values.get("messages", [])
    if not msgs:
        return False

    human_count = 0
    target_idx = None
    for i, m in enumerate(msgs):
        if isinstance(m, HumanMessage):
            if human_count == user_message_index:
                target_idx = i
                break
            human_count += 1

    if target_idx is None:
        return False

    to_remove = msgs[target_idx:]
    chatbot.update_state(
        config,
        {"messages": [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]},
    )
    return True
