"""
frontend/app.py
================
Multi-Agent AI Research Chatbot — Main Entry Point.

ChatGPT / Gemini-style UI backed by a LangGraph multi-agent system.

Run (from the project root):
    streamlit run frontend/app.py
    # or
    python run.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from pathlib import Path

# `core/`, `agents/` and `tools/` live at the project root, one level above
# this file. Streamlit only puts this file's own directory on sys.path, so
# the root must be added explicitly for these absolute imports to resolve
# regardless of the working directory the app is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.config import settings
from core.export import export_to_markdown, export_to_txt
from core.memory import (
    list_user_threads,
    make_config,
    save_thread_title,
    get_thread_title,
    load_all_thread_titles,
    delete_thread,
    search_user_threads,
    get_user_setting,
    set_user_setting,
    rewind_for_edit,
    save_doubt_meta,
    get_doubt_meta,
    add_doubt_message,
    get_doubt_messages,
    clear_doubt_messages,
)
from core.doubt import answer_doubt
from core.graph import chatbot
from tools.tools import (
    ingest_document,
    ingest_pdf,
    get_retriever_meta,
    delete_vector_store,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Message Formatting & Idempotent History Utilities
# ─────────────────────────────────────────────────────────────────────────────

def clean_markdown_content(text: str) -> str:
    """
    Format message text cleanly for Markdown rendering.
    Converts literal HTML <br>, <br/>, <br /> into native Markdown newlines,
    and strips unwanted trailing whitespace so literal tags do not appear.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)


def append_user_message(content: str) -> bool:
    """
    Safely append a user message to message_history with idempotency.
    Returns True if appended, False if duplicate was blocked.
    """
    cleaned = clean_markdown_content(str(content or "")).strip()
    if not cleaned:
        return False

    msgs = st.session_state.setdefault("message_history", [])

    # Idempotency guard: block consecutive duplicate user messages
    if msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == cleaned:
        logger.warning("Idempotency guard: blocked duplicate user message: '%s'", cleaned[:40])
        return False

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    msgs.append({
        "id": msg_id,
        "role": "user",
        "content": cleaned,
        "agent_name": None,
    })
    st.session_state["message_history"] = msgs
    return True


def append_assistant_message(content: str, agent_name: str = "General") -> bool:
    """
    Safely append an assistant message to message_history with idempotency.
    """
    cleaned = clean_markdown_content(str(content or "")).strip()
    if not cleaned:
        return False

    msgs = st.session_state.setdefault("message_history", [])

    # Idempotency guard: block consecutive duplicate assistant messages
    if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("content") == cleaned:
        logger.warning("Idempotency guard: blocked duplicate assistant message: '%s'", cleaned[:40])
        return False

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    msgs.append({
        "id": msg_id,
        "role": "assistant",
        "content": cleaned,
        "agent_name": agent_name,
    })
    st.session_state["message_history"] = msgs
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Session Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    """Initialise all Streamlit session state variables."""

    # Unique per-browser session user ID
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = f"user_{uuid.uuid4().hex[:12]}"

    _uid = st.session_state["user_id"]

    # Active theme: persisted in SQLite user_settings (default: dark)
    if "theme" not in st.session_state:
        saved_theme = get_user_setting(_uid, "theme", default="dark")
        st.session_state["theme"] = saved_theme if saved_theme in ("dark", "light") else "dark"

    # Current conversation thread
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Display-side message history {role, content, agent_name}
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    # Sidebar thread list {thread_id: title}
    if "thread_titles" not in st.session_state:
        existing = list_user_threads(_uid)
        persisted = load_all_thread_titles(_uid)
        st.session_state["thread_titles"] = {
            tid: persisted.get(tid, f"Chat {i + 1}")
            for i, tid in enumerate(existing)
        }

    # Register current thread if not already tracked
    tid = st.session_state["thread_id"]
    if tid not in st.session_state["thread_titles"]:
        st.session_state["thread_titles"][tid] = "New Chat"

    # Streaming states
    if "_is_generating" not in st.session_state:
        st.session_state["_is_generating"] = False
    if "_current_stream_text" not in st.session_state:
        st.session_state["_current_stream_text"] = ""
    if "_current_stream_agent" not in st.session_state:
        st.session_state["_current_stream_agent"] = "General"

    # Auto-restore thread messages if thread exists but message_history is empty
    if not st.session_state["message_history"]:
        try:
            cfg = make_config(_uid, tid)
            state = chatbot.get_state(config=cfg)
            raw = state.values.get("messages", [])
            if raw:
                restored = []
                for m in raw:
                    if isinstance(m, HumanMessage):
                        c = clean_markdown_content(str(m.content or "")).strip()
                        if c and (not restored or restored[-1].get("role") != "user" or restored[-1].get("content") != c):
                            restored.append({
                                "id": getattr(m, "id", None) or f"msg_{uuid.uuid4().hex[:12]}",
                                "role": "user",
                                "content": c,
                                "agent_name": None,
                            })
                    elif isinstance(m, AIMessage) and m.content:
                        tool_calls = getattr(m, "tool_calls", [])
                        c = clean_markdown_content(str(m.content or "")).strip()
                        if tool_calls and not c:
                            continue
                        if c and (not restored or restored[-1].get("role") != "assistant" or restored[-1].get("content") != c):
                            agent = state.values.get("agent_name", "General")
                            restored.append({
                                "id": getattr(m, "id", None) or f"msg_{uuid.uuid4().hex[:12]}",
                                "role": "assistant",
                                "content": c,
                                "agent_name": agent,
                            })
                st.session_state["message_history"] = restored
        except Exception as exc:
            logger.debug("Could not auto-restore session messages: %s", exc)

    # Documents indexed in this session {composite_key: [metadata]}
    if "session_docs" not in st.session_state:
        st.session_state["session_docs"] = {}

    if "pending_upload_confirmation" not in st.session_state:
        st.session_state["pending_upload_confirmation"] = None

    # Search filter query
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    # Dialog & action trigger states
    for _key in (
        "_rename_tid",
        "_rename_current",
        "_delete_tid",
        "_delete_title",
        "_edit_msg_idx",
        "_edit_msg_content",
        "_trigger_edit",
        "_trigger_card_prompt",
        "_stop_generation",
        "_active_doubt",
        "_pending_doubt_query",
    ):
        if _key not in st.session_state:
            st.session_state[_key] = None


_init_session()

user_id = st.session_state["user_id"]
thread_id = st.session_state["thread_id"]
composite_key = f"{user_id}:{thread_id}"
current_theme = st.session_state.get("theme", "dark")


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic CSS (Dark / Light theme)
# ─────────────────────────────────────────────────────────────────────────────

def get_theme_css(theme: str) -> str:
    is_dark = (theme == "dark")

    # ── Orange + Coral + Peach colour tokens ───────────────────────────────
    # Primary palette (always orange/coral regardless of theme)
    primary       = "#FF6B35"
    primary_coral = "#FF8066"
    accent        = "#FFAA70"
    peach_light   = "#FFF0E6"

    # Theme-dependent surfaces
    if is_dark:
        bg_main          = "#17131A"
        bg_sidebar       = "linear-gradient(180deg, #1E1820 0%, #17131A 100%)"
        bg_sidebar_solid = "#1E1820"
        border_color     = "#2E2530"
        text_main        = "#F5EEE8"
        text_muted       = "#9E8E84"
        card_bg          = "#221C26"
        card_bg_alt      = "#251F29"
        card_hover       = "#2A2030"
        user_bubble      = "rgba(255,107,53,0.08)"
        user_border      = "rgba(255,107,53,0.25)"
        chat_input_bg    = "#221C26"
        chat_input_border= "#3A2F38"
        btn_text         = "#C8B8B0"
        btn_action       = "#7A6870"
        code_bg          = "#1A1420"
        shadow_primary   = "rgba(255,107,53,0.30)"
        shadow_card      = "rgba(0,0,0,0.35)"
        scrollbar_thumb  = "#3D3040"
        active_conv_bg   = "rgba(255,107,53,0.12)"
        active_conv_border= "rgba(255,107,53,0.35)"
    else:
        bg_main          = "#FFF8F3"
        bg_sidebar       = "linear-gradient(180deg, #FFFFFF 0%, #FFF5EE 100%)"
        bg_sidebar_solid = "#FFFFFF"
        border_color     = "#F0E0D6"
        text_main        = "#24202A"
        text_muted       = "#8A7570"
        card_bg          = "#FFFFFF"
        card_bg_alt      = "#FFF5EE"
        card_hover       = "#FFF0E6"
        user_bubble      = "#FFF0E6"
        user_border      = "#FFCBA8"
        chat_input_bg    = "#FFFFFF"
        chat_input_border= "#F0D8C8"
        btn_text         = "#5A4040"
        btn_action       = "#9A8070"
        code_bg          = "#FFF5EE"
        shadow_primary   = "rgba(255,107,53,0.20)"
        shadow_card      = "rgba(255,107,53,0.08)"
        scrollbar_thumb  = "#F0C8B0"
        active_conv_bg   = "#FFF0E6"
        active_conv_border= "#FF8066"

    max_w = "1420px" if st.session_state.get("_active_doubt") else "920px"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Reset & Base ─────────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; }}
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {bg_main} !important;
        color: {text_main} !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none !important; }}

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {bg_sidebar} !important;
        border-right: 1px solid {border_color} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {text_main} !important;
    }}
    /* Default sidebar buttons — thread list */
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        color: {btn_text};
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 13px;
        font-weight: 400;
        transition: all 0.18s ease;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {active_conv_bg} !important;
        border-color: {active_conv_border} !important;
        color: {primary} !important;
    }}

    /* ── New Chat button ─────────────────────────────────────────────── */
    .new-chat-btn > button {{
        background: linear-gradient(135deg, {primary} 0%, {primary_coral} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 11px 18px !important;
        transition: all 0.22s ease !important;
        box-shadow: 0 4px 18px {shadow_primary} !important;
        letter-spacing: 0.3px !important;
    }}
    .new-chat-btn > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px {shadow_primary} !important;
        filter: brightness(1.06) !important;
    }}
    .new-chat-btn > button:active {{
        transform: translateY(0) !important;
        filter: brightness(0.97) !important;
    }}

    /* ── Theme toggle buttons ─────────────────────────────────────────── */
    .theme-btn-active > button {{
        background: linear-gradient(135deg, {primary} 0%, {primary_coral} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        box-shadow: 0 2px 10px {shadow_primary} !important;
    }}
    .theme-btn-inactive > button {{
        background: transparent !important;
        color: {btn_action} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        font-size: 12px !important;
    }}
    .theme-btn-inactive > button:hover {{
        border-color: {primary} !important;
        color: {primary} !important;
        background: {user_bubble} !important;
    }}

    /* ── Active conversation item ─────────────────────────────────────── */
    .conv-active > button {{
        background: {active_conv_bg} !important;
        border-color: {active_conv_border} !important;
        color: {primary} !important;
        font-weight: 600 !important;
    }}

    /* ── Conversation rename / delete icon buttons ───────────────────── */
    .conv-action-btn > button {{
        background: transparent !important;
        border: none !important;
        color: {btn_action} !important;
        padding: 2px 6px !important;
        font-size: 12px !important;
        min-height: 0 !important;
        line-height: 1 !important;
        width: auto !important;
        border-radius: 4px !important;
        transition: all 0.15s ease !important;
    }}
    .conv-action-btn > button:hover {{
        color: {primary} !important;
        background: {user_bubble} !important;
    }}
    .conv-delete-btn > button:hover {{
        color: #EF4444 !important;
        background: rgba(239,68,68,0.10) !important;
    }}

    /* ── Sidebar section labels ───────────────────────────────────────── */
    .sidebar-section {{
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: {btn_action};
        padding: 10px 4px 5px;
        margin-top: 4px;
    }}

    /* ── Main content area ───────────────────────────────────────────── */
    [data-testid="stMainBlockContainer"], [data-testid="stMain"], .main .block-container {{
        background: {bg_main} !important;
        padding-top: 16px !important;
        max-width: {max_w} !important;
        margin: 0 auto !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        transition: max-width 0.25s ease !important;
    }}

    /* ── Hero / Welcome screen ───────────────────────────────────────── */
    .hero-header {{
        text-align: center;
        padding: 48px 20px 28px;
    }}
    .hero-logo-ring {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 72px;
        height: 72px;
        border-radius: 20px;
        background: linear-gradient(135deg, {primary} 0%, {primary_coral} 100%);
        box-shadow: 0 8px 32px {shadow_primary};
        font-size: 2rem;
        margin-bottom: 20px;
    }}
    .hero-title {{
        font-size: 2.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, {primary} 0%, {primary_coral} 55%, {accent} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.18;
        margin-bottom: 14px;
        letter-spacing: -0.5px;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: {text_muted};
        font-weight: 400;
        line-height: 1.6;
        max-width: 540px;
        margin: 0 auto;
    }}

    /* ── Capability / suggestion cards ───────────────────────────────── */
    .capability-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin: 28px 0 8px;
    }}
    .capability-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 20px;
        transition: all 0.22s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }}
    .capability-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {primary}, {primary_coral});
        opacity: 0;
        transition: opacity 0.22s ease;
        border-radius: 16px 16px 0 0;
    }}
    .capability-card:hover {{
        border-color: {primary_coral};
        background: {card_hover};
        transform: translateY(-3px);
        box-shadow: 0 10px 32px {shadow_card};
    }}
    .capability-card:hover::before {{ opacity: 1; }}
    .capability-icon {{
        font-size: 1.7rem;
        margin-bottom: 10px;
        display: block;
    }}
    .capability-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {text_main};
        margin-bottom: 5px;
    }}
    .capability-desc {{ font-size: 0.82rem; color: {text_muted}; line-height: 1.5; }}

    /* ── Interactive Suggestion Cards ───────────────────────────────── */
    .card-btn > button {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 16px !important;
        padding: 18px 20px !important;
        text-align: left !important;
        color: {text_main} !important;
        transition: all 0.22s ease !important;
        width: 100% !important;
        min-height: 96px !important;
        box-shadow: 0 2px 10px {shadow_card} !important;
        position: relative !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        white-space: pre-wrap !important;
        line-height: 1.45 !important;
    }}
    .card-btn > button:hover {{
        border-color: {primary} !important;
        background: {card_hover} !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 28px {shadow_primary} !important;
        color: {text_main} !important;
    }}
    .card-btn > button:active {{
        transform: translateY(0) !important;
    }}

    /* ── Chat messages ───────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        border: none !important;
        padding: 5px 0 !important;
        margin-bottom: 5px !important;
    }}

    /* Assistant messages stay open and readable without a card surface. */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    [data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) {{
        width: 100% !important;
        max-width: 100% !important;
        margin-right: auto !important;
        padding: 6px 0 !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) [data-testid="stChatMessageContent"] {{
        width: 100% !important;
        max-width: 840px !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
        color: {text_main} !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }}

    /* User messages use a text-sized bubble aligned to the right. */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    .stChatMessage.user,
    [data-testid="stChatMessage"][class*="user"],
    [data-testid="stChatMessage"][data-testid*="user"] {{
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 0 5px auto !important;
        padding: 4px 0 !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"],
    .stChatMessage.user [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"][class*="user"] [data-testid="stChatMessageContent"] {{
        flex: 1 1 auto !important;
        width: 100% !important;
        max-width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdown"] {{
        width: fit-content !important;
        max-width: min(78%, 680px) !important;
        align-self: flex-end !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {{
        width: fit-content !important;
        max-width: 100% !important;
        background: {user_bubble} !important;
        border: 1px solid {user_border} !important;
        border-radius: 18px 5px 18px 18px !important;
        padding: 9px 14px !important;
        color: {text_main} !important;
        box-shadow: 0 2px 8px {shadow_card} !important;
        font-size: 14px !important;
        line-height: 1.55 !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p {{
        margin-bottom: 0 !important;
        color: {text_main} !important;
    }}

    /* Compact edit button for user message */
    .user-edit-btn > button {{
        background: transparent !important;
        border: 1px solid transparent !important;
        color: {btn_action} !important;
        font-size: 11px !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        min-height: 0 !important;
        line-height: 1.2 !important;
        transition: all 0.15s ease !important;
        margin-top: 2px !important;
    }}
    .user-edit-btn > button:hover {{
        color: {primary} !important;
        background: {user_bubble} !important;
        border-color: {user_border} !important;
    }}

    /* ── Typography & HTML elements in responses ─────────────────────── */
    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3,
    [data-testid="stChatMessageContent"] h4 {{
        color: {text_main} !important;
        font-weight: 700 !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
        line-height: 1.3 !important;
    }}
    [data-testid="stChatMessageContent"] p {{
        margin-bottom: 9px !important;
        line-height: 1.7 !important;
    }}
    [data-testid="stChatMessageContent"] ul,
    [data-testid="stChatMessageContent"] ol {{
        margin-bottom: 12px !important;
        padding-left: 24px !important;
    }}
    [data-testid="stChatMessageContent"] li {{
        margin-bottom: 4px !important;
        line-height: 1.6 !important;
    }}
    [data-testid="stChatMessageContent"] table {{
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 12px 0 !important;
        font-size: 13.5px !important;
    }}
    [data-testid="stChatMessageContent"] th {{
        background: {card_bg_alt} !important;
        border: 1px solid {border_color} !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        text-align: left !important;
        color: {primary} !important;
    }}
    [data-testid="stChatMessageContent"] td {{
        border: 1px solid {border_color} !important;
        padding: 8px 12px !important;
        color: {text_main} !important;
    }}
    [data-testid="stChatMessageContent"] blockquote {{
        border-left: 3px solid {primary} !important;
        background: {card_bg_alt} !important;
        padding: 8px 14px !important;
        border-radius: 0 8px 8px 0 !important;
        margin: 10px 0 !important;
        color: {text_muted} !important;
        font-style: italic !important;
    }}

    /* ── Thinking Indicator Animation ────────────────────────────────── */
    @keyframes pulse-dot {{
        0%, 100% {{ transform: scale(0.85); opacity: 0.5; }}
        50% {{ transform: scale(1.2); opacity: 1; }}
    }}
    .thinking-dot {{
        width: 8px;
        height: 8px;
        background: linear-gradient(135deg, {primary}, {primary_coral});
        border-radius: 50%;
        display: inline-block;
        animation: pulse-dot 1.2s infinite ease-in-out;
    }}
    .thinking-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .thinking-dot:nth-child(3) {{ animation-delay: 0.4s; }}

    /* ── Code blocks & syntax highlighting ───────────────────────────── */
    code {{
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
        background: {code_bg} !important;
        color: {primary} !important;
        padding: 2px 6px !important;
        border-radius: 6px !important;
        font-size: 0.88em !important;
        border: 1px solid {border_color} !important;
    }}
    pre {{
        background: {code_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        overflow-x: auto !important;
    }}
    pre code {{
        background: transparent !important;
        color: {text_main} !important;
        padding: 0 !important;
        border: none !important;
    }}

    /* ── Message action toolbar ──────────────────────────────────────── */
    .msg-action-btn > button {{
        background: transparent !important;
        border: 1px solid {border_color} !important;
        color: {btn_action} !important;
        padding: 3px 10px !important;
        font-size: 11px !important;
        border-radius: 7px !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
        transition: all 0.15s ease !important;
        font-weight: 500 !important;
    }}
    .msg-action-btn > button:hover {{
        color: {primary} !important;
        border-color: {primary} !important;
        background: {user_bubble} !important;
    }}
    /* ── Floating Ask Doubt Button (Task 5) ────────────────────────── */
    [data-testid="stColumn"]:has(.doubt-btn) [data-testid="stButton"] > button {{
        background: rgba(255,107,53,0.12) !important;
        border: 1px solid rgba(255,107,53,0.35) !important;
        color: {primary} !important;
        padding: 3px 12px !important;
        font-size: 11.5px !important;
        border-radius: 7px !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
        font-weight: 600 !important;
        transition: all 0.18s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 4px !important;
    }}
    [data-testid="stColumn"]:has(.doubt-btn) [data-testid="stVerticalBlock"] {{
        align-items: flex-end !important;
    }}
    [data-testid="stColumn"]:has(.doubt-btn) [data-testid="stButton"] > button:hover {{
        background: linear-gradient(135deg, {primary} 0%, {primary_coral} 100%) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        box-shadow: 0 4px 14px {shadow_primary} !important;
        transform: translateY(-1px) !important;
    }}
    [data-testid="stColumn"]:has(.doubt-btn-active) [data-testid="stButton"] > button {{
        background: linear-gradient(135deg, {primary} 0%, {primary_coral} 100%) !important;
        border: 1px solid transparent !important;
        color: #FFFFFF !important;
        padding: 3px 12px !important;
        font-size: 11.5px !important;
        border-radius: 7px !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px {shadow_primary} !important;
    }}
    .doubt-close-btn > button {{
        background: transparent !important;
        border: 1px solid {border_color} !important;
        color: {text_muted} !important;
        padding: 2px 8px !important;
        font-size: 13px !important;
        border-radius: 6px !important;
        line-height: 1.2 !important;
        transition: all 0.15s ease !important;
    }}
    .doubt-close-btn > button:hover {{
        border-color: {primary} !important;
        color: {primary} !important;
        background: {user_bubble} !important;
    }}
    .stForm .stButton > button {{
        background: linear-gradient(135deg, {primary} 0%, {primary_coral} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px {shadow_primary} !important;
    }}
    .stForm .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px {shadow_primary} !important;
        filter: brightness(1.05) !important;
    }}
    .msg-toolbar {{
        display: flex;
        gap: 8px;
        margin-top: 8px;
        align-items: center;
        flex-wrap: wrap;
    }}

    /* ── Agent badges ───────────────────────────────────────────────── */
    .agent-badge {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }}
    .badge-general  {{ background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.28); }}
    .badge-research {{ background: rgba(59,130,246,0.12); color: #3b82f6; border: 1px solid rgba(59,130,246,0.28); }}
    .badge-rag      {{ background: rgba(255,107,53,0.12); color: {primary}; border: 1px solid rgba(255,107,53,0.28); }}
    .badge-coding   {{ background: rgba(168,85,247,0.12); color: #a855f7; border: 1px solid rgba(168,85,247,0.28); }}

    /* ── Chat input box ─────────────────────────────────────────────── */
    [data-testid="stChatInput"] {{
        background: {chat_input_bg} !important;
        border: 1.5px solid {chat_input_border} !important;
        border-radius: 16px !important;
        color: {text_main} !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 18px {shadow_card} !important;
    }}
    [data-testid="stChatInput"]:hover {{
        border-color: {accent} !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {primary} !important;
        box-shadow: 0 0 0 3px rgba(255,107,53,0.22), 0 6px 24px {shadow_primary} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: {text_main} !important;
        font-size: 14px !important;
    }}
    [data-testid="stChatInput"] button {{
        background: linear-gradient(135deg, {primary}, {primary_coral}) !important;
        border-radius: 10px !important;
        color: white !important;
        border: none !important;
        transition: all 0.18s ease !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        transform: scale(1.05) !important;
        filter: brightness(1.08) !important;
    }}
    [data-testid="stChatInput"] button:disabled {{
        background: {btn_action} !important;
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }}

    /* ── Contextual doubt layout ─────────────────────────────────────── */
    [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) {{
        align-items: flex-start !important;
        gap: 1rem !important;
    }}
    [data-testid="stColumn"]:has(.chat-scroll-marker) {{
        min-width: 0 !important;
    }}
    [data-testid="stColumn"]:has(.chat-scroll-marker) > [data-testid="stVerticalBlock"] {{
        height: calc(100dvh - 6rem) !important;
        max-height: calc(100dvh - 6rem) !important;
        overflow-y: auto !important;
        overscroll-behavior: contain !important;
        scrollbar-gutter: stable;
    }}
    [data-testid="stColumn"]:has(.doubt-panel-anchor) {{
        position: fixed !important;
        top: 0.75rem !important;
        right: 1rem !important;
        width: clamp(300px, calc((100vw - 320px) * 0.52), 560px) !important;
        bottom: max(7.5rem, env(safe-area-inset-bottom)) !important;
        height: auto !important;
        min-width: 0 !important;
        z-index: 20 !important;
    }}
    [data-testid="stColumn"]:has(.doubt-panel-anchor) > [data-testid="stVerticalBlock"] {{
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        overscroll-behavior: contain !important;
        scrollbar-gutter: stable;
    }}
    .st-key-doubt-panel-header {{
        position: sticky !important;
        top: 0 !important;
        z-index: 5 !important;
        background: {bg_main} !important;
        padding: 8px 0 !important;
        border-bottom: 1px solid {border_color} !important;
    }}
    .st-key-doubt-chat-history {{
        overscroll-behavior: contain !important;
        scrollbar-gutter: stable;
    }}
    .st-key-doubt-panel-form {{
        position: sticky !important;
        bottom: 0 !important;
        z-index: 4 !important;
        background: {bg_main} !important;
        padding: 8px 0 !important;
        border-top: 1px solid {border_color} !important;
    }}

    /* ── Status / tool indicator ─────────────────────────────────────── */
    [data-testid="stStatus"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stStatus"] [data-testid="stStatusLabel"] {{
        color: {text_main} !important;
    }}

    /* ── Expander (Sources panel) ────────────────────────────────────── */
    [data-testid="stExpander"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {text_main} !important;
    }}

    /* ── File uploader ───────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        background: {card_bg} !important;
        border: 1.5px dashed {user_border} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {primary} !important;
    }}

    /* ── Dividers & scrollbars ────────────────────────────────────────── */
    hr {{ border-color: {border_color} !important; }}
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: {scrollbar_thumb};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {primary}; }}

    /* ── Download buttons ─────────────────────────────────────────────── */
    [data-testid="stDownloadButton"] > button {{
        background: {card_bg_alt} !important;
        border: 1px solid {border_color} !important;
        color: {text_main} !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stDownloadButton"] > button:hover {{
        border-color: {primary} !important;
        color: {primary} !important;
        background: {user_bubble} !important;
    }}

    /* ── Caption & small text ─────────────────────────────────────────── */
    .stCaption {{ color: {text_muted} !important; font-size: 11.5px !important; }}

    /* ── Toast notifications ─────────────────────────────────────────── */
    [data-testid="stToast"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        color: {text_main} !important;
    }}

    /* ── Responsive ───────────────────────────────────────────────────── */
    @media (max-width: 768px) {{
        .capability-grid {{ grid-template-columns: 1fr; gap: 10px; }}
        .hero-title {{ font-size: 2.1rem; }}
        .hero-header {{ padding: 32px 12px 20px; }}
        [data-testid="stMain"], .main .block-container {{
            padding-left: 10px !important;
            padding-right: 10px !important;
        }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdown"] {{
            max-width: 88% !important;
        }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{
            font-size: 14.5px !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.doubt-panel-anchor) {{
            display: block !important;
        }}
        [data-testid="stColumn"]:has(.chat-scroll-marker) {{
            width: 100% !important;
        }}
        [data-testid="stColumn"]:has(.chat-scroll-marker) > [data-testid="stVerticalBlock"] {{
            height: calc(100dvh - 6rem) !important;
            max-height: calc(100dvh - 6rem) !important;
        }}
        [data-testid="stColumn"]:has(.doubt-panel-anchor) {{
            position: fixed !important;
            inset: max(8px, env(safe-area-inset-top)) 8px max(7.5rem, env(safe-area-inset-bottom)) !important;
            width: auto !important;
            height: auto !important;
            z-index: 1000 !important;
            padding: 10px !important;
            border: 1px solid {border_color} !important;
            border-radius: 14px !important;
            background: {bg_main} !important;
            box-shadow: 0 12px 40px {shadow_card} !important;
        }}
        [data-testid="stColumn"]:has(.doubt-panel-anchor) > [data-testid="stVerticalBlock"] {{
            height: 100% !important;
            max-height: 100% !important;
        }}
    }}
    @media (max-width: 480px) {{
        .hero-title {{ font-size: 1.7rem; }}
        .hero-logo-ring {{ width: 56px; height: 56px; font-size: 1.5rem; }}
        .capability-card {{ padding: 14px; }}
    }}
    </style>
    """

st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Agent badge helper
# ─────────────────────────────────────────────────────────────────────────────

AGENT_BADGES = {
    "General":  ("badge-general",  "💬 General"),
    "Research": ("badge-research", "🔍 Research"),
    "RAG":      ("badge-rag",      "📄 Document"),
    "Coding":   ("badge-coding",   "💻 Coding"),
}

def render_agent_badge(agent_name: str | None) -> str:
    if not agent_name:
        return ""
    css_class, label = AGENT_BADGES.get(agent_name, ("badge-general", agent_name))
    return f'<span class="agent-badge {css_class}">{label}</span>'


# Friendly icon + label per tool, shown while a tool is executing.
TOOL_STATUS_LABELS = {
    "duckduckgo_search": ("🌐", "Searching the web"),
    "rag_tool": ("📄", "Searching the document"),
    "calculator": ("🧮", "Calculating"),
    "get_stock_price": ("📈", "Fetching stock price"),
}


def tool_status_label(tool_name: str) -> str:
    icon, label = TOOL_STATUS_LABELS.get(tool_name, ("🔧", f"Using `{tool_name}`"))
    return f"{icon} {label}…"


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def new_chat():
    """Start a fresh conversation thread."""
    new_tid = str(uuid.uuid4())
    st.session_state["thread_id"] = new_tid
    st.session_state["message_history"] = []
    st.session_state["pending_upload_confirmation"] = None
    st.session_state["thread_titles"][new_tid] = "New Chat"
    st.session_state["_active_doubt"] = None


def switch_thread(tid: str):
    """Switch to an existing thread and restore its messages without duplication."""
    st.session_state["thread_id"] = tid
    st.session_state["_active_doubt"] = None
    # Load from LangGraph checkpointer
    config = make_config(user_id, tid)
    try:
        state = chatbot.get_state(config=config)
        raw_msgs = state.values.get("messages", [])
        restored = []
        for msg in raw_msgs:
            if isinstance(msg, HumanMessage):
                c = clean_markdown_content(str(msg.content or "")).strip()
                if c and (not restored or restored[-1].get("role") != "user" or restored[-1].get("content") != c):
                    restored.append({
                        "id": getattr(msg, "id", None) or f"msg_{uuid.uuid4().hex[:12]}",
                        "role": "user",
                        "content": c,
                        "agent_name": None,
                    })
            elif isinstance(msg, AIMessage) and msg.content:
                tool_calls = getattr(msg, "tool_calls", [])
                c = clean_markdown_content(str(msg.content or "")).strip()
                if tool_calls and not c:
                    continue
                if c and (not restored or restored[-1].get("role") != "assistant" or restored[-1].get("content") != c):
                    agent = state.values.get("agent_name", "General")
                    restored.append({
                        "id": getattr(msg, "id", None) or f"msg_{uuid.uuid4().hex[:12]}",
                        "role": "assistant",
                        "content": c,
                        "agent_name": agent,
                    })
        st.session_state["message_history"] = restored
    except Exception as exc:
        logger.warning("Could not restore thread %s: %s", tid, exc)
        st.session_state["message_history"] = []


def auto_title(text: str) -> str:
    """Generate a short conversation title from the first message."""
    cleaned = text.strip()
    if len(cleaned) <= 40:
        return cleaned
    return cleaned[:37] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Rename / Delete dialog definitions (must be top-level, before sidebar)
# ─────────────────────────────────────────────────────────────────────────────

@st.dialog("Rename Conversation")
def _rename_dialog():
    """Modal dialog to rename a conversation and persist the new title."""
    tid = st.session_state.get("_rename_tid", "")
    current = st.session_state.get("_rename_current", "")
    new_title = st.text_input("New name", value=current, max_chars=120)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", type="primary", use_container_width=True):
            new_title = new_title.strip()
            if not new_title:
                st.error("Title cannot be empty.")
            else:
                try:
                    save_thread_title(user_id, tid, new_title)
                    st.session_state["thread_titles"][tid] = new_title
                    st.session_state["_rename_tid"] = None
                    st.session_state["_rename_current"] = None
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state["_rename_tid"] = None
            st.session_state["_rename_current"] = None
            st.rerun()


@st.dialog("Delete Conversation")
def _delete_dialog():
    """Modal confirmation dialog to permanently delete a conversation."""
    tid = st.session_state.get("_delete_tid", "")
    title = st.session_state.get("_delete_title", "this conversation")
    st.warning(
        f"Permanently delete **\"{title}\"**?\n\n"
        "This cannot be undone. All messages and uploaded documents will be removed."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Delete", type="primary", use_container_width=True):
            # Remove from SQLite (checkpoints + writes + thread_titles)
            try:
                delete_thread(user_id, tid)
            except Exception as exc:
                st.error(f"Database error: {exc}")
                return
            # Remove FAISS index from disk and in-memory caches
            delete_vector_store(user_id, tid)
            # Update session state
            composite_del = f"{user_id}:{tid}"
            st.session_state["thread_titles"].pop(tid, None)
            st.session_state["session_docs"].pop(composite_del, None)
            # If the deleted thread was the active one, open a fresh thread
            if st.session_state["thread_id"] == tid:
                new_tid = str(uuid.uuid4())
                st.session_state["thread_id"] = new_tid
                st.session_state["message_history"] = []
                st.session_state["thread_titles"][new_tid] = "New Chat"
            st.session_state["_active_doubt"] = None
            st.session_state["_delete_tid"] = None
            st.session_state["_delete_title"] = None
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state["_delete_tid"] = None
            st.session_state["_delete_title"] = None
            st.rerun()


@st.dialog("Edit Message")
def _edit_dialog():
    """Modal dialog to edit a previous user message and re-branch the conversation."""
    idx = st.session_state.get("_edit_msg_idx")
    current_text = st.session_state.get("_edit_msg_content", "")
    new_text = st.text_area("Edit your message:", value=current_text, height=130)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save & Resubmit", type="primary", use_container_width=True):
            cleaned = new_text.strip()
            if not cleaned:
                st.error("Message cannot be empty.")
            else:
                # Count HumanMessages in message_history up to idx
                human_count = sum(
                    1 for m in st.session_state["message_history"][:idx] if m.get("role") == "user"
                )
                rewind_for_edit(user_id, thread_id, human_count)
                # Slice message_history up to this message
                st.session_state["message_history"] = st.session_state["message_history"][:idx]
                st.session_state["_trigger_edit"] = cleaned
                st.session_state["_active_doubt"] = None
                st.session_state["_edit_msg_idx"] = None
                st.session_state["_edit_msg_content"] = None
                st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state["_edit_msg_idx"] = None
            st.session_state["_edit_msg_content"] = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Logo / Branding ──
    st.markdown("""
    <div style="padding: 16px 4px 12px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="
                width:40px; height:40px; border-radius:12px;
                background: linear-gradient(135deg, #FF6B35, #FF8066);
                display:flex; align-items:center; justify-content:center;
                font-size:1.2rem; box-shadow: 0 4px 14px rgba(255,107,53,0.35);
                flex-shrink:0;">🤖</div>
            <div>
                <div style="font-size:1.1rem; font-weight:800; background: linear-gradient(135deg, #FF6B35, #FF8066); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:-0.3px;">AI Assistant</div>
                <div style="font-size:10px; color:#9E8E84; margin-top:1px; font-weight:500; letter-spacing:0.5px; text-transform:uppercase;">Multi-Agent Research</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Toggle ──
    col_th1, col_th2 = st.columns(2)
    with col_th1:
        dark_active = (current_theme == "dark")
        dark_cls = "theme-btn-active" if dark_active else "theme-btn-inactive"
        st.markdown(f'<div class="{dark_cls}">', unsafe_allow_html=True)
        if st.button(
            "🌙 Dark",
            use_container_width=True,
            key="theme_dark_btn",
        ):
            if not dark_active:
                st.session_state["theme"] = "dark"
                set_user_setting(user_id, "theme", "dark")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_th2:
        light_active = (current_theme == "light")
        light_cls = "theme-btn-active" if light_active else "theme-btn-inactive"
        st.markdown(f'<div class="{light_cls}">', unsafe_allow_html=True)
        if st.button(
            "☀️ Light",
            use_container_width=True,
            key="theme_light_btn",
        ):
            if not light_active:
                st.session_state["theme"] = "light"
                set_user_setting(user_id, "theme", "light")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── New Chat ──
    st.markdown('<div class="new-chat-btn" style="margin-top:10px;">', unsafe_allow_html=True)
    if st.button("✏️  New Chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Search Conversations ──
    search_query = st.text_input(
        "Search conversations",
        value=st.session_state.get("search_query", ""),
        placeholder="🔍 Search title or text…",
        label_visibility="collapsed",
        key="chat_search_input",
    )
    st.session_state["search_query"] = search_query

    # ── Document Upload ──
    st.markdown('<div class="sidebar-section">📎 Document Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="doc_uploader",
    )

    session_docs = st.session_state["session_docs"]
    current_docs = session_docs.get(composite_key, [])
    if not current_docs:
        persisted_meta = get_retriever_meta(user_id, thread_id)
        if persisted_meta:
            session_docs[composite_key] = list(persisted_meta)
            current_docs = session_docs[composite_key]

    if uploaded_files:
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            already_loaded = any(d.get("filename") == filename for d in current_docs)
            if already_loaded:
                continue

            with st.status(f"Indexing `{filename}`…", expanded=True):
                try:
                    meta = ingest_document(
                        file_bytes=uploaded_file.getvalue(),
                        user_id=user_id,
                        thread_id=thread_id,
                        filename=filename,
                    )
                    if composite_key not in session_docs:
                        session_docs[composite_key] = []
                    session_docs[composite_key] = [
                        d for d in session_docs[composite_key] if d.get("filename") != meta["filename"]
                    ]
                    session_docs[composite_key].append(meta)
                    current_docs = session_docs[composite_key]
                    st.session_state["pending_upload_confirmation"] = meta["filename"]

                    ext = meta.get("doc_type", "txt")
                    icon = "📄" if ext == "pdf" else ("📑" if ext == "docx" else ("🗒️" if ext == "md" else "📝"))
                    st.write(f"✅ {icon} **{meta['filename']}**")
                    if ext == "pdf":
                        pages_info = f"{meta.get('non_empty_pages', meta.get('documents', 1))} usable / {meta.get('documents', 1)} total pages"
                        st.write(f"   {pages_info} · {meta.get('chunks', 0)} chunks")
                    else:
                        st.write(f"   {meta.get('chunks', 0)} chunks")
                    st.success("Ready for search and questions!")
                except Exception as exc:
                    st.error(f"Failed to index `{filename}`: {exc}")

    # Show indexed docs for current thread
    if current_docs:
        st.markdown('<div class="sidebar-section" style="margin-top:8px;">📚 Indexed Documents</div>', unsafe_allow_html=True)
        for doc in current_docs:
            fname = doc.get("filename", "document")
            ext = doc.get("doc_type") or (fname.rsplit(".", 1)[-1].lower() if "." in fname else "txt")
            icon = "📄" if ext == "pdf" else ("📑" if ext == "docx" else ("🗒️" if ext == "md" else "📝"))
            chunks = doc.get("chunks", 0)
            st.caption(f"{icon} **{fname}** ({chunks} chunks)")

    st.markdown("---")

    # ── Conversation History ──
    st.markdown('<div class="sidebar-section">💬 Conversations</div>', unsafe_allow_html=True)

    thread_titles = st.session_state["thread_titles"]
    all_rev = list(reversed(list(thread_titles.keys())))

    if search_query.strip():
        matching_tids = search_user_threads(user_id, search_query)
        sorted_threads = [t for t in all_rev if t in matching_tids]
        st.caption(f"Found {len(sorted_threads)} match(es)")
    else:
        sorted_threads = all_rev

    if not sorted_threads:
        msg = "No matching conversations found." if search_query.strip() else "No conversations yet. Start a New Chat!"
        st.markdown(
            f'<div style="color:#64748b;font-size:12px;padding:8px 4px;">{msg}</div>',
            unsafe_allow_html=True,
        )

    for tid in sorted_threads:
        title = thread_titles.get(tid, "Chat")
        is_active = tid == st.session_state["thread_id"]
        display_title = f"▶ {title}" if is_active else title

        col_title, col_rename, col_delete = st.columns([6, 1, 1])
        with col_title:
            active_cls = "conv-active" if is_active else ""
            st.markdown(f'<div class="{active_cls}">', unsafe_allow_html=True)
            if st.button(display_title, key=f"thread_btn_{tid}", use_container_width=True):
                if not is_active:
                    switch_thread(tid)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col_rename:
            st.markdown('<div class="conv-action-btn">', unsafe_allow_html=True)
            if st.button("✏️", key=f"rename_btn_{tid}", help="Rename conversation"):
                st.session_state["_rename_tid"] = tid
                st.session_state["_rename_current"] = thread_titles.get(tid, "")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_delete:
            st.markdown('<div class="conv-action-btn conv-delete-btn">', unsafe_allow_html=True)
            if st.button("🗑️", key=f"delete_btn_{tid}", help="Delete conversation"):
                st.session_state["_delete_tid"] = tid
                st.session_state["_delete_title"] = thread_titles.get(tid, "Chat")
            st.markdown('</div>', unsafe_allow_html=True)

    # Trigger dialogs if requested
    if st.session_state.get("_rename_tid"):
        _rename_dialog()
    if st.session_state.get("_delete_tid"):
        _delete_dialog()
    if st.session_state.get("_edit_msg_idx") is not None:
        _edit_dialog()

    # ── Export Options ──
    if st.session_state.get("message_history"):
        st.markdown("---")
        st.markdown('<div class="sidebar-section">📥 Export Chat</div>', unsafe_allow_html=True)
        curr_title = thread_titles.get(thread_id, "conversation")
        safe_fn = re.sub(r'[^a-zA-Z0-9_\-]', '_', curr_title).strip('_') or "chat"
        col_md, col_txt = st.columns(2)
        with col_md:
            st.download_button(
                "📄 .MD",
                data=export_to_markdown(st.session_state["message_history"], curr_title),
                file_name=f"{safe_fn}.md",
                mime="text/markdown",
                use_container_width=True,
                key="export_md_sidebar_btn",
            )
        with col_txt:
            st.download_button(
                "📝 .TXT",
                data=export_to_txt(st.session_state["message_history"], curr_title),
                file_name=f"{safe_fn}.txt",
                mime="text/plain",
                use_container_width=True,
                key="export_txt_sidebar_btn",
            )

    st.markdown("---")

    # ── System Info ──
    st.markdown('<div class="sidebar-section">⚙️ System</div>', unsafe_allow_html=True)
    st.caption(f"Model: `{settings.groq_model}`")
    st.caption(f"Embeddings: `all-MiniLM-L6-v2`")

    from tools.mcp_tools import load_mcp_tools
    mcp = load_mcp_tools()
    if mcp:
        st.caption(f"MCP tools: {len(mcp)} loaded ✅")
    else:
        st.caption("MCP: not connected ⚠️")

    st.caption(f"Session: `{user_id[-8:]}`")
    st.markdown("""
    <div style="padding: 10px 4px 4px; font-size:10px; color:#9E8E84; text-align:center;">
        🟠 Multi-Agent AI Research Assistant
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

# Re-read current values (may have changed after sidebar interaction)
thread_id = st.session_state["thread_id"]
composite_key = f"{user_id}:{thread_id}"
message_history = st.session_state["message_history"]

# Validate active doubt state
active_doubt = st.session_state.get("_active_doubt")
if active_doubt is not None:
    d_idx = active_doubt.get("message_index")
    if (
        d_idx is None
        or d_idx >= len(message_history)
        or message_history[d_idx].get("role") != "assistant"
    ):
        st.session_state["_active_doubt"] = None
        active_doubt = None

# Create side-by-side columns if contextual doubt window is open
if active_doubt is not None:
    col_main, col_doubt = st.columns([0.55, 0.45], gap="medium")
else:
    col_main = st.container()
    col_doubt = None

# Evaluate user inputs and triggers
trigger_edit = st.session_state.pop("_trigger_edit", None)
trigger_card = st.session_state.pop("_trigger_card_prompt", None)

is_generating = st.session_state.get("_is_generating", False)
user_input = st.chat_input(
    "Ask anything about research, documents, code, or general questions...",
    disabled=is_generating,
)

prompt_to_stream = None
if trigger_edit:
    prompt_to_stream = trigger_edit
    append_user_message(prompt_to_stream)
elif trigger_card:
    prompt_to_stream = trigger_card
    if not message_history:
        auto_t = auto_title(trigger_card)
        st.session_state["thread_titles"][thread_id] = auto_t
        try:
            save_thread_title(user_id, thread_id, auto_t)
        except Exception:
            pass
    append_user_message(prompt_to_stream)
elif user_input and not is_generating:
    cleaned_input = clean_markdown_content(user_input).strip()
    if cleaned_input:
        prompt_to_stream = cleaned_input
        if not message_history:
            auto_t = auto_title(cleaned_input)
            st.session_state["thread_titles"][thread_id] = auto_t
            try:
                save_thread_title(user_id, thread_id, auto_t)
            except Exception:
                pass
        append_user_message(cleaned_input)

# ── Render Main Chat Column ──
with col_main:
    if active_doubt is not None:
        st.markdown('<span class="chat-scroll-marker" aria-hidden="true"></span>', unsafe_allow_html=True)

    # ── Welcome screen (empty chat) ──
    if not message_history:
        st.markdown("""
        <div class="hero-header">
            <div class="hero-logo-ring">🤖</div>
            <div class="hero-title">AI Research Assistant</div>
            <div class="hero-subtitle">
                Powered by Multi-Agent Intelligence · Groq · LangGraph<br>
                Select a starter prompt below or ask any question to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown('<div class="card-btn">', unsafe_allow_html=True)
            if st.button(
                "🔍  **Web Research**\n\nSearch live web, news, stock quotes & current data in real time",
                key="card_btn_research",
                use_container_width=True,
            ):
                st.session_state["_trigger_card_prompt"] = "Search the web for the latest breakthroughs in multi-agent AI systems and summarize key trends."
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card-btn" style="margin-top:12px;">', unsafe_allow_html=True)
            if st.button(
                "💻  **Coding Assistant**\n\nGenerate, debug, review and explain code in any language",
                key="card_btn_coding",
                use_container_width=True,
            ):
                st.session_state["_trigger_card_prompt"] = "Write a Python script for an asynchronous task queue with retry logic and error handling."
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c_col2:
            st.markdown('<div class="card-btn">', unsafe_allow_html=True)
            doc_card_label = (
                "📄  **Document Q&A**\n\nAsk questions about your uploaded documents"
                if current_docs
                else "📄  **Document Q&A**\n\nUpload a PDF, DOCX, TXT or MD in sidebar to ask questions"
            )
            if st.button(
                doc_card_label,
                key="card_btn_doc_qa",
                use_container_width=True,
            ):
                if current_docs:
                    st.session_state["_trigger_card_prompt"] = "Summarize the key topics and findings across my uploaded documents."
                else:
                    st.session_state["_trigger_card_prompt"] = "How does multi-document RAG work and what file types can I upload?"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card-btn" style="margin-top:12px;">', unsafe_allow_html=True)
            if st.button(
                "💬  **General Chat**\n\nAsk anything — from timeless knowledge to creative reasoning",
                key="card_btn_general",
                use_container_width=True,
            ):
                st.session_state["_trigger_card_prompt"] = "Explain how multi-agent LLM systems with a supervisor agent compare to single-prompt models."
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Attachment / Document Context Indicator
    if current_docs:
        doc_names = ", ".join(f"`{d.get('filename')}`" for d in current_docs[:3])
        extra = f" +{len(current_docs) - 3} more" if len(current_docs) > 3 else ""
        st.markdown(
            f'<div style="font-size:11.5px; color:#FF8066; margin-bottom:8px; display:flex; align-items:center; gap:6px;">'
            f'<span>📎</span> <span><b>{len(current_docs)} document(s) active:</b> {doc_names}{extra}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Handle interrupted / stopped generation from previous run ──
    if st.session_state.get("_is_generating"):
        st.session_state["_is_generating"] = False
        partial_text = st.session_state.get("_current_stream_text", "").strip()
        partial_agent = st.session_state.get("_current_stream_agent", "General")
        if partial_text:
            append_assistant_message(
                partial_text + "\n\n*(Generation stopped by user)*",
                agent_name=partial_agent,
            )
        st.session_state["_current_stream_text"] = ""
        st.toast("⏹️ Generation stopped.")

    # ── Render existing messages ──
    for idx, msg in enumerate(message_history):
        role = msg["role"]
        content = clean_markdown_content(msg.get("content", ""))
        agent_name = msg.get("agent_name")

        with st.chat_message(role):
            if role == "assistant" and agent_name:
                st.markdown(render_agent_badge(agent_name), unsafe_allow_html=True)
            st.markdown(content)

            if role == "user":
                _, col_u_edit = st.columns([0.88, 0.12])
                with col_u_edit:
                    st.markdown('<div class="user-edit-btn">', unsafe_allow_html=True)
                    if st.button("✏️ Edit", key=f"edit_btn_{idx}", help="Edit this message and re-branch the conversation"):
                        st.session_state["_edit_msg_idx"] = idx
                        st.session_state["_edit_msg_content"] = content
                        st.session_state["_active_doubt"] = None
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            elif role == "assistant":
                is_doubt_active = (
                    active_doubt is not None and active_doubt.get("message_index") == idx
                )
                doubt_cls = "doubt-btn-active" if is_doubt_active else "doubt-btn"
                doubt_label = "💭 Active" if is_doubt_active else "💬 Ask Doubt"

                _, col_d = st.columns([0.72, 0.28])
                with col_d:
                    st.markdown(f'<div class="{doubt_cls}">', unsafe_allow_html=True)
                    if st.button(doubt_label, key=f"doubt_btn_{idx}", help="Ask a doubt about this response"):
                        if is_doubt_active:
                            st.session_state["_active_doubt"] = None
                        else:
                            doubt_meta = get_doubt_meta(user_id, thread_id, idx)
                            if doubt_meta and doubt_meta.get("original_response") == content:
                                orig_prompt = doubt_meta.get("original_prompt", "")
                            else:
                                orig_prompt = ""
                                for p_i in range(idx - 1, -1, -1):
                                    if message_history[p_i].get("role") == "user":
                                        orig_prompt = str(message_history[p_i].get("content", ""))
                                        break
                                if doubt_meta:
                                    clear_doubt_messages(user_id, thread_id, idx)
                                save_doubt_meta(user_id, thread_id, idx, orig_prompt, content)
                            st.session_state["_active_doubt"] = {
                                "message_index": idx,
                                "original_prompt": orig_prompt,
                                "original_response": content,
                            }
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── Stream new response if user submitted input ──
    if prompt_to_stream:
        pending_filename = st.session_state.get("pending_upload_confirmation")
        if not pending_filename and current_docs:
            pending_filename = current_docs[-1].get("filename")
        normalized_input = prompt_to_stream.strip().lower().rstrip(".!?")
        if pending_filename and normalized_input in {
            "i have uploaded",
            "i uploaded",
            "uploaded",
            "i uploaded files",
            "i uploaded the document",
            "uploaded documents",
        }:
            st.session_state["pending_upload_confirmation"] = None
            num_docs = len(current_docs)
            if num_docs > 1:
                doc_names = ", ".join(f"`{d.get('filename')}`" for d in current_docs)
                confirmation = (
                    f"✅ **{num_docs} documents indexed and ready** ({doc_names}).\n\n"
                    "You can ask questions comparing them, finding specific facts, or summarizing across documents."
                )
            else:
                confirmation = (
                    f"✅ **{pending_filename}** is uploaded and ready.\n\n"
                    "Try asking: *What is the main topic?* or *Summarize the key findings.*"
                )
            confirmation = clean_markdown_content(confirmation)
            with st.chat_message("assistant"):
                st.markdown(render_agent_badge("RAG"), unsafe_allow_html=True)
                st.markdown(confirmation)
            append_assistant_message(confirmation, agent_name="RAG")
            st.rerun()

        st.session_state["pending_upload_confirmation"] = None
        config = make_config(user_id, thread_id)

        with st.chat_message("assistant"):
            stop_placeholder = st.empty()
            with stop_placeholder:
                st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
                if st.button("⏹️ Stop Generating", key="stop_gen_active_btn", type="secondary"):
                    pass
                st.markdown('</div>', unsafe_allow_html=True)

            status_holder = {"box": None, "error": False}
            sources_holder = {"web": [], "rag_sources": {}}
            badge_placeholder = st.empty()
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; padding:6px 0;">
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                    <span style="color:#FF8066; font-size:13px; font-weight:600; margin-left:4px;">AI Assistant is thinking…</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.session_state["_is_generating"] = True
            st.session_state["_current_stream_text"] = ""
            st.session_state["_current_stream_agent"] = "General"

            def _parse_tool_payload(raw_content) -> dict | None:
                if not isinstance(raw_content, str):
                    return None
                try:
                    payload = json.loads(raw_content)
                    return payload if isinstance(payload, dict) else None
                except (ValueError, TypeError):
                    return None

            def stream_response():
                first_token = True
                try:
                    for message_chunk, metadata in chatbot.stream(
                        {
                            "messages": [HumanMessage(content=prompt_to_stream)],
                            "user_id": user_id,
                            "thread_id": thread_id,
                            "next_agent": None,
                            "agent_name": None,
                            "documents": None,
                        },
                        config=config,
                        stream_mode="messages",
                    ):
                        node_name = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
                        if node_name == "supervisor":
                            continue

                        if isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool")
                            label = tool_status_label(tool_name)
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(label, expanded=True)
                            else:
                                status_holder["box"].update(label=label, state="running", expanded=True)

                            payload = _parse_tool_payload(message_chunk.content)
                            if payload:
                                if payload.get("error"):
                                    status_holder["box"].write(f"⚠️ {payload['error']}")
                                elif tool_name == "duckduckgo_search" and payload.get("results"):
                                    results = payload["results"]
                                    status_holder["box"].write(f"Found {len(results)} result(s)")
                                    for r in results[:5]:
                                        if r.get("title") and r.get("link"):
                                            entry = {"title": r["title"], "link": r["link"]}
                                            if entry not in sources_holder["web"]:
                                                sources_holder["web"].append(entry)
                                elif tool_name == "rag_tool" and payload.get("passages"):
                                    passages = payload["passages"]
                                    status_holder["box"].write(f"Retrieved {len(passages)} passage(s)")
                                    for p in passages:
                                        src = p.get("source") or "Document"
                                        if src not in sources_holder["rag_sources"]:
                                            sources_holder["rag_sources"][src] = set()
                                        if p.get("page"):
                                            sources_holder["rag_sources"][src].add(p["page"])

                        if isinstance(message_chunk, AIMessage) and message_chunk.content:
                            tool_calls = getattr(message_chunk, "tool_calls", [])
                            if tool_calls and not str(message_chunk.content).strip():
                                continue
                            if first_token:
                                thinking_placeholder.empty()
                                first_token = False
                            chunk_str = str(message_chunk.content)
                            st.session_state["_current_stream_text"] += chunk_str
                            yield chunk_str

                except Exception as exc:
                    logger.error("Graph stream error: %s", exc)
                    thinking_placeholder.empty()
                    status_holder["error"] = True
                    if status_holder["box"] is not None:
                        status_holder["box"].update(state="error", expanded=True)
                    yield f"\n\n⚠️ Something went wrong while generating a response: {exc}"

            ai_response = st.write_stream(stream_response())
            thinking_placeholder.empty()
            stop_placeholder.empty()
            st.session_state["_is_generating"] = False
            st.session_state["_current_stream_text"] = ""

            if status_holder["box"] is not None:
                if status_holder["error"]:
                    status_holder["box"].update(label="⚠️ Tool error", state="error", expanded=True)
                else:
                    status_holder["box"].update(
                        label="✅ Tool finished", state="complete", expanded=False
                    )

            try:
                final_state = chatbot.get_state(config=config)
                agent_name = final_state.values.get("agent_name", "General")
            except Exception:
                agent_name = "General"

            badge_placeholder.markdown(render_agent_badge(agent_name), unsafe_allow_html=True)

            if sources_holder["web"] or sources_holder["rag_sources"]:
                with st.expander(f"📚 Sources & Citations ({len(sources_holder['web']) + len(sources_holder['rag_sources'])})", expanded=False):
                    if sources_holder["web"]:
                        st.markdown("<div style='font-size:11px; font-weight:700; color:#FF8066; text-transform:uppercase; margin-bottom:6px;'>🌐 Web Sources</div>", unsafe_allow_html=True)
                        for s in sources_holder["web"]:
                            st.markdown(f"- [{s['title']}]({s['link']})")
                    if sources_holder["rag_sources"]:
                        st.markdown("<div style='font-size:11px; font-weight:700; color:#FF8066; text-transform:uppercase; margin-top:8px; margin-bottom:6px;'>📄 Document Passages</div>", unsafe_allow_html=True)
                        for src, pages in sources_holder["rag_sources"].items():
                            ext = src.rsplit(".", 1)[-1].lower() if "." in src else ""
                            icon = "📄" if ext == "pdf" else ("📑" if ext == "docx" else ("🗒️" if ext == "md" else "📝"))
                            if pages:
                                pages_str = ", ".join(f"page {p}" for p in sorted(pages))
                                st.markdown(f"- {icon} **{src}** ({pages_str})")
                            else:
                                st.markdown(f"- {icon} **{src}**")

        clean_ai_response = clean_markdown_content(ai_response or "")
        append_assistant_message(clean_ai_response, agent_name=agent_name)
        st.rerun()

# ── Render Right-Side Contextual Doubt Window (Task 5) ──
if col_doubt is not None and active_doubt is not None:
    with col_doubt:
        st.markdown('<span class="doubt-panel-anchor" aria-hidden="true"></span>', unsafe_allow_html=True)
        d_idx = active_doubt["message_index"]
        d_prompt = active_doubt.get("original_prompt", "")
        d_resp = active_doubt.get("original_response", "")

        # Top Header Bar
        with st.container(key="doubt-panel-header", border=False):
            col_dh_title, col_dh_close = st.columns([0.84, 0.16])
            with col_dh_title:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                        <span style="font-size:1.15rem;">💭</span>
                        <span style="font-weight:700; font-size:1.05rem; background: linear-gradient(135deg, #FF6B35, #FF8066); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">Ask Doubt</span>
                        <span style="font-size:11px; background:rgba(255,107,53,0.15); color:#FF6B35; border:1px solid rgba(255,107,53,0.3); border-radius:6px; padding:2px 6px; font-weight:600;">Response #{d_idx + 1}</span>
                    </div>
                    <div style="font-size:11px; color:#9E8E84; margin-bottom:8px;">
                        Ask targeted questions about this specific response
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_dh_close:
                st.markdown('<div class="doubt-close-btn">', unsafe_allow_html=True)
                if st.button("✕", key="close_doubt_panel_btn", help="Close doubt window"):
                    st.session_state["_active_doubt"] = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # Context Card
        with st.expander("📌 Context Excerpt", expanded=False):
            if d_prompt:
                st.markdown(f"**Your Question:**\n> {d_prompt}")
            resp_preview = d_resp[:280] + "…" if len(d_resp) > 280 else d_resp
            st.markdown(f"**Original Response Excerpt:**\n> {resp_preview}")

        # Retrieve doubt messages from SQLite
        doubt_messages = get_doubt_messages(user_id, thread_id, d_idx)

        # Scrollable container for doubt conversation
        doubt_container = st.container(height=240, key="doubt-chat-history")
        with doubt_container:
            if not doubt_messages:
                st.markdown(
                    """
                    <div style="text-align:center; padding:28px 12px; color:#9E8E84; font-size:12.5px;">
                        <div style="font-size:1.8rem; margin-bottom:8px;">🤔</div>
                        <b>What would you like clarified?</b><br>
                        Ask for an explanation of any concept, request an example, or ask to break it down step-by-step.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                q_c1, q_c2 = st.columns(2)
                with q_c1:
                    if st.button("💡 Simplify this", key="doubt_chip_simplify", use_container_width=True):
                        st.session_state["_pending_doubt_query"] = "Can you simplify and explain the key point of this response in layman's terms?"
                        st.rerun()
                with q_c2:
                    if st.button("📝 Give an example", key="doubt_chip_example", use_container_width=True):
                        st.session_state["_pending_doubt_query"] = "Can you provide a concrete real-world example to illustrate this response?"
                        st.rerun()
            else:
                for d_msg in doubt_messages:
                    d_role = d_msg.get("role", "user")
                    d_text = d_msg.get("content", "")
                    with st.chat_message(d_role):
                        st.markdown(d_text)

        # Clear doubt history button
        if doubt_messages:
            col_d_clear, _ = st.columns([0.45, 0.55])
            with col_d_clear:
                st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
                if st.button("🗑️ Clear doubts", key=f"clear_doubt_hist_{d_idx}", help="Clear doubt conversation for this response"):
                    clear_doubt_messages(user_id, thread_id, d_idx)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        pending_doubt = st.session_state.pop("_pending_doubt_query", None)

        with st.container(key="doubt-panel-form", border=False):
            with st.form(key=f"doubt_form_{d_idx}", clear_on_submit=True):
                f_col_in, f_col_btn = st.columns([0.68, 0.32])
                with f_col_in:
                    new_doubt_text = st.text_input(
                        "Your doubt",
                        placeholder="Ask a doubt about this response...",
                        label_visibility="collapsed",
                        key=f"doubt_text_input_{d_idx}",
                    )
                with f_col_btn:
                    submit_doubt = st.form_submit_button("Ask ➔", type="primary", use_container_width=True)

        query_to_process = None
        if submit_doubt and new_doubt_text and new_doubt_text.strip():
            query_to_process = new_doubt_text.strip()
        elif pending_doubt:
            query_to_process = pending_doubt

        if query_to_process:
            add_doubt_message(user_id, thread_id, d_idx, "user", query_to_process)
            history_for_llm = get_doubt_messages(user_id, thread_id, d_idx)[:-1]

            with doubt_container:
                with st.chat_message("user"):
                    st.markdown(query_to_process)
                with st.chat_message("assistant"):
                    with st.spinner("Clarifying..."):
                        answer = answer_doubt(
                            original_prompt=d_prompt,
                            original_response=d_resp,
                            doubt_history=history_for_llm,
                            new_query=query_to_process,
                        )
                        st.markdown(answer)

            add_doubt_message(user_id, thread_id, d_idx, "assistant", answer)
            st.rerun()
