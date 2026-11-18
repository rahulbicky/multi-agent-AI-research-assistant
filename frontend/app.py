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
import pyperclip
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
    rewind_for_regeneration,
    rewind_for_edit,
)
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
        "_trigger_regen",
        "_trigger_edit",
        "_stop_generation",
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
    bg_main = "#0f0f13" if is_dark else "#f8fafc"
    bg_sidebar = "linear-gradient(180deg, #16161e 0%, #1a1a24 100%)" if is_dark else "linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%)"
    border_color = "#2a2a3a" if is_dark else "#e2e8f0"
    text_main = "#e8e8f0" if is_dark else "#0f172a"
    text_muted = "#6b7280" if is_dark else "#64748b"
    card_bg = "#1c1c28" if is_dark else "#ffffff"
    card_hover = "#1e1e2e" if is_dark else "#f1f5f9"
    user_bubble = "rgba(99, 102, 241, 0.08)" if is_dark else "#eef2ff"
    user_border = "rgba(99, 102, 241, 0.2)" if is_dark else "#c7d2fe"
    chat_input_bg = "#1c1c28" if is_dark else "#ffffff"
    chat_input_border = "#3a3a50" if is_dark else "#cbd5e1"
    btn_text = "#c0c0d0" if is_dark else "#334155"
    btn_action = "#4b5563" if is_dark else "#64748b"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {bg_main} !important;
        color: {text_main} !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}
    [data-testid="stSidebar"] {{
        background: {bg_sidebar} !important;
        border-right: 1px solid {border_color} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {text_main} !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        color: {btn_text};
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
        transition: all 0.15s ease;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #a5b4fc !important;
    }}
    .new-chat-btn > button {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
    }}
    .new-chat-btn > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    }}
    [data-testid="stMain"], .main .block-container {{
        background: {bg_main} !important;
        padding-top: 20px !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }}
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        border: none !important;
        padding: 6px 0 !important;
    }}
    [data-testid="stChatMessage"][data-testid*="user"],
    .stChatMessage.user {{
        background: {user_bubble} !important;
        border: 1px solid {user_border} !important;
        border-radius: 16px !important;
        padding: 16px !important;
        color: {text_main} !important;
    }}
    [data-testid="stChatInput"] {{
        background: {chat_input_bg} !important;
        border: 1px solid {chat_input_border} !important;
        border-radius: 14px !important;
        color: {text_main} !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: {text_main} !important;
    }}
    .agent-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }}
    .badge-general   {{ background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }}
    .badge-research  {{ background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }}
    .badge-rag       {{ background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }}
    .badge-coding    {{ background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }}
    .hero-header {{
        text-align: center;
        padding: 40px 20px 24px;
    }}
    .hero-title {{
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a78bfa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 12px;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: {text_muted};
        font-weight: 400;
    }}
    .capability-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin: 24px 0;
    }}
    .capability-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 18px;
        transition: all 0.2s ease;
        cursor: default;
    }}
    .capability-card:hover {{
        border-color: #6366f1;
        background: {card_hover};
        transform: translateY(-2px);
    }}
    .capability-icon {{ font-size: 1.5rem; margin-bottom: 8px; }}
    .capability-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {text_main};
        margin-bottom: 4px;
    }}
    .capability-desc {{ font-size: 0.82rem; color: {text_muted}; }}
    .sidebar-section {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {btn_action};
        padding: 8px 4px 4px;
        margin-top: 8px;
    }}
    [data-testid="stStatus"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
    }}
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {border_color}; border-radius: 3px; }}
    [data-testid="stFileUploader"] {{
        background: {card_bg} !important;
        border: 1px dashed {border_color} !important;
        border-radius: 10px !important;
    }}
    hr {{ border-color: {border_color} !important; }}
    [data-testid="stExpander"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
    }}
    @media (max-width: 640px) {{
        .capability-grid {{ grid-template-columns: 1fr; }}
        .hero-title {{ font-size: 2rem; }}
        .hero-header {{ padding: 32px 12px 24px; }}
        [data-testid="stMain"], .main .block-container {{ padding-left: 8px !important; padding-right: 8px !important; }}
    }}
    .conv-action-btn > button {{
        background: transparent !important;
        border: none !important;
        color: {btn_action} !important;
        padding: 2px 6px !important;
        font-size: 12px !important;
        min-height: 0 !important;
        line-height: 1 !important;
        width: auto !important;
    }}
    .conv-action-btn > button:hover {{
        color: #a78bfa !important;
        background: rgba(99,102,241,0.1) !important;
        border-radius: 4px !important;
    }}
    .conv-delete-btn > button:hover {{
        color: #f87171 !important;
        background: rgba(239,68,68,0.1) !important;
    }}
    .msg-toolbar {{
        display: flex;
        gap: 8px;
        margin-top: 6px;
        align-items: center;
    }}
    .msg-action-btn > button {{
        background: transparent !important;
        border: 1px solid {border_color} !important;
        color: {btn_action} !important;
        padding: 2px 8px !important;
        font-size: 11px !important;
        border-radius: 6px !important;
        min-height: 0 !important;
        line-height: 1.2 !important;
    }}
    .msg-action-btn > button:hover {{
        color: #a78bfa !important;
        border-color: #6366f1 !important;
        background: rgba(99,102,241,0.1) !important;
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


def switch_thread(tid: str):
    """Switch to an existing thread and restore its messages."""
    st.session_state["thread_id"] = tid
    # Load from LangGraph checkpointer
    config = make_config(user_id, tid)
    try:
        state = chatbot.get_state(config=config)
        raw_msgs = state.values.get("messages", [])
        restored = []
        for msg in raw_msgs:
            if isinstance(msg, HumanMessage):
                restored.append({"role": "user", "content": msg.content, "agent_name": None})
            elif isinstance(msg, AIMessage) and msg.content:
                agent = state.values.get("agent_name", "General")
                restored.append({"role": "assistant", "content": msg.content, "agent_name": agent})
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
    # Logo / title
    st.markdown("""
    <div style="padding: 12px 4px 10px;">
        <div style="font-size:1.4rem; font-weight:700; color:#a78bfa;">🤖 AI Assistant</div>
        <div style="font-size:11px; color:#4b5563; margin-top:2px;">Multi-Agent Research System</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Toggle ──
    col_th1, col_th2 = st.columns(2)
    with col_th1:
        dark_active = (current_theme == "dark")
        if st.button(
            "🌙 Dark",
            use_container_width=True,
            key="theme_dark_btn",
            type="primary" if dark_active else "secondary",
        ):
            if not dark_active:
                st.session_state["theme"] = "dark"
                set_user_setting(user_id, "theme", "dark")
                st.rerun()
    with col_th2:
        light_active = (current_theme == "light")
        if st.button(
            "☀️ Light",
            use_container_width=True,
            key="theme_light_btn",
            type="primary" if light_active else "secondary",
        ):
            if not light_active:
                st.session_state["theme"] = "light"
                set_user_setting(user_id, "theme", "light")
                st.rerun()

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
            if st.button(display_title, key=f"thread_btn_{tid}", use_container_width=True):
                if not is_active:
                    switch_thread(tid)
                    st.rerun()
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
        st.caption(f"MCP tools: {len(mcp)} loaded")
    else:
        st.caption("MCP: not connected")

    st.caption(f"Session: `{user_id[-8:]}`")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

# Re-read current values (may have changed after sidebar interaction)
thread_id = st.session_state["thread_id"]
composite_key = f"{user_id}:{thread_id}"

message_history = st.session_state["message_history"]

# ── Welcome screen (empty chat) ──
if not message_history:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">AI Research Assistant</div>
        <div class="hero-subtitle">Powered by Multi-Agent Intelligence · Groq · LangGraph</div>
    </div>
    <div class="capability-grid">
        <div class="capability-card">
            <div class="capability-icon">🔍</div>
            <div class="capability-title">Web Research</div>
            <div class="capability-desc">Search the web, get stock prices, find current news and data</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">📄</div>
            <div class="capability-title">Document Q&A</div>
            <div class="capability-desc">Upload a PDF and ask detailed questions about its contents</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">💻</div>
            <div class="capability-title">Coding Assistant</div>
            <div class="capability-desc">Generate, debug, review and explain code in any language</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">💬</div>
            <div class="capability-title">General Chat</div>
            <div class="capability-desc">Ask anything — the right agent is selected automatically</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Handle interrupted / stopped generation from previous run ──
if st.session_state.get("_is_generating"):
    st.session_state["_is_generating"] = False
    partial_text = st.session_state.get("_current_stream_text", "").strip()
    partial_agent = st.session_state.get("_current_stream_agent", "General")
    if partial_text:
        message_history.append({
            "role": "assistant",
            "content": partial_text + "\n\n*(Generation stopped by user)*",
            "agent_name": partial_agent,
        })
        st.session_state["message_history"] = message_history
    st.session_state["_current_stream_text"] = ""
    st.toast("⏹️ Generation stopped.")

# ── Render existing messages ──
last_assistant_idx = None
for i in range(len(message_history) - 1, -1, -1):
    if message_history[i].get("role") == "assistant":
        last_assistant_idx = i
        break

for idx, msg in enumerate(message_history):
    role = msg["role"]
    content = msg["content"]
    agent_name = msg.get("agent_name")

    with st.chat_message(role):
        if role == "assistant" and agent_name:
            st.markdown(render_agent_badge(agent_name), unsafe_allow_html=True)
        st.markdown(content)

        if role == "user":
            col_u, _ = st.columns([0.15, 0.85])
            with col_u:
                st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
                if st.button("✏️ Edit", key=f"edit_btn_{idx}", help="Edit this message and re-branch the conversation"):
                    st.session_state["_edit_msg_idx"] = idx
                    st.session_state["_edit_msg_content"] = content
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        elif role == "assistant":
            is_last = (idx == last_assistant_idx)
            col_c1, col_c2, _ = st.columns([0.15, 0.22, 0.63])
            with col_c1:
                st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
                if st.button("📋 Copy", key=f"copy_btn_{idx}", help="Copy response to clipboard"):
                    try:
                        pyperclip.copy(content)
                        st.toast("✅ Copied response to clipboard!")
                    except Exception as exc:
                        st.toast(f"⚠️ Copy failed: {exc}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_c2:
                st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
                if is_last:
                    if st.button("🔄 Regenerate", key=f"regen_btn_{idx}", help="Regenerate this response"):
                        rewound_prompt = rewind_for_regeneration(user_id, thread_id)
                        if st.session_state["message_history"] and st.session_state["message_history"][-1]["role"] == "assistant":
                            st.session_state["message_history"].pop()
                        st.session_state["_trigger_regen"] = rewound_prompt or (
                            st.session_state["message_history"][-1]["content"] if st.session_state["message_history"] else ""
                        )
                        st.rerun()
                else:
                    st.button("🔄 Regenerate", key=f"regen_btn_{idx}", disabled=True, help="Only the most recent response can be regenerated")
                st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat input + response
# ─────────────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask anything — research, documents, code, or general questions…")

trigger_regen = st.session_state.pop("_trigger_regen", None)
trigger_edit = st.session_state.pop("_trigger_edit", None)

prompt_to_stream = None
is_regeneration = False

if trigger_regen:
    prompt_to_stream = trigger_regen
    is_regeneration = True
elif trigger_edit:
    prompt_to_stream = trigger_edit
    is_regeneration = False
    message_history.append({"role": "user", "content": prompt_to_stream, "agent_name": None})
    with st.chat_message("user"):
        st.markdown(prompt_to_stream)
elif user_input:
    prompt_to_stream = user_input
    is_regeneration = False
    # Auto-title first message
    if not message_history:
        auto_t = auto_title(user_input)
        st.session_state["thread_titles"][thread_id] = auto_t
        try:
            save_thread_title(user_id, thread_id, auto_t)
        except Exception:
            pass  # Non-fatal — in-memory title is still set

    # Display user message
    message_history.append({"role": "user", "content": user_input, "agent_name": None})
    with st.chat_message("user"):
        st.markdown(user_input)

if prompt_to_stream:
    pending_filename = st.session_state.get("pending_upload_confirmation")
    if not pending_filename and current_docs:
        pending_filename = current_docs[-1].get("filename")
    normalized_input = prompt_to_stream.strip().lower().rstrip(".!?")
    if not is_regeneration and pending_filename and normalized_input in {
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
        with st.chat_message("assistant"):
            st.markdown(render_agent_badge("RAG"), unsafe_allow_html=True)
            st.markdown(confirmation)
        message_history.append({
            "role": "assistant",
            "content": confirmation,
            "agent_name": "RAG",
        })
        st.session_state["message_history"] = message_history
        st.stop()

    st.session_state["pending_upload_confirmation"] = None

    # Build LangGraph config (isolated per user+thread)
    config = make_config(user_id, thread_id)

    # ── Stream the assistant response ──
    with st.chat_message("assistant"):
        stop_placeholder = st.empty()
        with stop_placeholder:
            st.markdown('<div class="msg-action-btn">', unsafe_allow_html=True)
            if st.button("⏹️ Stop Generating", key="stop_gen_active_btn", type="secondary"):
                pass  # Clicking triggers Streamlit rerun, caught by top-level _is_generating handler
            st.markdown('</div>', unsafe_allow_html=True)

        status_holder = {"box": None, "error": False}
        sources_holder = {"web": [], "rag_sources": {}}
        badge_placeholder = st.empty()
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '<span style="color:#6b7280;">🧭 Thinking…</span>', unsafe_allow_html=True
        )

        st.session_state["_is_generating"] = True
        st.session_state["_current_stream_text"] = ""
        st.session_state["_current_stream_agent"] = "General"

        def _parse_tool_payload(raw_content) -> dict | None:
            """Best-effort JSON parse of a tool's structured output."""
            if not isinstance(raw_content, str):
                return None
            try:
                payload = json.loads(raw_content)
                return payload if isinstance(payload, dict) else None
            except (ValueError, TypeError):
                return None

        def stream_response():
            """Generator yielding text tokens from the multi-agent graph."""
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
                    # ── Skip supervisor node output (classification word leaks into UI) ──
                    node_name = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
                    if node_name == "supervisor":
                        continue

                    # ── Tool execution indicator ──
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        label = tool_status_label(tool_name)
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(label, expanded=True)
                        else:
                            status_holder["box"].update(label=label, state="running", expanded=True)

                        # Best-effort: surface a short progress line + capture sources.
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

                    # ── AI response tokens (skip tool-call-only messages with no text) ──
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        # Skip if message has ONLY tool_calls and no actual text
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

        # Resolve tool status indicator
        if status_holder["box"] is not None:
            if status_holder["error"]:
                status_holder["box"].update(label="⚠️ Tool error", state="error", expanded=True)
            else:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

        # Retrieve agent name from final graph state
        try:
            final_state = chatbot.get_state(config=config)
            agent_name = final_state.values.get("agent_name", "General")
        except Exception:
            agent_name = "General"

        # Show agent badge
        badge_placeholder.markdown(render_agent_badge(agent_name), unsafe_allow_html=True)

        # ── Sources panel (web links / document sources used for this answer) ──
        if sources_holder["web"] or sources_holder["rag_sources"]:
            with st.expander("🔗 Sources", expanded=False):
                for s in sources_holder["web"]:
                    st.markdown(f"- [{s['title']}]({s['link']})")
                for src, pages in sources_holder["rag_sources"].items():
                    ext = src.rsplit(".", 1)[-1].lower() if "." in src else ""
                    icon = "📄" if ext == "pdf" else ("📑" if ext == "docx" else ("🗒️" if ext == "md" else "📝"))
                    if pages:
                        pages_str = ", ".join(f"p. {p}" for p in sorted(pages))
                        st.markdown(f"- {icon} **{src}** ({pages_str})")
                    else:
                        st.markdown(f"- {icon} **{src}**")

    # Save to display history
    message_history.append({
        "role": "assistant",
        "content": ai_response or "",
        "agent_name": agent_name,
    })
    st.session_state["message_history"] = message_history
    st.rerun()
