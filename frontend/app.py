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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.config import settings
from core.memory import list_user_threads, make_config
from core.graph import chatbot
from tools.tools import ingest_pdf, get_retriever_meta

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
# CSS — Modern dark ChatGPT-style theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0f0f13 !important;
    color: #e8e8f0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16161e 0%, #1a1a24 100%) !important;
    border-right: 1px solid #2a2a3a !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: #c0c0d0;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    transition: all 0.15s ease;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #a5b4fc !important;
}

/* ── New Chat button ── */
.new-chat-btn > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.new-chat-btn > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* ── Main content area ── */
[data-testid="stMain"], .main .block-container {
    background: #0f0f13 !important;
    padding-top: 20px !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage.user {
    background: rgba(99, 102, 241, 0.08) !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #1c1c28 !important;
    border: 1px solid #3a3a50 !important;
    border-radius: 14px !important;
    color: #e8e8f0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8e8f0 !important;
}

/* ── Agent badge ── */
.agent-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    text-transform: uppercase;
}
.badge-general   { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-research  { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-rag       { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-coding    { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 60px 20px 40px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 12px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    font-weight: 400;
}

/* ── Capability cards ── */
.capability-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin: 32px 0;
}
.capability-card {
    background: #1c1c28;
    border: 1px solid #2a2a3a;
    border-radius: 14px;
    padding: 18px;
    transition: all 0.2s ease;
    cursor: default;
}
.capability-card:hover {
    border-color: #6366f1;
    background: #1e1e2e;
    transform: translateY(-2px);
}
.capability-icon { font-size: 1.5rem; margin-bottom: 8px; }
.capability-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #c0c0d0;
    margin-bottom: 4px;
}
.capability-desc { font-size: 0.82rem; color: #6b7280; }

/* ── Sidebar section headers ── */
.sidebar-section {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4b5563;
    padding: 8px 4px 4px;
    margin-top: 8px;
}

/* ── Status / tool usage ── */
[data-testid="stStatus"] {
    background: #1c1c28 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a50; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1c1c28 !important;
    border: 1px dashed #3a3a50 !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: #1e1e2e !important; }

/* ── Sources expander ── */
[data-testid="stExpander"] {
    background: #1c1c28 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
}

/* ── Responsive: narrow / mobile screens ── */
@media (max-width: 640px) {
    .capability-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 2rem; }
    .hero-header { padding: 32px 12px 24px; }
    [data-testid="stMain"], .main .block-container { padding-left: 8px !important; padding-right: 8px !important; }
}
</style>
""", unsafe_allow_html=True)

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
# Session Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    """Initialise all Streamlit session state variables."""

    # Unique per-browser session user ID
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = f"user_{uuid.uuid4().hex[:12]}"

    # Current conversation thread
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Display-side message history {role, content, agent_name}
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    # Sidebar thread list {thread_id: title}
    if "thread_titles" not in st.session_state:
        # Load persisted threads for this user from SQLite
        existing = list_user_threads(st.session_state["user_id"])
        st.session_state["thread_titles"] = {
            tid: f"Chat {i+1}" for i, tid in enumerate(existing)
        }

    # Register current thread
    tid = st.session_state["thread_id"]
    if tid not in st.session_state["thread_titles"]:
        st.session_state["thread_titles"][tid] = "New Chat"

    # Documents indexed in this session {composite_key: [metadata]}
    if "session_docs" not in st.session_state:
        st.session_state["session_docs"] = {}

    if "pending_upload_confirmation" not in st.session_state:
        st.session_state["pending_upload_confirmation"] = None


_init_session()

user_id = st.session_state["user_id"]
thread_id = st.session_state["thread_id"]
composite_key = f"{user_id}:{thread_id}"


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
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo / title
    st.markdown("""
    <div style="padding: 12px 4px 20px;">
        <div style="font-size:1.4rem; font-weight:700; color:#a78bfa;">🤖 AI Assistant</div>
        <div style="font-size:11px; color:#4b5563; margin-top:4px;">Multi-Agent Research System</div>
    </div>
    """, unsafe_allow_html=True)

    # ── New Chat ──
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✏️  New Chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── PDF Upload ──
    st.markdown('<div class="sidebar-section">📎 Document Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    session_docs = st.session_state["session_docs"]
    current_docs = session_docs.get(composite_key, [])

    if uploaded_file is not None:
        filename = uploaded_file.name
        already_loaded = any(d.get("filename") == filename for d in current_docs)
        if already_loaded:
            st.success(f"✅ `{filename}` already indexed")
        else:
            with st.status("📄 Indexing PDF…", expanded=True):
                try:
                    meta = ingest_pdf(
                        file_bytes=uploaded_file.getvalue(),
                        user_id=user_id,
                        thread_id=thread_id,
                        filename=filename,
                    )
                    if composite_key not in session_docs:
                        session_docs[composite_key] = []
                    session_docs[composite_key].append(meta)
                    st.session_state["pending_upload_confirmation"] = meta["filename"]
                    st.write(f"✅ **{meta['filename']}**")
                    pages_info = f"{meta['non_empty_pages']} usable / {meta['documents']} total pages"
                    st.write(f"   {pages_info} · {meta['chunks']} chunks")
                    st.success("Ready to answer questions about this document!")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    # Show indexed docs for current thread
    if current_docs:
        for doc in current_docs:
            st.caption(f"📄 {doc.get('filename')} ({doc.get('chunks')} chunks)")

    st.markdown("---")

    # ── Conversation History ──
    st.markdown('<div class="sidebar-section">💬 Conversations</div>', unsafe_allow_html=True)

    thread_titles = st.session_state["thread_titles"]
    # Show newest first
    sorted_threads = list(reversed(list(thread_titles.keys())))

    for tid in sorted_threads:
        title = thread_titles.get(tid, "Chat")
        is_active = tid == st.session_state["thread_id"]
        btn_label = f"{'▶ ' if is_active else ''}{title}"
        if st.button(btn_label, key=f"thread_btn_{tid}", use_container_width=True):
            if not is_active:
                switch_thread(tid)
                st.rerun()

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

# ── Render existing messages ──
for msg in message_history:
    role = msg["role"]
    content = msg["content"]
    agent_name = msg.get("agent_name")

    with st.chat_message(role):
        if role == "assistant" and agent_name:
            st.markdown(render_agent_badge(agent_name), unsafe_allow_html=True)
        st.markdown(content)


# ─────────────────────────────────────────────────────────────────────────────
# Chat input + response
# ─────────────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask anything — research, documents, code, or general questions…")

if user_input:
    # Auto-title first message
    if not message_history:
        auto_t = auto_title(user_input)
        st.session_state["thread_titles"][thread_id] = auto_t

    # Display user message
    message_history.append({"role": "user", "content": user_input, "agent_name": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    pending_filename = st.session_state.get("pending_upload_confirmation")
    if not pending_filename and current_docs:
        pending_filename = current_docs[-1].get("filename")
    normalized_input = user_input.strip().lower().rstrip(".!?")
    if pending_filename and normalized_input in {
        "i have uploaded",
        "i uploaded",
        "uploaded",
    }:
        st.session_state["pending_upload_confirmation"] = None
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
        status_holder = {"box": None, "error": False}
        sources_holder = {"web": [], "rag_pages": set()}
        badge_placeholder = st.empty()
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '<span style="color:#6b7280;">🧭 Thinking…</span>', unsafe_allow_html=True
        )

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
                        "messages": [HumanMessage(content=user_input)],
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
                                    if p.get("page"):
                                        sources_holder["rag_pages"].add(p["page"])

                    # ── AI response tokens (skip tool-call-only messages with no text) ──
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        # Skip if message has ONLY tool_calls and no actual text
                        tool_calls = getattr(message_chunk, "tool_calls", [])
                        if tool_calls and not str(message_chunk.content).strip():
                            continue
                        if first_token:
                            thinking_placeholder.empty()
                            first_token = False
                        yield message_chunk.content

            except Exception as exc:
                logger.error("Graph stream error: %s", exc)
                thinking_placeholder.empty()
                status_holder["error"] = True
                if status_holder["box"] is not None:
                    status_holder["box"].update(state="error", expanded=True)
                yield f"\n\n⚠️ Something went wrong while generating a response: {exc}"

        ai_response = st.write_stream(stream_response())
        thinking_placeholder.empty()

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

        # ── Sources panel (web links / document pages used for this answer) ──
        if sources_holder["web"] or sources_holder["rag_pages"]:
            with st.expander("🔗 Sources", expanded=False):
                for s in sources_holder["web"]:
                    st.markdown(f"- [{s['title']}]({s['link']})")
                if sources_holder["rag_pages"]:
                    pages = ", ".join(str(p) for p in sorted(sources_holder["rag_pages"]))
                    st.markdown(f"- 📄 Document page(s): {pages}")

    # Save to display history
    message_history.append({
        "role": "assistant",
        "content": ai_response or "",
        "agent_name": agent_name,
    })
    st.session_state["message_history"] = message_history
