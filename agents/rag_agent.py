"""
agents/rag_agent.py
====================
RAG / Document Agent.

Handles questions about uploaded PDF documents.
Reuses:
  - ingest_pdf() from tools/tools.py (same logic as langraph_rag_backend.py)
  - FAISS per-session retriever store
  - rag_tool for retrieval

Falls back gracefully if no document has been uploaded.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from core.config import settings
from core.state import AgentState
from tools.tools import RAG_TOOLS, get_retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a document analysis assistant.

You have access to:
  - rag_tool: Retrieves relevant passages from the uploaded PDF document.
  - search_tool: Falls back to web search if the document doesn't contain the answer.
  - calculator: For any numeric computations needed.

Your process:
1. Use rag_tool to retrieve relevant passages from the document.
2. Answer the question based on the retrieved passages.
3. If the document does not contain the answer, say so clearly and optionally
   use the web search tool for supplementary information.

Rules:
- Always use rag_tool FIRST for document questions.
- Always pass the user_id and thread_id arguments to rag_tool.
- Quote or paraphrase from the document — do not fabricate.
- If no document is uploaded, politely ask the user to upload one.
"""

RAG_NO_DOCUMENT_PROMPT = """You are a helpful assistant.

The user seems to be asking about a document, but no document has been uploaded yet.

Politely inform the user that they need to upload a PDF document first using the
upload button in the sidebar. Then you will be able to answer questions about it.

Keep your response brief and friendly.
"""

# ---------------------------------------------------------------------------
# LLM + tools
# ---------------------------------------------------------------------------

_rag_llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,
    api_key=settings.groq_api_key,
).bind_tools(RAG_TOOLS)

rag_tool_node = ToolNode(RAG_TOOLS)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> AgentState:
    """
    Main RAG node.

    Injects user_id and thread_id into the system prompt so the LLM
    can pass them to rag_tool correctly.
    """
    user_id = state.get("user_id", "")
    thread_id = state.get("thread_id", "")

    # Check if a document is available for this session
    has_doc = get_retriever(user_id, thread_id) is not None

    if not has_doc:
        # No document — politely redirect
        messages = [
            SystemMessage(content=RAG_NO_DOCUMENT_PROMPT),
            *state["messages"],
        ]
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=0.2,
            api_key=settings.groq_api_key,
        )
        response = llm.invoke(messages)
        return {**state, "messages": [response], "agent_name": "RAG"}

    # Document available — run RAG with tool calling
    system = SystemMessage(
        content=(
            RAG_SYSTEM_PROMPT
            + f"\n\nFor this conversation: user_id='{user_id}', thread_id='{thread_id}'."
            + "\nWhen calling rag_tool, pass these values as the user_id and thread_id arguments."
        )
    )
    messages = [system, *state["messages"]]
    response = _rag_llm.invoke(messages)
    return {**state, "messages": [response], "agent_name": "RAG"}


def get_rag_tools():
    return RAG_TOOLS
