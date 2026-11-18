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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode

from core.config import settings
from core.context import trim_history
from core.state import AgentState
from tools.tools import RAG_TOOLS, get_retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a document analysis assistant.

You have access to these sources:
     - rag_tool: Retrieves relevant passages from uploaded documents (PDF, DOCX, TXT, Markdown).
     - duckduckgo_search: Searches the live web for current or supplementary information.
     - calculator: Performs numeric calculations when needed.

Your process:
1. Use rag_tool to retrieve relevant passages from the uploaded documents.
2. Use duckduckgo_search when the question needs current information, when
    supplementary context is useful, or when the documents have no relevant answer.
3. Combine the sources into one clear answer, clearly separating:
    - **From the uploaded documents:** facts supported by retrieved passages, citing the source document filename and page (if available).
    - **From the web:** current or supplementary facts, with links to returned sources.
4. If the documents have no relevant information, say exactly:
    "I couldn't find this information in the uploaded documents."
    Then provide the useful web answer under **From the web** when search results support it.

Rules:
- Always use rag_tool FIRST for document questions.
- Always pass the user_id and thread_id arguments to rag_tool.
- Quote or paraphrase from the documents — do not fabricate.
- Each retrieved passage includes a source filename and, when available, a page number.
  Cite the source inline next to each fact:
  - For PDFs: "[filename.pdf, p. 4]"
  - For other formats (DOCX, TXT, MD): "[filename.docx]" or "[filename.txt]"
- When synthesizing across multiple documents, clearly indicate which facts come from which document.
- Never present web information as if it came from the documents.
- Never present document information as if it came from the web.
- Include a **Sources** section with the document filenames and web links.
- If no document is uploaded, politely ask the user to upload one.
"""

RAG_NO_DOCUMENT_PROMPT = """You are a helpful assistant.

The user seems to be asking about a document, but no document has been uploaded yet.

Politely inform the user that they need to upload a document first (PDF, DOCX, TXT, or Markdown)
using the upload button in the sidebar. Then you will be able to answer questions about it.

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
            *trim_history(state["messages"]),
        ]
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=0.2,
            api_key=settings.groq_api_key,
        )
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            logger.error("RAG (no-document) LLM error: %s", exc)
            response = AIMessage(
                content="⚠️ I ran into an error. Please try again, or upload a PDF from the sidebar."
            )
        return {**state, "messages": [response], "agent_name": "RAG"}

    # Document available — run RAG with tool calling
    system = SystemMessage(
        content=(
            RAG_SYSTEM_PROMPT
            + f"\n\nFor this conversation: user_id='{user_id}', thread_id='{thread_id}'."
            + "\nWhen calling rag_tool, pass these values as the user_id and thread_id arguments."
        )
    )
    messages = [system, *trim_history(state["messages"])]
    try:
        response = _rag_llm.invoke(messages)
    except Exception as exc:
        logger.error("RAG agent LLM error: %s", exc)
        response = AIMessage(
            content="⚠️ I ran into an error while reading the document. Please try again."
        )
    return {**state, "messages": [response], "agent_name": "RAG"}


def get_rag_tools():
    return RAG_TOOLS
