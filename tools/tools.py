"""
tools/tools.py
==============
Single source of truth for ALL tool definitions.

Consolidates calculator, get_stock_price, web_search, and rag_tool from
the original langraph_rag_backend.py.  Agents import from here — no
duplicate tool definitions anywhere.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from ddgs import DDGS
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HuggingFace embeddings (loaded once, reused everywhere)
# ---------------------------------------------------------------------------

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the singleton HuggingFaceEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading HuggingFace embeddings model: %s", settings.embedding_model)
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


# ---------------------------------------------------------------------------
# Per-session, per-thread FAISS retriever store
# ---------------------------------------------------------------------------
# Key: "{user_id}:{thread_id}"
# Value: FAISS retriever
#
# This replaces _THREAD_RETRIEVERS in langraph_rag_backend.py and adds
# user-level isolation so User A cannot read User B's documents.
# ---------------------------------------------------------------------------

_RETRIEVERS: Dict[str, Any] = {}          # composite_key -> retriever
_RETRIEVER_META: Dict[str, List[dict]] = {}  # composite_key -> list of doc metadata


def _composite(user_id: str, thread_id: str) -> str:
    return f"{user_id}:{thread_id}"


def get_retriever(user_id: str, thread_id: str) -> Optional[Any]:
    return _RETRIEVERS.get(_composite(user_id, thread_id))


def get_retriever_meta(user_id: str, thread_id: str) -> List[dict]:
    return _RETRIEVER_META.get(_composite(user_id, thread_id), [])


def ingest_pdf(
    file_bytes: bytes,
    user_id: str,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Load, split, embed and store a PDF for a specific (user, thread).

    Returns a summary dict surfaced in the UI.

    ROOT CAUSE of 'list index out of range' was FAISS.from_documents([]) when
    all pages were blank (image-only PDFs) — now guarded explicitly.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    key = _composite(user_id, thread_id)
    embeddings = get_embeddings()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # ── Step 1: Load pages ──────────────────────────────────────────────
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        except Exception as load_exc:
            logger.error("PyPDFLoader failed for '%s': %s", filename, load_exc)
            raise ValueError(
                f"Could not read '{filename}'. Ensure it is a valid, non-encrypted PDF."
                f" Error: {load_exc}"
            ) from load_exc

        logger.info("PDF '%s' loaded: %d raw pages", filename, len(docs))

        if not docs:
            raise ValueError(
                f"'{filename}' appears empty — no pages were extracted."
            )

        # ── Step 2: Filter blank pages ──────────────────────────────────────
        # THIS IS THE ROOT CAUSE OF THE BUG.
        # PyPDFLoader returns Document objects even for image-only pages with
        # page_content="" or pure whitespace. FAISS.from_documents([]) then
        # raises IndexError because internally it calls texts[0] on an empty list.
        non_empty_docs = [
            doc for doc in docs
            if doc.page_content and doc.page_content.strip()
        ]
        skipped = len(docs) - len(non_empty_docs)
        if skipped:
            logger.warning(
                "PDF '%s': %d/%d page(s) had no text and were skipped.",
                filename, skipped, len(docs),
            )

        if not non_empty_docs:
            raise ValueError(
                f"'{filename}' has {len(docs)} page(s) but no extractable text. "
                "This is likely a scanned/image-only PDF. Please use a text-based PDF."
            )

        # ── Step 3: Split into chunks ────────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(non_empty_docs)
        logger.info("PDF '%s': %d pages → %d chunks", filename, len(non_empty_docs), len(chunks))

        if not chunks:
            raise ValueError(
                f"'{filename}' produced no text chunks. The pages may contain very little text."
            )

        # ── Step 4: Build FAISS index ────────────────────────────────────────
        try:
            vector_store = FAISS.from_documents(chunks, embeddings)
        except Exception as faiss_exc:
            logger.error("FAISS indexing failed for '%s': %s", filename, faiss_exc)
            raise ValueError(
                f"Failed to build search index for '{filename}': {faiss_exc}"
            ) from faiss_exc

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        _RETRIEVERS[key] = retriever

        meta = {
            "filename": filename or os.path.basename(tmp_path),
            "documents": len(docs),
            "non_empty_pages": len(non_empty_docs),
            "chunks": len(chunks),
        }
        _RETRIEVER_META.setdefault(key, []).append(meta)

        logger.info(
            "PDF ingested OK for %s: '%s' (%d pages, %d usable, %d chunks)",
            key, meta["filename"], meta["documents"], meta["non_empty_pages"], meta["chunks"],
        )
        return meta

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Tool: DuckDuckGo web search — structured results (title + link + snippet)
# ---------------------------------------------------------------------------
# A plain DuckDuckGoSearchRun only returns a blob of concatenated snippet
# text with no links, so answers could never cite real sources. This wraps
# the same underlying API but keeps titles/links so agents can present and
# cite sources instead of just paraphrasing search text.
# ---------------------------------------------------------------------------

@tool
def duckduckgo_search(query: str) -> dict:
    """
    Search the live web via DuckDuckGo.
    Returns a structured list of results (title, link, snippet) so the
    answer can cite real, clickable sources instead of just search text.
    """
    try:
        raw_results = list(DDGS().text(query, max_results=5))
    except Exception as exc:
        logger.error("Web search failed for '%s': %s", query, exc)
        return {"query": query, "results": [], "error": str(exc)}

    if not raw_results:
        return {"query": query, "results": [], "message": "No results found."}

    return {
        "query": query,
        "results": [
            {
                "title": (r.get("title") or "").strip(),
                "link": (r.get("href") or "").strip(),
                "snippet": (r.get("body") or "").strip(),
            }
            for r in raw_results
        ],
    }


# Backward-compatible alias — used throughout this module's tool lists.
search_tool = duckduckgo_search


# ---------------------------------------------------------------------------
# Tool: Calculator
# ---------------------------------------------------------------------------

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Stock price (Alpha Vantage — optional key)
# ---------------------------------------------------------------------------

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA').
    Requires ALPHAVANTAGE_API_KEY in the environment.
    """
    api_key = settings.alphavantage_api_key
    if not api_key:
        return {
            "error": (
                "ALPHAVANTAGE_API_KEY is not configured. "
                "Stock lookup is unavailable."
            ),
            "symbol": symbol,
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc), "symbol": symbol}


# ---------------------------------------------------------------------------
# Tool: RAG retrieval (requires a pre-indexed document)
# ---------------------------------------------------------------------------

@tool
def rag_tool(query: str, user_id: str = "", thread_id: str = "") -> dict:
    """
    Retrieve relevant passages from the uploaded PDF document for this
    conversation.  Always pass user_id and thread_id when calling this tool.
    Each passage includes its source page number so the answer can cite it.
    """
    retriever = get_retriever(user_id, thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this conversation. Upload a PDF first.",
            "query": query,
        }

    try:
        results = retriever.invoke(query)
        passages = []
        for doc in results:
            page = doc.metadata.get("page")
            passages.append(
                {
                    "content": doc.page_content,
                    # PyPDFLoader pages are 0-indexed internally — show human page numbers.
                    "page": (page + 1) if isinstance(page, int) else None,
                }
            )
        return {
            "query": query,
            "passages": passages,
            "source_files": [
                m.get("filename") for m in get_retriever_meta(user_id, thread_id)
            ],
        }
    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc)
        return {"error": str(exc), "query": query}


# ---------------------------------------------------------------------------
# Tool collections for each agent
# ---------------------------------------------------------------------------

RESEARCH_TOOLS = [search_tool, get_stock_price, calculator]
RAG_TOOLS = [rag_tool, search_tool, calculator]
CODING_TOOLS = [calculator]
GENERAL_TOOLS = [search_tool, calculator]
