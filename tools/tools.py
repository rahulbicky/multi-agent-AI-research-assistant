"""
tools/tools.py
==============
Single source of truth for ALL tool definitions.

Consolidates calculator, get_stock_price, web_search, and rag_tool from
the original langraph_rag_backend.py.  Agents import from here — no
duplicate tool definitions anywhere.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from ddgs import DDGS
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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
# Per-session, per-thread FAISS retriever & persistent storage
# ---------------------------------------------------------------------------
# Key: "{user_id}:{thread_id}"
# In-memory caches:
#   _RETRIEVERS: composite_key -> retriever
#   _VECTOR_STORES: composite_key -> FAISS vector_store instance
#   _RETRIEVER_META: composite_key -> list of doc metadata
#
# Persistent disk layout:
#   <VECTOR_STORE_DIR>/<user_id>/<thread_id>/
#     ├── index.faiss
#     ├── index.pkl
#     └── metadata.json
#
# Provides complete isolation between users and conversation threads, with
# automatic disk persistence across restarts.
# ---------------------------------------------------------------------------

_RETRIEVERS: Dict[str, Any] = {}             # composite_key -> retriever
_VECTOR_STORES: Dict[str, FAISS] = {}        # composite_key -> FAISS vector_store
_RETRIEVER_META: Dict[str, List[dict]] = {}  # composite_key -> list of doc metadata


def _composite(user_id: str, thread_id: str) -> str:
    return f"{user_id}:{thread_id}"


def _sanitize_path_component(component: str) -> str:
    """Sanitize user_id or thread_id to prevent directory traversal attacks."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", (component or "").strip())
    return cleaned or "default"


def get_vector_store_path(user_id: str, thread_id: str) -> Path:
    """
    Return the isolated filesystem path for a (user_id, thread_id) vector store.
    Guarantees confinement within the configured vector store directory.
    Rejects any path traversal characters explicitly.
    """
    raw_user = (user_id or "").strip()
    raw_thread = (thread_id or "").strip()
    if any(seq in raw_user or seq in raw_thread for seq in ("..", "/", "\\")):
        raise ValueError(
            f"Path traversal detected in identifier: user='{user_id}', thread='{thread_id}'"
        )

    base_dir = Path(settings.vector_store_dir).resolve()
    safe_user = _sanitize_path_component(raw_user)
    safe_thread = _sanitize_path_component(raw_thread)
    target_dir = (base_dir / safe_user / safe_thread).resolve()

    if not target_dir.is_relative_to(base_dir):
        raise ValueError(
            f"Path traversal detected: user='{user_id}', thread='{thread_id}'"
        )
    return target_dir


def save_vector_store(
    vector_store: FAISS,
    user_id: str,
    thread_id: str,
    metadata: List[dict],
) -> None:
    """
    Persist a FAISS vector store and its document metadata to isolated disk storage.
    """
    target_dir = get_vector_store_path(user_id, thread_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index and docstore files (index.faiss, index.pkl)
    vector_store.save_local(str(target_dir))

    # Save metadata JSON
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Persisted FAISS vector store and metadata to %s", target_dir)


def load_vector_store(user_id: str, thread_id: str) -> Optional[FAISS]:
    """
    Restore a persisted FAISS vector store from disk for a specific (user_id, thread_id).
    Returns None if missing, corrupted, or untrusted.
    """
    key = _composite(user_id, thread_id)
    if key in _VECTOR_STORES:
        return _VECTOR_STORES[key]

    try:
        target_dir = get_vector_store_path(user_id, thread_id)
    except ValueError as exc:
        logger.warning("Invalid storage path: %s", exc)
        return None

    index_file = target_dir / "index.faiss"
    pkl_file = target_dir / "index.pkl"

    if not (index_file.exists() and pkl_file.exists()):
        return None

    try:
        embeddings = get_embeddings()
        # allow_dangerous_deserialization=True is strictly confined to local paths
        # validated by get_vector_store_path() within settings.vector_store_dir.
        vector_store = FAISS.load_local(
            str(target_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _VECTOR_STORES[key] = vector_store
        logger.info("Restored FAISS index from disk for %s at %s", key, target_dir)
        return vector_store
    except Exception as exc:
        logger.error(
            "Failed to load FAISS index from %s (file may be corrupted): %s",
            target_dir, exc,
        )
        return None


def load_metadata_from_disk(user_id: str, thread_id: str) -> List[dict]:
    """Load metadata.json from disk if present."""
    try:
        target_dir = get_vector_store_path(user_id, thread_id)
        meta_file = target_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as exc:
        logger.warning("Could not read metadata from disk: %s", exc)
    return []


def delete_vector_store(user_id: str, thread_id: str) -> bool:
    """
    Delete the persisted FAISS index directory for a (user_id, thread_id) pair
    and evict the in-memory caches.

    Returns True if the directory existed and was deleted, False if it did not
    exist (or could not be validated). Never raises — errors are logged.

    Safety
    ------
    - Path is validated through ``get_vector_store_path()`` which rejects ``..``,
      ``/``, and ``\\`` in IDs and verifies the resolved path is inside the
      configured ``vector_store_dir`` base directory.
    - Only the in-process caches and the specific directory on disk are touched.
    """
    key = _composite(user_id, thread_id)

    # Evict from all in-memory caches regardless of disk state
    _RETRIEVERS.pop(key, None)
    _VECTOR_STORES.pop(key, None)
    _RETRIEVER_META.pop(key, None)

    try:
        target_dir = get_vector_store_path(user_id, thread_id)
    except ValueError as exc:
        logger.warning("delete_vector_store: invalid path rejected: %s", exc)
        return False

    if not target_dir.exists():
        logger.info(
            "delete_vector_store: no index at %s (already deleted or never created)",
            target_dir,
        )
        return False

    try:
        import shutil
        shutil.rmtree(target_dir)
        logger.info("Deleted FAISS vector store at %s", target_dir)
        # Remove the parent user directory if it is now empty
        parent = target_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            logger.info("Removed empty user vector store dir at %s", parent)
        return True
    except Exception as exc:
        logger.error("delete_vector_store failed for %s: %s", key, exc)
        return False


def get_retriever(user_id: str, thread_id: str) -> Optional[Any]:
    """
    Get the FAISS retriever for (user_id, thread_id).
    Checks memory cache first, then loads from disk if previously saved.
    """
    key = _composite(user_id, thread_id)
    if key in _RETRIEVERS:
        return _RETRIEVERS[key]

    # Attempt to restore from disk
    vector_store = load_vector_store(user_id, thread_id)
    if vector_store is not None:
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )
        _RETRIEVERS[key] = retriever
        if key not in _RETRIEVER_META:
            _RETRIEVER_META[key] = load_metadata_from_disk(user_id, thread_id)
        return retriever

    return None


def get_retriever_meta(user_id: str, thread_id: str) -> List[dict]:
    """
    Get the document metadata list for (user_id, thread_id).
    Checks memory cache first, then loads from disk if previously saved.
    """
    key = _composite(user_id, thread_id)
    if key in _RETRIEVER_META and _RETRIEVER_META[key]:
        return _RETRIEVER_META[key]

    meta = load_metadata_from_disk(user_id, thread_id)
    if meta:
        _RETRIEVER_META[key] = meta
        return meta

    return []


# ---------------------------------------------------------------------------
# Multi-Format Document Extraction Helpers
# Supported formats: PDF, DOCX, TXT, Markdown (.md)
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def _extract_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract pages from a PDF file as langchain Document objects."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        try:
            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()
        except Exception as load_exc:
            logger.error("PyPDFLoader failed for '%s': %s", filename, load_exc)
            raise ValueError(
                f"Could not read '{filename}'. Ensure it is a valid, non-encrypted PDF."
                f" Error: {load_exc}"
            ) from load_exc

        if not raw_docs:
            raise ValueError(f"'{filename}' appears empty — no pages were extracted.")

        docs = []
        for doc in raw_docs:
            if doc.page_content and doc.page_content.strip():
                doc.metadata["source"] = filename
                doc.metadata["doc_type"] = "pdf"
                docs.append(doc)

        if not docs:
            raise ValueError(
                f"'{filename}' has {len(raw_docs)} page(s) but no extractable text. "
                "This is likely a scanned/image-only PDF. Please use a text-based PDF."
            )

        return docs

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _extract_docx(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from a Word document (.docx) as langchain Document objects."""
    try:
        # First attempt: if python-docx is available, use it
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_texts = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if row_texts:
                    paragraphs.append(" | ".join(row_texts))
        full_text = "\n\n".join(paragraphs).strip()
    except ImportError:
        # Robust fallback using Python's built-in zipfile and ElementTree
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise ValueError(
                        f"'{filename}' does not contain 'word/document.xml'. "
                        "It may not be a valid .docx file."
                    )
                xml_content = zf.read("word/document.xml")

            root = ET.fromstring(xml_content)
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            p_tag = f"{{{ns}}}p"
            t_tag = f"{{{ns}}}t"
            br_tag = f"{{{ns}}}br"

            paragraphs = []
            for p in root.iter(p_tag):
                parts = []
                for elem in p.iter():
                    if elem.tag == t_tag and elem.text:
                        parts.append(elem.text)
                    elif elem.tag == br_tag:
                        parts.append("\n")
                line = "".join(parts).strip()
                if line:
                    paragraphs.append(line)

            full_text = "\n\n".join(paragraphs).strip()
        except zipfile.BadZipFile as bz_exc:
            raise ValueError(
                f"Could not read '{filename}'. File is not a valid or uncorrupted DOCX archive."
            ) from bz_exc
        except ET.ParseError as pe_exc:
            raise ValueError(
                f"Could not parse XML structure of '{filename}': {pe_exc}"
            ) from pe_exc
    except Exception as exc:
        raise ValueError(f"Could not read Word document '{filename}': {exc}") from exc

    if not full_text:
        raise ValueError(f"'{filename}' appears empty — no text could be extracted.")

    return [
        Document(
            page_content=full_text,
            metadata={"source": filename, "doc_type": "docx", "page": None},
        )
    ]


def _extract_txt(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from a plain text (.txt) file."""
    text = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    clean_text = text.strip()
    if not clean_text:
        raise ValueError(f"'{filename}' appears empty — no text content found.")

    return [
        Document(
            page_content=clean_text,
            metadata={"source": filename, "doc_type": "txt", "page": None},
        )
    ]


def _extract_md(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from a Markdown (.md) file."""
    text = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    clean_text = text.strip()
    if not clean_text:
        raise ValueError(f"'{filename}' appears empty — no Markdown content found.")

    return [
        Document(
            page_content=clean_text,
            metadata={"source": filename, "doc_type": "md", "page": None},
        )
    ]


def ingest_document(
    file_bytes: bytes,
    user_id: str,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Load, split, embed, and store a document (PDF, DOCX, TXT, Markdown) for a
    specific (user, thread).
    Appends document chunks to any existing conversation FAISS index instead
    of replacing it.
    Persists the resulting unified FAISS index and metadata to disk so it survives restarts.
    Prevents duplicate ingestion using SHA-256 content hashes.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File '{filename}' exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
        )

    target_filename = (filename or "document.txt").strip()
    ext = target_filename.rsplit(".", 1)[-1].lower() if "." in target_filename else ""
    if ext not in ("pdf", "docx", "txt", "md", "markdown"):
        raise ValueError(
            f"Unsupported file format '.{ext}' for '{target_filename}'. "
            "Supported formats: .pdf, .docx, .txt, .md"
        )
    if ext == "markdown":
        ext = "md"

    key = _composite(user_id, thread_id)
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # ── Check for existing index to avoid redundant re-embedding ──
    existing_meta = get_retriever_meta(user_id, thread_id)
    for doc_meta in existing_meta:
        is_same_hash = doc_meta.get("content_hash") == file_hash
        is_same_name = doc_meta.get("filename") == target_filename and doc_meta.get("chunks", 0) > 0
        if is_same_hash or is_same_name:
            # Ensure retriever is active in memory
            retriever = get_retriever(user_id, thread_id)
            if retriever is not None:
                logger.info(
                    "Document '%s' already indexed for %s (hash=%s). Reusing index without re-embedding.",
                    target_filename, key, file_hash[:8],
                )
                return doc_meta

    embeddings = get_embeddings()

    # ── Step 1: Extract Document objects based on file type ──
    if ext == "pdf":
        docs = _extract_pdf(file_bytes, target_filename)
    elif ext == "docx":
        docs = _extract_docx(file_bytes, target_filename)
    elif ext == "txt":
        docs = _extract_txt(file_bytes, target_filename)
    elif ext == "md":
        docs = _extract_md(file_bytes, target_filename)
    else:
        raise ValueError(f"Unsupported document format: {ext}")

    # ── Step 2: Split into chunks ──
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(f"'{target_filename}' produced no text chunks after splitting.")

    # Ensure metadata on all chunks
    for i, c in enumerate(chunks):
        c.metadata["source"] = target_filename
        c.metadata["doc_type"] = ext
        if ext != "pdf":
            c.metadata["page"] = None
        c.metadata["chunk_id"] = i

    # ── Step 3: Build or extend FAISS index incrementally ──
    try:
        existing_vs = load_vector_store(user_id, thread_id)
        if existing_vs is not None:
            existing_vs.add_documents(chunks)
            vector_store = existing_vs
            logger.info("Appended %d chunks from '%s' to existing FAISS index for %s", len(chunks), target_filename, key)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info("Created new FAISS index with %d chunks from '%s' for %s", len(chunks), target_filename, key)
    except Exception as faiss_exc:
        logger.error("FAISS indexing failed for '%s': %s", target_filename, faiss_exc)
        raise ValueError(f"Failed to build search index for '{target_filename}': {faiss_exc}") from faiss_exc

    # Cache in memory
    _VECTOR_STORES[key] = vector_store
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    _RETRIEVERS[key] = retriever

    meta = {
        "filename": target_filename,
        "doc_type": ext,
        "documents": len(docs),
        "non_empty_pages": len(docs) if ext == "pdf" else 1,
        "chunks": len(chunks),
        "content_hash": file_hash,
    }

    # Update metadata list (avoid duplicates: remove previous record for same filename or hash if any)
    meta_list = [
        m for m in get_retriever_meta(user_id, thread_id)
        if m.get("filename") != target_filename and m.get("content_hash") != file_hash
    ]
    meta_list.append(meta)
    _RETRIEVER_META[key] = meta_list

    # ── Step 4: Persist to disk ──
    try:
        save_vector_store(vector_store, user_id, thread_id, meta_list)
    except Exception as save_exc:
        logger.error("Failed to persist vector store to disk: %s", save_exc)

    logger.info(
        "Document ingested OK for %s: '%s' (%s, %d chunks)",
        key, meta["filename"], meta["doc_type"], meta["chunks"],
    )
    return meta


def ingest_pdf(
    file_bytes: bytes,
    user_id: str,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Backward-compatible wrapper around ingest_document for PDF files.
    Preserves exact function signature and return structure.
    """
    return ingest_document(
        file_bytes=file_bytes,
        user_id=user_id,
        thread_id=thread_id,
        filename=filename or "document.pdf",
    )


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
    Retrieve relevant passages from uploaded documents (PDF, DOCX, TXT, Markdown)
    for this conversation. Always pass user_id and thread_id when calling this tool.
    Each passage includes its source filename, human page number (when available),
    and document type so answers can cite the exact source.
    """
    retriever = get_retriever(user_id, thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this conversation. Upload a document (PDF, DOCX, TXT, MD) first.",
            "query": query,
        }

    try:
        results = retriever.invoke(query)
        passages = []
        for doc in results:
            page = doc.metadata.get("page")
            source = doc.metadata.get("source") or doc.metadata.get("filename") or "document"
            doc_type = doc.metadata.get("doc_type") or "unknown"
            passages.append(
                {
                    "content": doc.page_content,
                    "source": source,
                    # PyPDFLoader pages are 0-indexed internally — show human page numbers (1-indexed).
                    "page": (page + 1) if isinstance(page, int) else None,
                    "doc_type": doc_type,
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
