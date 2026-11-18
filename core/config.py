"""
core/config.py
==============
Central configuration for the Multi-Agent AI Research Chatbot.

All environment variables are read here. Import `settings` anywhere
in the application instead of calling os.getenv() directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    # ── LLM ─────────────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    )
    groq_temperature: float = field(
        default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0"))
    )

    # ── Embeddings (local, no API key required) ──────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # ── Database ─────────────────────────────────────────────────────────────
    db_path: str = field(
        default_factory=lambda: os.getenv("DB_PATH", "chatbot.db")
    )

    # ── Persistent Vector Store ──────────────────────────────────────────────
    vector_store_dir: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE_DIR", "data/vector_stores")
    )

    # ── Alpha Vantage (optional) ─────────────────────────────────────────────
    alphavantage_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY") or None
    )

    # ── MCP (optional) ───────────────────────────────────────────────────────
    mcp_server_url: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "MCP_SERVER_URL", "http://127.0.0.1:8000/mcp"
        )
    )
    mcp_enabled: bool = field(
        default_factory=lambda: os.getenv("MCP_ENABLED", "true").lower() == "true"
    )

    # ── LangSmith tracing (optional) ─────────────────────────────────────────
    langsmith_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY") or None
    )
    langsmith_project: str = field(
        default_factory=lambda: os.getenv(
            "LANGSMITH_PROJECT", "Multi-Agent-AI-Chatbot"
        )
    )
    langsmith_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower()
        == "true"
    )

    # ── HuggingFace (optional) ───────────────────────────────────────────────
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN") or None
    )

    # ── App behaviour ────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    max_pdf_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_PDF_SIZE_MB", "50"))
    )

    def validate(self) -> None:
        """Raise ValueError for missing required settings."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        logger.info("Configuration validated successfully.")
        logger.info("Groq model: %s", self.groq_model)
        logger.info("Database: %s", self.db_path)
        logger.info("Vector store directory: %s", self.vector_store_dir)
        logger.info("MCP enabled: %s, URL: %s", self.mcp_enabled, self.mcp_server_url)
        logger.info(
            "Alpha Vantage stock tool: %s",
            "enabled" if self.alphavantage_api_key else "disabled (no key)",
        )
        logger.info(
            "LangSmith tracing: %s",
            "enabled" if self.langsmith_tracing else "disabled",
        )


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------

settings = Settings()
