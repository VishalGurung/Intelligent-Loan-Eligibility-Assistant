"""Central configuration loaded from environment variables.

Phase 1 only: this module loads settings from a `.env` file (if present) so
later phases can build the LLM, embeddings, and vector store from a single
place. It intentionally contains no LangChain chains, agents, or loan logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the assistant."""

    anthropic_api_key: str | None
    anthropic_base_url: str | None
    anthropic_model: str
    embedding_model: str
    vector_store_dir: str
    policy_doc_path: str


def get_settings() -> Settings:
    """Read settings from the environment."""
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        vector_store_dir=os.getenv("VECTOR_STORE_DIR", "vectorstore"),
        policy_doc_path=os.getenv("POLICY_DOC_PATH", "data/loan_policy.md"),
    )
