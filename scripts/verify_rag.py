"""Phase 2B smoke-test: RetrievalQA chain end-to-end verification.

Run from the project root AFTER adding ANTHROPIC_API_KEY to .env:

    $env:PYTHONIOENCODING="utf-8"
    .venv\\Scripts\\python.exe scripts/verify_rag.py

The script
----------
1. Loads settings (including the API key from .env).
2. Loads the persisted FAISS retriever (built in Phase 2).
3. Builds the RetrievalQA chain against the configured Claude model.
4. Runs 3 sample questions from the case study and prints answers.
"""

from __future__ import annotations

import sys
import os

# Ensure stdout handles Unicode on Windows (e.g. ₹ symbol in policy text).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make the project root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.vector_store import load_retriever
from src.rag_chain import build_rag_chain

QUERIES = [
    "What is the minimum salary required for a personal loan?",
    "What documents are required to apply for a loan?",
    "What credit score is needed for a home loan?",
]


def main() -> None:
    settings = get_settings()

    # Guard: fail fast with a clear message if the key is missing.
    if not settings.anthropic_api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Edit the file:  .env  (copy from .env.example if it doesn't exist)\n"
            "Set the variable:  ANTHROPIC_API_KEY=<your-key>\n"
            "Then re-run this script."
        )
        sys.exit(1)

    print("=" * 60)
    print("Phase 2B — RetrievalQA end-to-end verification")
    print("=" * 60)
    print(f"  Model      : {settings.anthropic_model}")
    print(f"  Base URL   : {settings.anthropic_base_url or '(default Anthropic API)'}")
    print(f"  Vector store: {settings.vector_store_dir}")
    print()

    # Step 1: Load retriever from the persisted Phase-2 FAISS index.
    print("Step 1 — Loading retriever …")
    retriever = load_retriever(settings, k=3)
    print("Retriever ready.\n")

    # Step 2: Build the RetrievalQA chain.
    print("Step 2 — Building RetrievalQA chain …")
    chain = build_rag_chain(settings, retriever)
    print("Chain ready.\n")

    # Step 3: Run sample queries.
    print("Step 3 — Running sample queries")
    print("=" * 60)

    for i, query in enumerate(QUERIES, 1):
        print(f"\nQ{i}: {query}")
        print("-" * 60)
        response = chain.invoke({"query": query})
        answer = response.get("result", "").strip()
        print(f"A{i}: {answer}")
        sources = response.get("source_documents", [])
        if sources:
            refs = {doc.metadata.get("source", "unknown") for doc in sources}
            print(f"     [Sources: {', '.join(sorted(refs))}]")

    print()
    print("=" * 60)
    print("Verification complete.")


if __name__ == "__main__":
    main()
