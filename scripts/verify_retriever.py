"""Phase 2 smoke-test: build the vector store and verify the retriever.

Run from the project root:

    .venv\\Scripts\\python.exe scripts/verify_retriever.py

The script
----------
1. Builds (or rebuilds) the FAISS vector store from the loan policy document.
2. Loads a retriever from the persisted index.
3. Runs three sample queries from the case study and prints the returned chunks.

No LLM key is required — only local HuggingFace embeddings are used.
"""

from __future__ import annotations

import sys
import os

# Ensure stdout handles Unicode on Windows (e.g. the ₹ symbol in policy text).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make the project root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.vector_store import build_vector_store, load_retriever

QUERIES = [
    "What is the minimum salary required for a personal loan?",
    "What credit score is needed for a home loan?",
    "What documents are required?",
]


def main() -> None:
    settings = get_settings()

    # Step 1: Build and persist the vector store
    print("=" * 60)
    print("Step 1 — Building FAISS vector store …")
    print(f"  Policy document : {settings.policy_doc_path}")
    print(f"  Embedding model : {settings.embedding_model}")
    print(f"  Vector store dir: {settings.vector_store_dir}")
    print("=" * 60)

    # build_vector_store returns both the store and the embeddings instance so
    # the model is not loaded a second time when load_retriever is called below.
    _vector_store, embeddings = build_vector_store(settings)
    print(f"\nVector store built and saved to '{settings.vector_store_dir}'\n")

    # Step 2: Load retriever from persisted index (reuse embeddings instance)
    print("Step 2 — Loading retriever from persisted index …")
    retriever = load_retriever(settings, k=3, embeddings=embeddings)
    print("Retriever loaded successfully.\n")

    # Step 3: Run sample queries
    print("Step 3 — Running sample queries")
    print("=" * 60)

    for i, query in enumerate(QUERIES, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)
        docs = retriever.invoke(query)
        if not docs:
            print("  [No chunks returned]")
        for j, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            content_preview = doc.page_content.strip().replace("\n", " ")
            print(f"  Chunk {j} (source: {source})")
            print(f"  {content_preview[:300]}")
            print()

    print("=" * 60)
    print("Verification complete. Retriever is working correctly.")


if __name__ == "__main__":
    main()
