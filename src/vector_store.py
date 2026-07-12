"""Phase 2 — RAG document pipeline.

Responsibilities
----------------
- Load the loan policy document with TextLoader.
- Split it into overlapping chunks with RecursiveCharacterTextSplitter.
- Generate embeddings with the configured HuggingFace sentence-transformer.
- Build and persist a FAISS vector store.
- Expose a retriever that returns the most relevant chunks for a query.

Nothing in this module touches the LLM, prompt templates, chains, or the
eligibility rule engine — those are added in later phases.
"""

from __future__ import annotations

import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import Settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance for the configured model.

    The underlying sentence-transformer model is loaded once per call.
    Callers that need to build and then query in the same process should
    pass the returned object directly to :func:`load_retriever` via the
    ``embeddings`` parameter to avoid loading the model a second time.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_vector_store(
    settings: Settings,
) -> tuple[FAISS, HuggingFaceEmbeddings]:
    """Load, chunk, embed, and persist the loan policy document.

    Steps
    -----
    1. Load ``settings.policy_doc_path`` with TextLoader.
    2. Split into chunks (size=500 chars, overlap=50 chars).
    3. Embed every chunk with HuggingFaceEmbeddings.
    4. Build a FAISS index and save it to ``settings.vector_store_dir``.

    Parameters
    ----------
    settings:
        Runtime settings (see :mod:`src.config`).

    Returns
    -------
    tuple[FAISS, HuggingFaceEmbeddings]
        The in-memory vector store (also persisted to disk) and the
        embeddings instance.  Pass the embeddings object to
        :func:`load_retriever` when building and querying in the same
        process so the model is only loaded once.
    """
    # 1. Load document
    loader = TextLoader(settings.policy_doc_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n---\n", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed
    embeddings = _get_embeddings(settings)

    # 4. Build FAISS index and persist
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(settings.vector_store_dir, exist_ok=True)
    vector_store.save_local(settings.vector_store_dir)

    return vector_store, embeddings


def load_retriever(
    settings: Settings,
    k: int = 3,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> VectorStoreRetriever:
    """Load the persisted FAISS index and return a retriever.

    Parameters
    ----------
    settings:
        Runtime settings (see :mod:`src.config`).
    k:
        Number of most-relevant chunks to retrieve per query.
    embeddings:
        Optional pre-built embeddings instance.  When provided the model
        is not loaded again — useful when calling this function right
        after :func:`build_vector_store` in the same process.

    Returns
    -------
    VectorStoreRetriever
        A LangChain retriever backed by the persisted FAISS index.

    Raises
    ------
    FileNotFoundError
        If the vector store has not been built yet (call
        :func:`build_vector_store` first).
    """
    index_file = os.path.join(settings.vector_store_dir, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"Vector store not found at '{settings.vector_store_dir}'. "
            "Run build_vector_store() first."
        )

    if embeddings is None:
        embeddings = _get_embeddings(settings)

    vector_store = FAISS.load_local(
        settings.vector_store_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
