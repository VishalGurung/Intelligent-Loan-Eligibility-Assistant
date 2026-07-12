"""Phase 2B — RetrievalQA chain for Loan Policy Q&A.

Responsibilities
----------------
- Instantiate the Claude LLM using the configured Anthropic-compatible
  gateway (base URL + API key from Settings).
- Wrap the Phase-2 retriever in a LangChain ``RetrievalQA`` chain so that
  every user question is answered with context drawn from the loan policy
  document.

Nothing in this module implements eligibility rules, PromptTemplate for
eligibility, agents, or conversation memory — those arrive in later phases.

LangChain components used (matching the case study Feature 1 list)
-------------------------------------------------------------------
- ChatAnthropic (LLM)
- RetrievalQA chain (chain_type="stuff")
- Custom system prompt scoping answers to the policy document
"""

from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import Settings

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful loan policy assistant for a bank. "
    "Use ONLY the context provided below — extracted from the bank's official "
    "loan policy document — to answer the customer's question. "
    "If the answer is not found in the context, say: "
    "'I don't have that information in the current policy document.' "
    "Be concise, accurate, and professional.\n\n"
    "Context:\n{context}"
)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_llm(settings: Settings) -> ChatAnthropic:
    """Instantiate the Claude LLM from Settings.

    Parameters
    ----------
    settings:
        Runtime settings (see :mod:`src.config`).

    Returns
    -------
    ChatAnthropic
        A configured LangChain LLM instance ready for use in chains.

    Raises
    ------
    ValueError
        If ``ANTHROPIC_API_KEY`` is not set in the environment / `.env`.
    """
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file and restart."
        )
    if not settings.anthropic_base_url:
        raise ValueError(
            "ANTHROPIC_BASE_URL is not set. "
            "This project uses a vendor gateway — the default Anthropic "
            "API endpoint is not used. Set ANTHROPIC_BASE_URL in .env."
        )

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        temperature=0,
        max_tokens=1024,
    )


def build_rag_chain(
    settings: Settings,
    retriever: VectorStoreRetriever,
) -> RetrievalQA:
    """Build a RetrievalQA chain backed by the FAISS retriever.

    The chain uses chain_type="stuff" — all retrieved chunks are stuffed
    into a single prompt alongside the question, which is appropriate for
    the size of the loan policy document.

    Parameters
    ----------
    settings:
        Runtime settings (see :mod:`src.config`).
    retriever:
        A LangChain retriever returned by
        :func:`src.vector_store.load_retriever`.

    Returns
    -------
    RetrievalQA
        A runnable LangChain chain.  Call ``chain.invoke({"query": "..."})``
        to get an answer dict with keys ``"query"`` and ``"result"``.
    """
    llm = build_llm(settings)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _RAG_PROMPT},
    )
    return chain
