"""Intelligent Loan Eligibility Assistant (LangChain).

Package root.
- Phase 1:  configuration and scaffolding (src.config).
- Phase 2:  RAG document pipeline — document loading, chunking, embeddings,
            FAISS vector store, and retriever (src.vector_store).
- Phase 2B: RetrievalQA chain — Claude LLM + retriever → policy Q&A
            (src.rag_chain).
- Phase 3:  Loan Eligibility Checker — business rules + PromptTemplate +
            LLMChain → eligibility decision with reasons (src.eligibility_checker).
"""

__version__ = "0.1.0"
