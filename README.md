# Intelligent Loan Eligibility Assistant (LangChain)

An AI-powered assistant that answers questions about loan products, checks loan
eligibility against bank policies, and provides personalized recommendations.
Built with **LangChain**, using **Claude** (Anthropic-compatible) as the LLM,
local **HuggingFace** embeddings, and a **FAISS** vector store, with a
**Streamlit** frontend.

> **Status: Phase 1 — Project Setup.** Only the project scaffolding, virtual
> environment, dependencies, sample policy document, and configuration exist.
> The RAG Q&A, eligibility checker, recommendations, and agent are implemented
> in later phases.

## Project structure

```
loan-eligibility-assistant/
├── app.py                # Streamlit entrypoint (Phase 1: setup check only)
├── data/
│   └── loan_policy.md    # Sample bank loan policy (used by RAG in later phases)
├── src/
│   ├── __init__.py
│   └── config.py         # Loads settings from environment (.env)
├── requirements.txt      # Pinned dependencies
├── .env.example          # Template for environment variables (no real keys)
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10+
- A Claude-compatible API key (Anthropic or a vendor Anthropic-compatible gateway)

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if your
# vendor uses an Anthropic-compatible endpoint).
```

## Run

```bash
streamlit run app.py
```

The Phase 1 app renders a setup-check page confirming the configured LLM,
embeddings, and vector store, and whether an API key is detected.

## Verify the setup (no API key required)

```bash
source .venv/bin/activate
python -c "import langchain, langchain_anthropic, langchain_community, langchain_huggingface, faiss, sentence_transformers, streamlit, dotenv; print('All imports OK')"
python -c "from src.config import get_settings; print(get_settings())"
```

## Configuration reference

See `.env.example` for all supported variables:

| Variable             | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| `ANTHROPIC_API_KEY`  | API key for the Claude-compatible LLM.                       |
| `ANTHROPIC_BASE_URL` | Optional Anthropic-compatible endpoint override.             |
| `ANTHROPIC_MODEL`    | Claude model name (Haiku / Sonnet / Opus).                   |
| `EMBEDDING_MODEL`    | Local HuggingFace sentence-transformers model for embeddings.|
| `VECTOR_STORE_DIR`   | Directory for the FAISS index (created in later phases).     |
| `POLICY_DOC_PATH`    | Path to the policy document used by the RAG feature.         |
