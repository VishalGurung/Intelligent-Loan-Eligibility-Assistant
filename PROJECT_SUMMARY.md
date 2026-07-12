# Intelligent Loan Eligibility Assistant — Project Summary

> **Purpose of this document:** A self-contained reference for revision, interview
> preparation, and onboarding onto similar LangChain RAG projects.  
> Written after completing Phases 1 → 2 → 2B → 3 of the case study.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Overall Architecture](#2-overall-architecture)
3. [File Map — Purpose and Interactions](#3-file-map--purpose-and-interactions)
4. [Key LangChain Components Used](#4-key-langchain-components-used)
5. [End-to-End Execution Flow](#5-end-to-end-execution-flow)
6. [Component Responsibility Matrix](#6-component-responsibility-matrix)
7. [Runtime Execution Order](#7-runtime-execution-order)
8. [Design Decisions](#8-design-decisions)
9. [Common Pitfalls and Debugging Tips](#9-common-pitfalls-and-debugging-tips)
10. [Top 10 Concepts from This Case Study](#10-top-10-concepts-from-this-case-study)
11. [Interview Questions and Answers](#11-interview-questions-and-answers)
12. [Key Takeaways for the Next Case Study](#12-key-takeaways-for-the-next-case-study)

---

## 1. Project Overview

### Business Problem

A bank receives numerous loan applications daily. Loan officers spend significant
time reviewing applicant information, validating policies, and answering customer
queries. This project builds an AI-powered assistant to automate those tasks.

### Objective

Build an **Intelligent Loan Eligibility Assistant** using LangChain that can:

1. Answer customer questions about loan products (**Feature 1 — RAG Q&A**)
2. Check basic loan eligibility based on bank policies (**Feature 2 — Eligibility Checker**)
3. Explain why a loan may be approved or rejected
4. Provide next steps for applicants

### Technology Stack

| Layer | Technology |
|---|---|
| LLM | Claude (Anthropic-compatible, via vendor gateway) |
| Framework | LangChain 0.3.x |
| Vector Database | FAISS (local, CPU) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| Frontend | Streamlit |
| Backend | Python 3.12 |
| Configuration | `python-dotenv` |

### Implementation Phases

| Phase | Description | Key Deliverable |
|---|---|---|
| 1 | Project scaffolding | `src/config.py`, `.env.example` |
| 2 | RAG document pipeline | `src/vector_store.py`, FAISS index on disk |
| 2B | RetrievalQA chain | `src/rag_chain.py`, policy Q&A working end-to-end |
| 3 | Loan Eligibility Checker | `src/eligibility_checker.py`, full Streamlit UI |

---

## 2. Overall Architecture

### High-Level System Diagram

```
                        User (Browser)
                             |
                     Streamlit  app.py
                             |
              ┌──────────────┴──────────────┐
              │                             │
     Tab 1: Policy Q&A          Tab 2: Eligibility Checker
              │                             │
     RetrievalQA Chain           Rule Engine (pure Python)
              │                             │
    ┌─────────┴────────┐         PromptTemplate + LLMChain
    │                  │                    │
Retriever          Claude LLM          Claude LLM
    │               (gateway)           (gateway)
FAISS Index     https://llmgwext-    https://llmgwext-
(vectorstore/)  wp.tekstac.com       wp.tekstac.com
    │
HuggingFace
Embeddings
(local model)
    |
data/loan_policy.md
```

### Data Flow Summary

```
loan_policy.md
     │
     ▼
TextLoader → raw Document
     │
     ▼
RecursiveCharacterTextSplitter → list of chunk Documents
     │
     ▼
HuggingFaceEmbeddings → dense vectors (384-dim)
     │
     ▼
FAISS.from_documents() → in-memory index
     │
     ▼
FAISS.save_local("vectorstore/") → index.faiss + index.pkl (persisted)


At query time:
User question
     │
     ▼
FAISS.as_retriever(k=3) → top-3 relevant chunks
     │
     ▼
RetrievalQA (chain_type="stuff") → chunks stuffed into prompt
     │
     ▼
ChatAnthropic (Claude) via gateway → natural language answer
```

---

## 3. File Map — Purpose and Interactions

```
project root/
│
├── app.py                        ← Streamlit UI (entrypoint)
│
├── src/
│   ├── __init__.py               ← Package marker + phase docs
│   ├── config.py                 ← Settings loaded from .env
│   ├── vector_store.py           ← Document loading, chunking, FAISS
│   ├── rag_chain.py              ← Claude LLM + RetrievalQA chain
│   └── eligibility_checker.py   ← Business rules + PromptTemplate + LLMChain
│
├── scripts/
│   ├── verify_retriever.py       ← Phase 2 smoke-test (no LLM)
│   ├── verify_rag.py             ← Phase 2B smoke-test (LLM + retriever)
│   └── verify_eligibility.py    ← Phase 3 smoke-test (eligibility checker)
│
├── data/
│   └── loan_policy.md            ← Bank policy source document
│
├── vectorstore/                  ← FAISS index (generated, not committed)
│   ├── index.faiss
│   └── index.pkl
│
├── .env                          ← Secrets (not committed)
├── .env.example                  ← Template (committed)
└── requirements.txt              ← Pinned dependencies
```

### How Files Interact

```
.env
 └─► config.py (get_settings)
          │
          ├─► vector_store.py (build_vector_store / load_retriever)
          │        └── reads: data/loan_policy.md
          │        └── writes: vectorstore/
          │
          ├─► rag_chain.py (build_llm / build_rag_chain)
          │        └── uses: load_retriever() from vector_store.py
          │
          └─► eligibility_checker.py (build_eligibility_chain / check_eligibility)
                   └── uses: build_llm() from rag_chain.py

app.py
 └── imports all four src modules
 └── exposes two Streamlit tabs
```

---

## 4. Key LangChain Components Used

### Feature 1 — Loan Policy Q&A (RAG)

| Component | Class / Function | Purpose |
|---|---|---|
| Document Loader | `TextLoader` | Load `loan_policy.md` as a LangChain `Document` |
| Text Splitter | `RecursiveCharacterTextSplitter` | Break the document into overlapping chunks |
| Embeddings | `HuggingFaceEmbeddings` | Convert text chunks to dense vectors |
| Vector Store | `FAISS` | Store and search vectors by similarity |
| Retriever | `VectorStoreRetriever` | Retrieve top-k chunks for a query |
| LLM | `ChatAnthropic` | Generate a grounded answer from retrieved context |
| Chain | `RetrievalQA` | Orchestrate retriever → prompt → LLM → answer |
| Prompt | `ChatPromptTemplate` | System + human message structure for the LLM |

### Feature 2 — Loan Eligibility Checker

| Component | Class / Function | Purpose |
|---|---|---|
| Data Model | `Applicant` (dataclass) | Structured input (name, age, salary, loan amount, credit score) |
| Rule Engine | `evaluate_rules()` | Pure Python — applies the 4 business rules, returns pass/fail per rule |
| Prompt | `PromptTemplate` | Single-string template with 7 input variables |
| Chain | `LLMChain` | Binds prompt + LLM, invoked with a dict of variables |
| LLM | `ChatAnthropic` (shared) | Generates the narrative eligibility decision |

---

## 5. End-to-End Execution Flow

### Feature 1 — Policy Q&A

```
Step 1: User types a question in Streamlit Tab 1

Step 2: app.py calls load_retriever(settings, k=3)
        └── FAISS.load_local("vectorstore/", embeddings)
        └── returns VectorStoreRetriever

Step 3: app.py calls build_rag_chain(settings, retriever)
        └── build_llm(settings) → ChatAnthropic(model, api_key, base_url)
        └── RetrievalQA.from_chain_type(llm, "stuff", retriever, prompt)

Step 4: chain.invoke({"query": user_question})
        ├── retriever.invoke(user_question)
        │   ├── HuggingFaceEmbeddings.embed_query(user_question) → query vector
        │   └── FAISS.similarity_search(query_vector, k=3) → top-3 chunks
        │
        └── "stuff" combiner: all 3 chunks joined into {context}
            └── ChatPromptTemplate filled: system(context) + human(question)
                └── ChatAnthropic.invoke(prompt) → LLM response via gateway
                    └── returns {"result": "...", "source_documents": [...]}

Step 5: Streamlit displays answer + expandable source chunks
```

### Feature 2 — Eligibility Checker

```
Step 1: User fills the form in Streamlit Tab 2 and clicks "Check Eligibility"

Step 2: app.py creates Applicant(name, age, salary, loan_amount, credit_score)

Step 3: evaluate_rules(applicant) — pure Python, no LLM
        ├── Age rule:         age >= 21
        ├── Salary rule:      salary >= 30,000
        ├── Credit rule:      credit_score >= 700
        └── Loan rule:        loan_amount <= 10 × salary
        → list of RuleResult(rule, passed, detail)

Step 4: Streamlit immediately displays per-rule pass/fail table

Step 5: build_eligibility_chain(llm) → LLMChain(prompt=_ELIGIBILITY_PROMPT, llm=llm)

Step 6: check_eligibility(applicant, chain)
        ├── evaluate_rules() → rule_summary string
        ├── overall_result = "ELIGIBLE" or "NOT ELIGIBLE"
        └── chain.invoke({name, age, salary, loan_amount, credit_score,
                          rule_summary, overall_result})
            └── PromptTemplate.format(...) → filled prompt string
                └── ChatAnthropic.invoke(prompt) → narrative decision
                    └── returns {"text": "Eligible\n\nReason:..."}

Step 7: Streamlit displays the LLM-generated narrative with reasons + next steps
```

---

## 6. Component Responsibility Matrix

| Component | Does | Does NOT do |
|---|---|---|
| `config.py` | Load env vars; expose typed `Settings` dataclass | Apply business logic; touch LangChain |
| `TextLoader` | Load the `.md` file as a `Document` object with metadata | Split, embed, or index |
| `RecursiveCharacterTextSplitter` | Split `Document` into overlapping chunks respecting separators | Embed, store, or retrieve |
| `HuggingFaceEmbeddings` | Convert text → 384-dim dense vectors using local model | Store vectors; know about the document structure |
| `FAISS` | Store vectors; answer ANN similarity queries | Understand language; call any LLM |
| `VectorStoreRetriever` | Take a text query, embed it, return top-k matching `Document` chunks | Synthesise an answer; call the LLM |
| `RetrievalQA` | Orchestrate retriever → context stuffing → LLM call → return answer | Apply business rules; access the FAISS index directly |
| `ChatAnthropic` | Generate natural language given a prompt; call the vendor gateway | Retrieve documents; apply rules; remember previous turns |
| `evaluate_rules()` | Apply 4 deterministic business rules; return pass/fail per rule | Generate explanations; call the LLM |
| `PromptTemplate` | Format a string template with input variables | Call the LLM; retrieve documents |
| `LLMChain` | Bind a prompt template to an LLM; invoke with a variable dict | Apply business rules; retrieve context |
| `app.py` | Render UI; wire all components together per user interaction | Contain business logic or LangChain chain construction |

---

## 7. Runtime Execution Order

### One-time setup (run once, persisted to disk)

```
1. load .env → Settings
2. TextLoader.load("data/loan_policy.md") → [Document]
3. RecursiveCharacterTextSplitter.split_documents([Document]) → [chunks]
4. HuggingFaceEmbeddings(model="all-MiniLM-L6-v2") → embeddings model loaded
5. FAISS.from_documents(chunks, embeddings) → in-memory FAISS index built
6. FAISS.save_local("vectorstore/") → index.faiss + index.pkl written
```

### Every query (Feature 1)

```
1. FAISS.load_local("vectorstore/", embeddings) → index loaded
2. VectorStoreRetriever.invoke(query) → [top-3 Document chunks]
3. "stuff" chain: chunks concatenated into {context}
4. ChatPromptTemplate.format(context, question) → final prompt
5. ChatAnthropic.invoke(prompt) → HTTP POST to gateway → answer text
6. Return {"result": answer, "source_documents": chunks}
```

### Every check (Feature 2)

```
1. evaluate_rules(applicant) → [RuleResult × 4] — no LLM call
2. Build rule_summary string from results
3. PromptTemplate.format(name, age, salary, ..., rule_summary) → prompt string
4. ChatAnthropic.invoke(prompt) → HTTP POST to gateway → decision text
5. Return {"eligible": bool, "rule_results": [...], "decision": str}
```

---

## 8. Design Decisions

### 1. LangChain 0.3.x pinned deliberately
`RetrievalQA` and `LLMChain` are the exact APIs named in the case study. LangChain
1.x removes both. The project pins `0.3.30` so the case study code runs as written,
even though deprecation warnings appear.

### 2. Local embeddings — no OpenAI key required
`sentence-transformers/all-MiniLM-L6-v2` runs fully offline. This eliminates an
entire class of API key errors during development and keeps embedding costs at zero.
Trade-off: the model produces 384-dim vectors vs. 1536-dim from OpenAI — slightly
less semantic precision, but more than sufficient for this document size.

### 3. FAISS over ChromaDB
Both are listed in the case study. FAISS was chosen because it:
- Is a single pip install (`faiss-cpu`)
- Persists to two plain files (`index.faiss`, `index.pkl`)
- Has zero server process overhead
- Is the de-facto standard in LangChain RAG tutorials

### 4. Business rules in pure Python, not prompted to the LLM
`evaluate_rules()` applies the four criteria deterministically in Python.
The LLM is only called to **narrate** the pre-computed result. This ensures:
- The pass/fail decision is always consistent and auditable
- The LLM cannot hallucinate a wrong eligibility outcome
- Rule evaluation is instant (no API latency)

### 5. `chunk_size=500, chunk_overlap=50` with Markdown-aware separators
The policy document has clear Markdown section headers. Custom separators
(`"\n## "`, `"\n### "`, `"\n---\n"`) ensure chunks break at section boundaries
first, preserving semantic coherence. Overlap of 50 chars ensures a sentence
split at the boundary doesn't lose context.

### 6. `chain_type="stuff"` for RetrievalQA
Three chunks of 500 chars each = ~1,500 chars of context. Claude Haiku's context
window is 200k tokens. "stuff" (concatenate all chunks into one prompt) is the
simplest and most transparent strategy for this document size.
For large documents, `map_reduce` or `refine` would be needed.

### 7. Vendor gateway — `base_url` is mandatory, not optional
The code raises `ValueError` if `ANTHROPIC_BASE_URL` is not set.
This prevents accidental fallback to the default Anthropic API, which would
fail with the vendor's API key.

### 8. `@st.cache_resource` for the LLM and retriever
Loading the HuggingFace embedding model and the FAISS index on every Streamlit
re-run would be slow. `@st.cache_resource` loads them once per server session
and reuses them across all user interactions.

---

## 9. Common Pitfalls and Debugging Tips

| Pitfall | Symptom | Fix |
|---|---|---|
| FAISS index not built | `FileNotFoundError: vectorstore/index.faiss` | Run `verify_retriever.py` first to build the index |
| Wrong `ANTHROPIC_BASE_URL` | `httpx.ConnectError` or 401 at a different host | Check `.env` — base URL must be `https://llmgwext-wp.tekstac.com` |
| Missing `ANTHROPIC_API_KEY` | `ValueError: ANTHROPIC_API_KEY is not set` | Add key to `.env`; never commit `.env` |
| `UnicodeEncodeError` on Windows | Error printing `₹` symbol | Set `$env:PYTHONIOENCODING="utf-8"` before running any script |
| `LLMChain` deprecation warning | `LangChainDeprecationWarning` in console | Expected — LLMChain is deprecated in 0.3.x but functional. Safe to ignore |
| HuggingFace rate-limit warning | `Warning: unauthenticated requests to HF Hub` | Set `HF_TOKEN` env var, or ignore — the model is cached after first download |
| Streamlit re-runs on every keypress | LLM called on every character typed | Always gate LLM calls behind `st.button()`, not `st.text_input()` change |
| `allow_dangerous_deserialization` error | `ValueError` loading FAISS | Pass `allow_dangerous_deserialization=True` to `FAISS.load_local()` — required in 0.3.x |
| Second `HuggingFaceEmbeddings()` call crashes | `OSError: Invalid argument` from tqdm | Pass the embeddings object from `build_vector_store()` into `load_retriever()` instead of constructing it twice |
| LLM hallucinates eligibility result | Wrong decision returned | Business rules must be evaluated in pure Python first; LLM only narrates the pre-computed result |

---

## 10. Top 10 Concepts from This Case Study

### 1. Retrieval-Augmented Generation (RAG)
RAG solves the problem of LLMs not having access to private or up-to-date
documents. Instead of fine-tuning, you: (a) store document chunks in a vector
database, (b) retrieve the most relevant chunks at query time, (c) inject them
into the LLM's prompt as context. The LLM then answers using that context.

### 2. Text Splitting Strategy
Raw documents cannot be fed directly to an LLM or vector store — they are too
large. A text splitter breaks them into chunks. `RecursiveCharacterTextSplitter`
tries each separator in order (headers → paragraphs → lines → words) to find
the most semantic split point. `chunk_overlap` prevents losing context at chunk
boundaries.

### 3. Dense Vector Embeddings
Embeddings are numerical representations of text where semantically similar
sentences are close together in vector space. A sentence-transformer model
converts each chunk and each query to a vector of fixed dimension (384 for
`all-MiniLM-L6-v2`). Similarity is measured by cosine distance.

### 4. FAISS — Approximate Nearest Neighbour Search
FAISS (Facebook AI Similarity Search) indexes dense vectors for fast retrieval.
Given a query vector, it returns the k most similar vectors (and their associated
documents) in milliseconds, even over millions of entries. For local use,
`faiss-cpu` persists to two files.

### 5. The `stuff` Chain Type
`chain_type="stuff"` means all retrieved chunks are "stuffed" into a single
prompt call. It is the simplest strategy: fast, transparent, and works well
when the total context is small. Alternative chain types (`map_reduce`,
`refine`, `map_rerank`) handle larger contexts at the cost of multiple LLM calls.

### 6. PromptTemplate and Variable Injection
`PromptTemplate` is a LangChain utility that wraps an f-string-like template
with named `{variables}`. At runtime, a dict is passed to `.format()` or
`.invoke()` to produce the final string sent to the LLM. It separates prompt
design from business logic.

### 7. LLMChain
`LLMChain` = `PromptTemplate` + `LLM`. It is the basic unit of LangChain v1
pipelines. You call `.invoke(input_dict)` and receive a dict with a `"text"` key.
Deprecated in 0.3.x in favour of LCEL (`prompt | llm`), but still functional.

### 8. Separation of Rule Logic from LLM
A critical design principle in BFSI AI: **never** let an LLM make deterministic
business decisions. Rules (age, salary, credit score thresholds) are evaluated
deterministically in code. The LLM only communicates the outcome in natural
language. This makes the system auditable, consistent, and regulation-ready.

### 9. `@st.cache_resource` for Expensive Objects
Streamlit re-runs the entire script on every user interaction. Without caching,
the embedding model and FAISS index would reload on every button click.
`@st.cache_resource` caches objects that are expensive to create (models,
database connections) for the lifetime of the Streamlit server session.

### 10. Environment-Based Configuration
All sensitive values (API keys, base URLs) and environment-specific paths live
in `.env`, loaded by `python-dotenv`. The code never hardcodes secrets.
`.env.example` is committed as a safe template. `.env` is always in `.gitignore`.

---

## 11. Interview Questions and Answers

### LangChain & RAG

**Q: What is RAG and why is it preferred over fine-tuning for this use case?**  
A: RAG (Retrieval-Augmented Generation) retrieves relevant document chunks at
query time and injects them into the LLM's prompt. Fine-tuning bakes knowledge
into the model weights — expensive, slow to update, and prone to hallucination.
RAG is preferred when the knowledge base (e.g. a bank policy) changes frequently,
because you only need to rebuild the vector index, not retrain the model.

---

**Q: Explain the role of a text splitter. What happens if chunks are too large or too small?**  
A: The text splitter breaks a document into chunks that fit within a prompt and
represent a coherent unit of meaning.  
- Too large: exceeds the LLM context window or dilutes relevance by including
  unrelated content.  
- Too small: loses the surrounding context needed to answer a question (e.g. a
  policy condition split from its associated threshold).  
`chunk_overlap` mitigates boundary loss.

---

**Q: Why use local HuggingFace embeddings instead of OpenAI embeddings?**  
A: Local embeddings (a) require no API key, (b) have zero cost at inference,
(c) run offline, and (d) are deterministic across runs. For a document of this
size (102 lines), `all-MiniLM-L6-v2` provides more than sufficient semantic
quality.

---

**Q: What is `chain_type="stuff"` and when would you not use it?**  
A: "stuff" concatenates all retrieved chunks into a single prompt call. Suitable
when total context is small (< ~100k tokens). You would not use it for large
document sets where stuffing exceeds the context window. Alternatives: `map_reduce`
(summarise each chunk independently then combine), `refine` (iteratively refine
an answer over each chunk).

---

**Q: What is `allow_dangerous_deserialization=True` and why is it required?**  
A: FAISS persists its metadata (document objects, sources) as Python pickle files
(`index.pkl`). Pickle can execute arbitrary code, so LangChain 0.3.x added this
explicit opt-in flag to warn users they are loading a pickle they must trust.
In production, only load FAISS indexes that your own code wrote.

---

**Q: What is the difference between `PromptTemplate` and `ChatPromptTemplate`?**  
A: `PromptTemplate` produces a single string. Used with older LLMs or `LLMChain`
for single-turn text completion.  
`ChatPromptTemplate` produces a list of messages (`system`, `human`, `assistant`).
Used with chat models (`ChatAnthropic`, `ChatOpenAI`) that expect a conversation
structure. In this project, `ChatPromptTemplate` is used for RAG (Feature 1)
and `PromptTemplate` is used for eligibility checking (Feature 2).

---

**Q: Why is `LLMChain` deprecated in LangChain 0.3.x and what replaces it?**  
A: The LangChain team introduced LCEL (LangChain Expression Language) in 0.1.x
as a composable, streaming-ready alternative. `prompt | llm` is the LCEL
equivalent of `LLMChain(prompt=prompt, llm=llm)`. `LLMChain` was retained for
backward compatibility but is removed in 1.0.

---

### System Design

**Q: Why are business rules evaluated in Python rather than prompted to the LLM?**  
A: Determinism, auditability, and speed. A rule like "credit_score >= 700" always
gives the same answer in Python. An LLM might reason differently across runs,
especially with slight prompt changes. In regulated industries (banking, finance,
insurance), decisions must be explainable and reproducible. The LLM's role is
communication, not decision-making.

---

**Q: How would you scale this system to handle 10,000 policy documents?**  
A: Replace FAISS (in-memory, single-file) with a production vector store such as
Pinecone, Weaviate, or pgvector (Postgres extension). Use a more capable embedding
model. Switch `chain_type` from "stuff" to "map_reduce" or "refine". Add metadata
filters so the retriever scopes queries to the right product category.

---

**Q: What is `@st.cache_resource` and why is it used here?**  
A: Streamlit re-runs the entire Python script on every user interaction. Without
caching, the HuggingFace embedding model (~90 MB) and the FAISS index would reload
on every button click. `@st.cache_resource` creates the object once per server
session and returns the same instance on subsequent calls, eliminating that
overhead. It is used for shared, stateful objects (models, DB connections).

---

**Q: What is the FAISS index made up of and what are the two persisted files?**  
A: The FAISS index stores dense vector representations of each document chunk.
- `index.faiss` — the binary vector index used for similarity search.
- `index.pkl` — a Python pickle of the associated `InMemoryDocstore` (maps
  vector IDs back to the original `Document` objects and their metadata).
Both files are needed to reconstruct the full `FAISS` object.

---

## 12. Key Takeaways for the Next Case Study

1. **Design the document pipeline first, independently of the LLM.** Build and
   verify the vector store with `verify_retriever.py` before touching the LLM.
   This isolates two separate failure domains.

2. **Separate configuration from logic.** All environment-specific values belong
   in `.env`. Never hardcode base URLs, model names, or paths.

3. **Never let the LLM make deterministic business decisions.** Rules = Python.
   LLM = natural language interface to those rules.

4. **Cache expensive objects in Streamlit.** Embedding models and vector stores
   are loaded once; use `@st.cache_resource`.

5. **Pin your LangChain version.** The API changed significantly between 0.1,
   0.2, 0.3, and 1.0. Pinning ensures the code you write today still runs next
   month.

6. **chunk_size and chunk_overlap matter.** Test your retriever with sample
   queries before connecting the LLM. Bad chunks produce bad answers regardless
   of how good the LLM is.

7. **Use `return_source_documents=True`** in `RetrievalQA`. Displaying which
   chunks were used builds user trust and makes debugging much easier.

8. **`chain_type="stuff"` is fine for small documents.** Only reach for
   `map_reduce` or `refine` when you have more context than fits in one prompt.

9. **Vendor gateways require explicit `base_url`.** Always validate that
   `ANTHROPIC_BASE_URL` is set and guard with a hard `ValueError` rather than
   silently falling back to the public API.

10. **Smoke-test scripts are more valuable than unit tests for LLM projects.**
    End-to-end scripts (like `verify_rag.py` and `verify_eligibility.py`) prove
    the entire pipeline works with real LLM calls, which unit mocks cannot do.

---

*Generated after completing Phases 1 → 2 → 2B → 3 of the Intelligent Loan Eligibility Assistant case study.*  
*Last updated: 2026-07-12*
