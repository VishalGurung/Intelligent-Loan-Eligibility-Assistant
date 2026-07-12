"""Streamlit entrypoint — Intelligent Loan Eligibility Assistant.

Phase 3: exposes Feature 1 (Loan Policy Q&A via RAG) and
         Feature 2 (Loan Eligibility Checker via PromptTemplate + LLMChain).
"""

from __future__ import annotations

import streamlit as st

from src.config import get_settings
from src.vector_store import load_retriever
from src.rag_chain import build_llm, build_rag_chain
from src.eligibility_checker import (
    Applicant,
    build_eligibility_chain,
    check_eligibility,
)

st.set_page_config(
    page_title="Intelligent Loan Eligibility Assistant",
    page_icon="🏦",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Cached resource initialisation (loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading policy retriever …")
def _get_retriever():
    settings = get_settings()
    return load_retriever(settings, k=3)


@st.cache_resource(show_spinner="Connecting to LLM …")
def _get_llm():
    return build_llm(get_settings())


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()

    st.title("🏦 Intelligent Loan Eligibility Assistant")
    st.caption("Powered by Claude via LangChain · Policy grounded in bank document")

    if not settings.anthropic_api_key:
        st.error(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your `.env` file and restart the app."
        )
        st.stop()

    tab1, tab2 = st.tabs(["📋 Loan Policy Q&A", "✅ Eligibility Checker"])

    # -----------------------------------------------------------------------
    # Feature 1 — Loan Policy Q&A (RAG)
    # -----------------------------------------------------------------------
    with tab1:
        st.subheader("Loan Policy Q&A")
        st.write(
            "Ask any question about loan products, credit score guidelines, "
            "required documents, or general eligibility rules. "
            "Answers are drawn directly from the bank's policy document."
        )

        query = st.text_input(
            "Your question",
            placeholder="e.g. What is the minimum salary required for a personal loan?",
        )

        if st.button("Ask", key="ask_btn") and query.strip():
            with st.spinner("Searching policy document …"):
                retriever = _get_retriever()
                llm = _get_llm()
                rag_chain = build_rag_chain(settings, retriever)
                response = rag_chain.invoke({"query": query})

            st.markdown("**Answer:**")
            st.write(response.get("result", "").strip())

            with st.expander("Source chunks used"):
                for doc in response.get("source_documents", []):
                    st.markdown(f"> {doc.page_content.strip()}")

    # -----------------------------------------------------------------------
    # Feature 2 — Loan Eligibility Checker
    # -----------------------------------------------------------------------
    with tab2:
        st.subheader("Loan Eligibility Checker")
        st.write(
            "Enter the applicant's details below. The system will apply the "
            "bank's eligibility rules and provide a decision with reasons."
        )

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Applicant Name", value="John")
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=30)
            credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=900, value=750
            )
        with col2:
            salary = st.number_input(
                "Monthly Salary (₹)", min_value=0, value=60000, step=1000
            )
            loan_amount = st.number_input(
                "Requested Loan Amount (₹)", min_value=0, value=500000, step=10000
            )

        if st.button("Check Eligibility", key="check_btn"):
            applicant = Applicant(
                name=name,
                age=int(age),
                salary=float(salary),
                loan_amount=float(loan_amount),
                credit_score=int(credit_score),
            )

            # Show rule results immediately (no LLM needed for the rules)
            from src.eligibility_checker import evaluate_rules
            rule_results = evaluate_rules(applicant)
            eligible = all(r.passed for r in rule_results)

            if eligible:
                st.success("**Eligible**")
            else:
                st.error("**Not Eligible**")

            st.markdown("**Rule Evaluation:**")
            for r in rule_results:
                icon = "✅" if r.passed else "❌"
                st.write(f"{icon} **{r.rule}:** {r.detail}")

            # LLM-generated narrative decision
            with st.spinner("Generating detailed decision …"):
                llm = _get_llm()
                chain = build_eligibility_chain(llm)
                result = check_eligibility(applicant, chain)

            st.markdown("---")
            st.markdown("**Detailed Decision:**")
            st.write(result["decision"])


if __name__ == "__main__":
    main()
