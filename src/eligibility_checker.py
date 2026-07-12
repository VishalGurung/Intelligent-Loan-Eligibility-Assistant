"""Phase 3 — Loan Eligibility Checker.

Implements Feature 2 from the case study:
  "Intelligent Loan Eligibility Assistant — Feature 2: Loan Eligibility Checker"

Responsibilities
----------------
- Apply the four business rules from the case study against applicant data.
- Use PromptTemplate + LLMChain (as specified in the case study) to generate
  a human-readable eligibility decision with per-rule reasons.
- Reuse the existing LLM from src.rag_chain — no new LLM is instantiated.

Business Rules (from case study)
---------------------------------
  Rule            Criteria
  Age             >= 21
  Salary          >= 30,000
  Credit Score    >= 700
  Loan Amount     <= 10 x Monthly Salary

LangChain components used
--------------------------
  - PromptTemplate
  - LLMChain
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Applicant data model (matches the case study JSON schema)
# ---------------------------------------------------------------------------


@dataclass
class Applicant:
    """Represents an applicant's loan application data.

    Matches the case study input schema:
        {
          "name": "John",
          "age": 30,
          "salary": 60000,
          "loanAmount": 500000,
          "creditScore": 750
        }
    """

    name: str
    age: int
    salary: float          # monthly salary in ₹
    loan_amount: float     # requested loan amount in ₹
    credit_score: int


# ---------------------------------------------------------------------------
# Business rule engine
# ---------------------------------------------------------------------------


@dataclass
class RuleResult:
    """Outcome of evaluating a single eligibility rule."""

    rule: str
    passed: bool
    detail: str


def evaluate_rules(applicant: Applicant) -> list[RuleResult]:
    """Apply all four business rules and return per-rule results.

    Parameters
    ----------
    applicant:
        The loan applicant's data.

    Returns
    -------
    list[RuleResult]
        One result per rule, in the order defined by the case study.
    """
    max_loan = 10 * applicant.salary

    return [
        RuleResult(
            rule="Age",
            passed=applicant.age >= 21,
            detail=(
                f"Age {applicant.age} >= 21 — satisfied"
                if applicant.age >= 21
                else f"Age {applicant.age} < 21 — not satisfied (minimum age is 21)"
            ),
        ),
        RuleResult(
            rule="Salary",
            passed=applicant.salary >= 30_000,
            detail=(
                f"Monthly salary ₹{applicant.salary:,.0f} >= ₹30,000 — satisfied"
                if applicant.salary >= 30_000
                else f"Monthly salary ₹{applicant.salary:,.0f} < ₹30,000 — not satisfied (minimum is ₹30,000)"
            ),
        ),
        RuleResult(
            rule="Credit Score",
            passed=applicant.credit_score >= 700,
            detail=(
                f"Credit score {applicant.credit_score} >= 700 — acceptable"
                if applicant.credit_score >= 700
                else f"Credit score {applicant.credit_score} < 700 — not acceptable (minimum is 700)"
            ),
        ),
        RuleResult(
            rule="Loan Amount",
            passed=applicant.loan_amount <= max_loan,
            detail=(
                f"Requested ₹{applicant.loan_amount:,.0f} <= ₹{max_loan:,.0f} (10× salary) — satisfied"
                if applicant.loan_amount <= max_loan
                else (
                    f"Requested ₹{applicant.loan_amount:,.0f} > ₹{max_loan:,.0f} "
                    f"(10× salary) — not satisfied (maximum allowed is ₹{max_loan:,.0f})"
                )
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Prompt Template (case study: Feature 2 uses PromptTemplate + LLMChain)
# ---------------------------------------------------------------------------

_ELIGIBILITY_TEMPLATE = """\
You are a bank loan officer assistant. Based on the rule evaluation below, \
produce a clear eligibility decision for the applicant.

Applicant: {name}
Age: {age}
Monthly Salary: ₹{salary}
Requested Loan Amount: ₹{loan_amount}
Credit Score: {credit_score}

Rule Evaluation:
{rule_summary}

Overall Result: {overall_result}

Instructions:
- Start with "Eligible" or "Not Eligible" on the first line.
- Then write "Reason:" followed by a bullet list of each rule outcome.
- If not eligible, add "Next Steps:" with actionable advice grounded in bank policy.
- Be concise and professional.
"""

_ELIGIBILITY_PROMPT = PromptTemplate(
    input_variables=[
        "name",
        "age",
        "salary",
        "loan_amount",
        "credit_score",
        "rule_summary",
        "overall_result",
    ],
    template=_ELIGIBILITY_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_eligibility_chain(llm: ChatAnthropic) -> LLMChain:
    """Build the LLMChain for the Loan Eligibility Checker.

    Parameters
    ----------
    llm:
        A configured ChatAnthropic instance (reuse from
        :func:`src.rag_chain.build_llm`).

    Returns
    -------
    LLMChain
        A LangChain chain that accepts applicant data and returns a
        formatted eligibility decision.
    """
    return LLMChain(llm=llm, prompt=_ELIGIBILITY_PROMPT)


def check_eligibility(applicant: Applicant, chain: LLMChain) -> dict:
    """Run the eligibility check for an applicant.

    Evaluates business rules locally, then invokes the LLMChain to
    produce a human-readable decision with reasons and next steps.

    Parameters
    ----------
    applicant:
        The loan applicant's data.
    chain:
        The LLMChain returned by :func:`build_eligibility_chain`.

    Returns
    -------
    dict with keys:
        - ``"eligible"``     (bool)
        - ``"rule_results"`` (list[RuleResult])
        - ``"decision"``     (str — LLM-generated narrative)
    """
    rule_results = evaluate_rules(applicant)
    eligible = all(r.passed for r in rule_results)

    rule_summary = "\n".join(
        f"- [{' OK ' if r.passed else 'FAIL'}] {r.rule}: {r.detail}"
        for r in rule_results
    )
    overall_result = "ELIGIBLE" if eligible else "NOT ELIGIBLE"

    response = chain.invoke(
        {
            "name": applicant.name,
            "age": applicant.age,
            "salary": f"{applicant.salary:,.0f}",
            "loan_amount": f"{applicant.loan_amount:,.0f}",
            "credit_score": applicant.credit_score,
            "rule_summary": rule_summary,
            "overall_result": overall_result,
        }
    )

    decision = response.get("text", "").strip()

    return {
        "eligible": eligible,
        "rule_results": rule_results,
        "decision": decision,
    }
