"""Phase 3 smoke-test: Loan Eligibility Checker end-to-end verification.

Run from the project root:

    $env:PYTHONIOENCODING="utf-8"
    .venv\\Scripts\\python.exe scripts/verify_eligibility.py

Tests three representative cases from the case study:
  1. Fully eligible applicant (case study example: John)
  2. Not eligible — low credit score  (case study Feature 3 example)
  3. Not eligible — loan amount too high
"""

from __future__ import annotations

import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.rag_chain import build_llm
from src.eligibility_checker import Applicant, build_eligibility_chain, check_eligibility

# ---------------------------------------------------------------------------
# Representative test cases from the case study
# ---------------------------------------------------------------------------

TEST_CASES = [
    # Case study Feature 2 example — should be ELIGIBLE
    Applicant(name="John", age=30, salary=60_000, loan_amount=500_000, credit_score=750),
    # Case study Feature 3 example — low credit score — NOT ELIGIBLE
    Applicant(name="Priya", age=28, salary=40_000, loan_amount=200_000, credit_score=650),
    # Loan amount exceeds 10x salary — NOT ELIGIBLE
    Applicant(name="Ravi", age=35, salary=30_000, loan_amount=400_000, credit_score=720),
]


def main() -> None:
    settings = get_settings()

    if not settings.anthropic_api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Edit .env and set ANTHROPIC_API_KEY, then re-run."
        )
        sys.exit(1)

    print("=" * 60)
    print("Phase 3 — Loan Eligibility Checker verification")
    print("=" * 60)
    print(f"  Model   : {settings.anthropic_model}")
    print(f"  Base URL: {settings.anthropic_base_url}")
    print()

    llm = build_llm(settings)
    chain = build_eligibility_chain(llm)

    for i, applicant in enumerate(TEST_CASES, 1):
        print(f"{'=' * 60}")
        print(f"Test Case {i}: {applicant.name}")
        print(f"  Age: {applicant.age}  |  Salary: ₹{applicant.salary:,.0f}  |  "
              f"Loan: ₹{applicant.loan_amount:,.0f}  |  Credit Score: {applicant.credit_score}")
        print("-" * 60)

        result = check_eligibility(applicant, chain)

        print(f"Rule Results:")
        for r in result["rule_results"]:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.rule}: {r.detail}")

        print()
        print("LLM Decision:")
        print(result["decision"])
        print()

    print("=" * 60)
    print("Verification complete.")


if __name__ == "__main__":
    main()
