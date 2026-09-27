"""Guardrail probe: inbound classes, canned refusals, outbound checks. No LLM, no UI.

Exits non-zero if a message is misclassified, a refusal is not safe to display
(fails its own outbound checks), a secret is echoed, or a good answer is rejected.
Run `python scripts/probe_guardrails.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.guardrails import (  # noqa: E402
    DEFAULT_URL,
    FRESHNESS_PREFIX,
    classify,
    inbound_message,
    template_fallback,
    validate,
)
from src.sources import SCHEMES  # noqa: E402

URL_BY_SCHEME = {scheme["scheme_name"]: scheme["url"] for scheme in SCHEMES}
LARGE_CAP = URL_BY_SCHEME["HDFC Large Cap Fund"]
DATE = "2026-09-27"

# (message, expected class) - spec §6.2 rows plus regression cases.
INBOUND_CASES: list[tuple[str, str]] = [
    # advice
    ("Should I buy HDFC Small Cap Fund?", "advice"),
    ("Is HDFC Large Cap Fund a good investment?", "advice"),
    ("Which fund should I choose for a 5 year goal?", "advice"),
    ("Recommend a portfolio for me", "advice"),
    ("Is now a good time to invest in HDFC Flexi Cap Fund?", "advice"),
    ("How much should I invest in HDFC ELSS Tax Saver Fund?", "advice"),
    # returns
    ("5-year returns of HDFC Large Cap Fund", "returns"),
    ("What is the CAGR of HDFC Flexi Cap Fund?", "returns"),
    ("Will HDFC Small Cap Fund outperform its benchmark?", "returns"),
    ("Predict the NAV of HDFC Large Cap Fund next year", "returns"),
    ("How has HDFC Balanced Advantage Fund performed since inception?", "returns"),
    # out of corpus
    ("SBI Bluechip expense ratio", "out_of_corpus"),
    ("ICICI Nifty 50 Index Fund expense ratio", "out_of_corpus"),
    ("Exit load of HDFC Mid Cap Fund", "out_of_corpus"),
    ("Should I buy Kotak Flexi Cap Fund?", "out_of_corpus"),
    ("What is the PPF interest rate?", "out_of_corpus"),
    ("Gold rate today", "out_of_corpus"),
    # pii
    ("My PAN is ABCDE1234F, can you check the exit load?", "pii"),
    ("Aadhaar 4321 8765 1234", "pii"),
    ("Email me at ramesh.kumar@example.com", "pii"),
    ("My folio number is 9876543210", "pii"),
    ("Send the OTP to +91 98765 43210", "pii"),
    ("My UPI id is ramesh@okhdfc", "pii"),
    # factual, still in scope
    ("Expense ratio of HDFC Large Cap Fund", "factual"),
    ("Minimum SIP for HDFC Flexi Cap Fund", "factual"),
    ("What is the lock-in period for ELSS?", "factual"),
    ("Riskometer of HDFC Small Cap Fund", "factual"),
    ("Benchmark of HDFC Balanced Advantage Fund", "factual"),
    ("Exit load of HDFC Large Cap Fund", "factual"),
    ("Should I download the capital gains statement from the page?", "factual"),
    ("How do I download the capital gains statement?", "factual"),
    ("Which scheme has the lowest expense ratio?", "factual"),
    ("", "factual"),
]

# (label, answer text, expect ok)
GOOD_ANSWER = (
    "The exit load is 1% if units are redeemed within 1 year, and nil after that. "
    "The expense ratio is 1.03% and the minimum SIP is ₹100. "
    f"Both figures are published on the scheme page.\n{LARGE_CAP}\n"
    f"{FRESHNESS_PREFIX} {DATE}"
)

OUTBOUND_CASES: list[tuple[str, str, bool]] = [
    ("good 3-sentence answer with url + stamp", GOOD_ANSWER, True),
    (
        "6 sentences, no url, no stamp",
        "One. Two. Three. Four. Five. Six.",
        False,
    ),
    (
        "no allowlisted url (invented source)",
        "The expense ratio is 1.03%.\n"
        "https://example.com/funds/hdfc\n"
        f"{FRESHNESS_PREFIX} {DATE}",
        False,
    ),
    (
        "missing freshness line",
        "The expense ratio is 1.03%.\n" + LARGE_CAP,
        False,
    ),
    (
        "malformed freshness line",
        "The expense ratio is 1.03%.\n"
        f"{LARGE_CAP}\n"
        "Last updated from sources: 27 September 2026",
        False,
    ),
    (
        "freshness line not last",
        f"{FRESHNESS_PREFIX} {DATE}\nThe expense ratio is 1.03%.\n" f"{LARGE_CAP}",
        False,
    ),
    (
        "advice phrasing",
        "You should invest in this fund for the long term.\n"
        f"{LARGE_CAP}\n"
        f"{FRESHNESS_PREFIX} {DATE}",
        False,
    ),
    (
        "invented return pitching",
        "This fund is expected to return 18% a year.\n"
        f"{LARGE_CAP}\n"
        f"{FRESHNESS_PREFIX} {DATE}",
        False,
    ),
    (
        "percent with no return word is allowed",
        "The expense ratio is 1.03%.\n"
        f"{LARGE_CAP}\n"
        f"{FRESHNESS_PREFIX} {DATE}",
        True,
    ),
    (
        "echoed pan",
        "Your PAN ABCDE1234F is on file.\n"
        f"{LARGE_CAP}\n"
        f"{FRESHNESS_PREFIX} {DATE}",
        False,
    ),
    (
        "markdown link to allowlisted url",
        f"The expense ratio is 1.03% ([source]({LARGE_CAP})).\n"
        f"{FRESHNESS_PREFIX} {DATE}",
        True,
    ),
]

# Reasons that must each produce a displayable, self-validating fallback.
FALLBACK_REASONS = (
    "body_sentences_exceeds_3",
    "missing_allowlisted_url",
    "missing_freshness_line",
    "malformed_freshness_line",
    "freshness_line_not_last",
    "advice_phrasing",
    "invented_return_pitch",
    "pii_echo",
    "something_new_from_phase6",
)


def main() -> int:
    failures = 0
    print("inbound classification (§6.2)\n")
    for message, expected in INBOUND_CASES:
        verdict = classify(message)
        ok = verdict == expected
        failures += 0 if ok else 1
        shown = message if len(message) <= 54 else message[:51] + "..."
        print(f"  {'ok  ' if ok else 'FAIL'} {verdict:<13} {shown}")
        if not ok:
            print(f"         expected {expected}, matched {verdict.matched}")
        if verdict.needs_retrieval:
            if verdict.response is not None:
                failures += 1
                print("         FAIL factual message must not carry a refusal")
        else:
            text = verdict.response or ""
            problems = _refusal_problems(text, message)
            failures += len(problems)
            for problem in problems:
                print(f"         FAIL refusal {problem}")
    print(f"\n  {len(INBOUND_CASES)} inbound cases checked")

    print("\noutbound validation (§6.4)\n")
    for label, answer, expect_ok in OUTBOUND_CASES:
        result = validate(answer)
        ok = result.ok == expect_ok
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
        print(
            f"       ok={result.ok} sentences={result.body_sentences} "
            f"urls={len(result.urls)} stamp={result.freshness_date} "
            f"reasons={list(result.reasons)}"
        )
        if not ok:
            for detail in result.details:
                print(f"       {detail}")
    print(f"\n  {len(OUTBOUND_CASES)} outbound cases checked")

    print("\nfallback templates\n")
    for reason in FALLBACK_REASONS:
        text = template_fallback(reason, LARGE_CAP, DATE)
        result = validate(text)
        ok = result.ok
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {reason:<28} reasons={list(result.reasons)}")
        if not ok:
            print(f"       {text!r}")
    print(f"\n  {len(FALLBACK_REASONS)} fallbacks checked")

    print("\nrefusal text samples\n")
    for kind in ("advice", "returns", "out_of_corpus", "pii"):
        print(f"  --- {kind}\n{_indent(inbound_message(kind, date=DATE))}")

    print()
    if failures:
        print(f"FAIL: {failures} guardrail check(s) failed")
        return 1
    print("OK: inbound classes, refusals, outbound checks and fallbacks all behave")
    return 0


def _refusal_problems(text: str, message: str) -> list[str]:
    """A canned refusal must be displayable: outbound-clean and free of secrets."""
    problems: list[str] = []
    if not text:
        return ["is empty"]
    result = validate(text)
    if not result.ok:
        problems.append(f"fails its own outbound checks: {list(result.reasons)}")
    if message.strip() and message.strip() in text:
        problems.append("echoes the user message")
    return problems


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
