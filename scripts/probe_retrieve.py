"""Gold retrieval probe: no LLM, no UI. Prints the top source_url per question.

Exits non-zero if any gold question fails to return a hit from the intended
scheme page, so a chunking or embed-prefix regression is caught before any
generation work starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieve import IndexNotBuiltError, detect_scheme, hits_by_scheme, search  # noqa: E402
from src.sources import SCHEMES  # noqa: E402

URL_BY_SCHEME = {scheme["scheme_name"]: scheme["url"] for scheme in SCHEMES}
DOC_CHARS = 320

GOLD_QUESTIONS: list[tuple[str, str]] = [
    ("Expense ratio of HDFC Large Cap Fund", "HDFC Large Cap Fund"),
    ("Minimum SIP for HDFC Flexi Cap Fund", "HDFC Flexi Cap Fund"),
    ("ELSS lock-in for HDFC ELSS Tax Saver", "HDFC ELSS Tax Saver Fund"),
    ("Riskometer of HDFC Small Cap Fund", "HDFC Small Cap Fund"),
    ("Benchmark of HDFC Balanced Advantage Fund", "HDFC Balanced Advantage Fund"),
]


def main() -> int:
    print("gold retrieval probe (no LLM)\n")
    failures = 0
    for question, expected_scheme in GOLD_QUESTIONS:
        expected_url = URL_BY_SCHEME[expected_scheme]
        detected = detect_scheme(question)
        try:
            hits = search(question)
        except IndexNotBuiltError as exc:
            print(exc, file=sys.stderr)
            return 1
        top = hits[0] if hits else None
        top_url = top.source_url if top else "(no hits)"
        in_top_k = [hit for hit in hits if hit.source_url == expected_url]
        ok = top_url == expected_url
        failures += 0 if ok else 1
        print("=" * 78)
        print(f"Q: {question}")
        print(
            f"   expected scheme : {expected_scheme} "
            f"(scheme filter detected: {detected or 'none -> no filter'})"
        )
        print(f"   hits            : {len(hits)}  {hits_by_scheme(hits)}")
        print(f"   top source_url  : {top_url}  {'PASS' if ok else 'FAIL'}")
        print(
            f"   expected url in top-{len(hits)}: "
            f"{'yes' if in_top_k else 'no'} ({len(in_top_k)} hit(s))"
        )
        if top:
            print(
                f"   top chunk       : rank={top.rank} score={top.score:.4f} "
                f"distance={top.distance:.4f} section={top.section} "
                f"fetched_at={top.fetched_at}"
            )
            body = top.document[:DOC_CHARS].replace("\n", "\n       ")
            print(f"       {body}{' ...' if len(top.document) > DOC_CHARS else ''}")
    print("=" * 78)
    if failures:
        print(f"FAIL: {failures}/{len(GOLD_QUESTIONS)} gold questions returned a wrong scheme")
        return 1
    print(f"OK: {len(GOLD_QUESTIONS)}/{len(GOLD_QUESTIONS)} gold questions hit the right scheme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
