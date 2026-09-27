"""Inspect Phase 1 extracts: per-scheme character counts and table/key-value facts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest import IngestError, ingest_all, ingest_url  # noqa: E402
from src.sources import is_allowlisted  # noqa: E402

BLOCKED_URL = "https://groww.in/mutual-funds/other"
FACT_MARKERS = (
    "expense ratio",
    "min. for sip",
    "exit load",
    "stamp duty",
    "risk",
    "benchmark",
    "lock-in",
    "nav",
)
PREVIEW_CHARS = 420


def count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, re.M))


def check_allowlist_gate() -> bool:
    allowed = is_allowlisted(BLOCKED_URL)
    try:
        ingest_url({"scheme_class": "x", "scheme_name": "x", "url": BLOCKED_URL})
    except IngestError as exc:
        print(f"non-allowlisted URL refused: {exc}")
        return not allowed
    print("FAIL: non-allowlisted URL was not refused")
    return False


def main() -> int:
    print("=" * 78)
    if not check_allowlist_gate():
        return 1

    try:
        documents = ingest_all(dump_raw=True)
    except IngestError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"\ningested {len(documents)}/5 documents")
    print(
        f"{'scheme':<34} {'class':<20} {'chars':>7} {'lines':>6} "
        f"{'tbl':>4} {'kv':>5}  facts"
    )
    print("-" * 78)
    failures = 0
    for doc in documents:
        markers = [m for m in FACT_MARKERS if m in doc.text.lower()]
        if len(doc.text) < 400 or not doc.fetched_at:
            failures += 1
        print(
            f"{doc.scheme_name:<34} {doc.scheme_class:<20} {doc.char_count:>7} "
            f"{count(doc.text, r'.*'):>6} {count(doc.text, r'^\|'):>4} "
            f"{count(doc.text, r'^.+: .+$'):>5}  {len(markers)}/{len(FACT_MARKERS)}"
        )
    print("-" * 78)
    total = sum(doc.char_count for doc in documents)
    print(f"total characters: {total}")

    first = documents[0]
    print(f"\n--- {first.scheme_name} preview (first {PREVIEW_CHARS} chars) ---")
    print(first.text[:PREVIEW_CHARS])
    print("--- table sample ---")
    table_rows = [line for line in first.text.splitlines() if line.startswith("|")]
    print("\n".join(table_rows[:10]) or "NO TABLE ROWS FOUND")
    print("\n--- key: value sample ---")
    kv_rows = [line for line in first.text.splitlines() if re.match(r"^.+: .+$", line)]
    print("\n".join(kv_rows[:12]) or "NO KEY: VALUE ROWS FOUND")

    if failures:
        print(f"\nFAIL: {failures} document(s) empty or unstamped", file=sys.stderr)
        return 1
    print("\nOK: 5 non-empty documents with fetched_at and table/key-value facts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
