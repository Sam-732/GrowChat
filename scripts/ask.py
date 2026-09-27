"""Ask one question through the real pipeline. Manual check for Phase 6.

    python scripts/ask.py "expense ratio of HDFC Large Cap Fund"
    python scripts/ask.py "expense ratio of HDFC Large Cap Fund" --dry-run

--dry-run stops before generation and prints the guardrail class, the retrieved
chunks, and the exact prompt, so prompt assembly can be checked without a key.

Exit codes: 0 answered, 1 refused or templated, 2 no LLM configured,
3 index not built, 4 bad usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate import (  # noqa: E402
    LLMError,
    LLMNotConfiguredError,
    assemble_prompt,
    latest_fetched_at,
)
from src.guardrails import classify  # noqa: E402
from src.pipeline import answer  # noqa: E402
from src.retrieve import IndexNotBuiltError, search  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_NO_LLM = 2
EXIT_NO_INDEX = 3
EXIT_USAGE = 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the facts-only assistant one question.",
        epilog=(
            'example: python scripts/ask.py "minimum SIP for HDFC Small Cap Fund"'
        ),
    )
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print guardrail class, retrieved chunks, and prompt; no LLM call",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="also print the prompt that was sent",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    if not question:
        print("give a question", file=sys.stderr)
        return EXIT_USAGE

    verdict = classify(question)
    print(f"question : {question}")
    print(f"guardrail: {verdict.kind} (matched: {', '.join(verdict.matched) or 'none'})")

    if not verdict.needs_retrieval:
        response = verdict.response or ""
        print("\n" + response)
        return EXIT_REFUSED

    try:
        hits = search(question)
    except IndexNotBuiltError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return EXIT_NO_INDEX

    if args.dry_run:
        print(f"\nretrieved {len(hits)} chunk(s):")
        for hit in hits:
            print(
                f"  [{hit.rank}] {hit.score:.4f} {hit.section:<26} {hit.source_url}"
            )
            print(f"      {hit.document[:220].replace(chr(10), ' ')}")
        print(f"\nlatest fetched_at: {latest_fetched_at(hits) or '(none)'}")
        print("\n" + "=" * 78)
        print(assemble_prompt(question, hits))
        return EXIT_OK

    if args.show_prompt:
        print("\n" + "=" * 78)
        print(assemble_prompt(question, hits))
        print("=" * 78)

    try:
        response = answer(question)
    except LLMNotConfiguredError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print(
            "hint: copy .env.example to .env, set LLM_API_KEY and LLM_MODEL, "
            "then rerun. Try --dry-run to inspect retrieval and the prompt.",
            file=sys.stderr,
        )
        return EXIT_NO_LLM
    except LLMError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return EXIT_NO_LLM

    print("\n" + response)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
