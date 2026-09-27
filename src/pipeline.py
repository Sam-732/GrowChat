"""The query path: guardrails -> retrieve -> generate -> validate (architecture §7).

`answer` is the only entry point the UI needs. Order is the security property: the
inbound guardrail runs before retrieval, so an advice, returns, out-of-corpus, or PII
message never reaches the vector store or the model. Nothing here logs or stores the
user's text (§8).
"""

from __future__ import annotations

from typing import Sequence

from src.generate import (
    assemble_messages,
    complete,
    latest_fetched_at,
    repair_messages,
)
from src.guardrails import (
    DEFAULT_URL,
    classify,
    insufficient_context,
    latest_source_date,
    template_fallback,
    validate,
)
from src.retrieve import Hit, search
from src.sources import is_allowlisted

MAX_REPAIRS = 1


def answer(question: str) -> str:
    """Answer one question end to end, or return a canned refusal/template.

    Raises `IndexNotBuiltError` when the collection is missing and
    `LLMNotConfiguredError` when a generated answer is needed but no provider is
    configured; both are operator errors, not user-facing answers.
    """
    verdict = classify(question)
    if not verdict.needs_retrieval:
        # No retrieval for a refusal: nothing to ground, nothing to leak.
        return verdict.response or insufficient_context()

    hits = search(question)
    if not hits:
        return insufficient_context()

    text = complete(assemble_messages(question, hits))
    result = validate(text)
    if result.ok:
        return text

    for _attempt in range(MAX_REPAIRS):
        repaired = complete(
            repair_messages(
                question, hits, previous=text, reasons=result.reasons, details=result.details
            )
        )
        retry = validate(repaired)
        if retry.ok:
            return repaired
        text, result = repaired, retry

    return template_fallback(
        result.reasons[0] if result.reasons else "unverified",
        _citation_url(hits),
        latest_fetched_at(hits) or latest_source_date(),
    )


def _citation_url(hits: Sequence[Hit]) -> str:
    """The best allowlisted URL to cite when an answer is discarded."""
    for hit in hits:
        if hit.source_url and is_allowlisted(hit.source_url):
            return hit.source_url
    return DEFAULT_URL


__all__ = ["MAX_REPAIRS", "answer"]
