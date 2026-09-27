"""Similarity search over the Chroma collection (architecture §5.4, §7).

Read-only: this module never writes to the collection and never calls an LLM. The
query is embedded with the same MiniLM helper and the same model used at ingest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from src.config import TOP_K
from src.embed import encode_query
from src.index import get_collection
from src.sources import SCHEMES

# Aliases are derived from src/sources.py so the filter cannot drift from the corpus.
_SLUG_STOPWORDS = {"hdfc", "fund", "direct", "plan", "growth", "mutual", "funds"}
_ALIAS_MIN_LEN = 4


class IndexNotBuiltError(Exception):
    """The Chroma collection is missing; the operator must run the build."""


@dataclass(frozen=True)
class Hit:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    score: float
    rank: int

    @property
    def source_url(self) -> str:
        return str(self.metadata.get("source_url", ""))

    @property
    def scheme_name(self) -> str:
        return str(self.metadata.get("scheme_name", ""))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    @property
    def fetched_at(self) -> str:
        return str(self.metadata.get("fetched_at", ""))


def _aliases() -> dict[str, set[str]]:
    """Names a user might type for each allowlisted scheme."""
    table: dict[str, set[str]] = {}
    for scheme in SCHEMES:
        name = scheme["scheme_name"]
        found = {name.lower(), scheme["scheme_class"].lower()}
        without_hdfc = re.sub(r"^hdfc\s+", "", name.lower())
        found.add(without_hdfc)
        found.add(re.sub(r"\s+fund$", "", without_hdfc))
        slug = re.sub(r"[^a-z0-9]+", " ", scheme["url"].rsplit("/", 1)[-1].lower())
        words = [word for word in slug.split() if word not in _SLUG_STOPWORDS]
        if len(words) >= 2:
            found.add(" ".join(words))
            found.add("hdfc " + " ".join(words))
        table[name] = {alias for alias in found if len(alias) >= _ALIAS_MIN_LEN}
    return table


_ALIASES = _aliases()


def detect_scheme(query: str) -> str | None:
    """Return the single scheme named in the query, else None.

    None means "do not filter": either nothing matched, or the query names more
    than one scheme and filtering would silently drop the other one.
    """
    haystack = re.sub(r"[^a-z0-9]+", " ", query.lower())
    padded = f" {haystack} "
    matches = {
        name
        for name, aliases in _ALIASES.items()
        if any(f" {alias} " in padded for alias in aliases)
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def search(
    query: str,
    *,
    top_k: int | None = None,
    scheme_name: str | None = None,
    auto_filter: bool = True,
) -> list[Hit]:
    """Top-k chunks for a question. Cosine distance from the cosine-space index."""
    collection = get_collection()
    if collection is None:
        raise IndexNotBuiltError(
            "No Chroma collection found. Run `python scripts/build_index.py` first."
        )
    k = TOP_K if top_k is None else top_k
    where: dict[str, Any] | None = None
    if scheme_name is None and auto_filter:
        scheme_name = detect_scheme(query)
    if scheme_name:
        where = {"scheme_name": scheme_name}

    vector = encode_query(query)
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    hits: list[Hit] = []
    for rank, (identifier, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        hits.append(
            Hit(
                id=str(identifier),
                document=str(document),
                metadata=dict(metadata or {}),
                distance=float(distance),
                score=round(1.0 - float(distance), 6),
                rank=rank,
            )
        )
    return hits


def hits_by_scheme(hits: Sequence[Hit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.scheme_name] = counts.get(hit.scheme_name, 0) + 1
    return counts
