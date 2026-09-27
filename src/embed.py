"""MiniLM embeddings, loaded once and reused at ingest and query time.

Architecture §5.3: one model (`sentence-transformers/all-MiniLM-L6-v2`) for both
planes, with the optional `"{scheme_name} | {section}"` prefix so the same fact
type stays distinguishable across funds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from src.chunk import Chunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 32

_model = None


def get_model():
    """Load the sentence-transformers model once per process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_prefix(scheme_name: str, section: str) -> str:
    return f"{scheme_name} | {section}\n"


def chunk_embed_text(chunk: "Chunk") -> str:
    """The exact string embedded for a chunk at ingest time (§5.3)."""
    return f"{embed_prefix(chunk.scheme_name, chunk.section)}{chunk.text}"


def encode(
    texts: Sequence[str],
    *,
    prefixes: Sequence[str] | None = None,
    normalize: bool = True,
) -> list[list[float]]:
    """Encode texts into MiniLM vectors. Same function serves ingest and query."""
    if prefixes is not None and len(prefixes) != len(texts):
        raise ValueError("prefixes must be the same length as texts")
    payload = [
        f"{prefix}{text}" for prefix, text in zip(prefixes or [""] * len(texts), texts)
    ]
    vectors = get_model().encode(
        payload,
        batch_size=BATCH_SIZE,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return [[float(value) for value in vector] for vector in vectors]


def encode_chunks(chunks: Sequence["Chunk"]) -> list[list[float]]:
    return encode([chunk_embed_text(chunk) for chunk in chunks])


def encode_query(question: str) -> list[float]:
    """Query-time embedding. No prefix: a question names no scheme or section."""
    return encode([question])[0]
