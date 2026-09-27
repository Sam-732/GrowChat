"""MiniLM embeddings for the ingest and query planes.

Architecture §5.3: one model — `all-MiniLM-L6-v2` — for both planes, with the
optional `"{scheme_name} | {section}"` prefix so the same fact type stays
distinguishable across funds.

Two runtimes reach that one model, because the deploy target has 512MB of RAM:

* Query time (`encode_query`) uses chromadb's bundled `ONNXMiniLM_L6_V2`, which
  runs the same weights under onnxruntime. onnxruntime and tokenizers are already
  chromadb dependencies, so the serving image needs no torch.
* Ingest time (`encode`, `encode_chunks`) still uses sentence-transformers. That
  path only runs in `scripts/build_index.py` and `scripts/preview_chunks.py` on an
  operator's machine, never in the deployed app, so sentence-transformers is an
  unpinned build-time extra rather than a runtime requirement.

The two runtimes are vector-compatible: over the evaluation questions, ONNX and
sentence-transformers query vectors agree to cosine 1.000000 and return byte-for-byte
identical top-k ids from the committed index (distances differ only by float32
rounding, ~1e-6). `scripts/probe_retrieve.py` is the check to re-run after touching
this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from src.chunk import Chunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 32

_model = None
_onnx = None

_NO_SENTENCE_TRANSFORMERS = (
    "sentence-transformers is not installed. It is a build-time-only extra, kept "
    "out of requirements.txt so the deployed app never loads torch. To rebuild the "
    "index, install it first:\n"
    "    pip install sentence-transformers\n"
    "Query-time embedding does not need it; see src/embed.py."
)


def get_model():
    """Load the sentence-transformers model once per process (ingest only)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - operator-facing message
            raise ImportError(_NO_SENTENCE_TRANSFORMERS) from exc

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_onnx_model():
    """Load chromadb's ONNX MiniLM once per process (query only, no torch).

    First call downloads ~80MB of weights to `~/.cache/chroma/onnx_models/`. Warm
    that cache during the deploy build so the first user request does not pay for it.
    """
    global _onnx
    if _onnx is None:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )

        _onnx = ONNXMiniLM_L6_V2()
    return _onnx


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
    """Encode texts into MiniLM vectors with sentence-transformers (ingest)."""
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
    """Query-time embedding. No prefix: a question names no scheme or section.

    Runs under onnxruntime so the serving process never imports torch. The ONNX
    embedding function L2-normalizes its output, matching the normalized vectors
    stored at ingest, so cosine distance stays comparable.
    """
    vector = get_onnx_model()([question])[0]
    return [float(value) for value in vector]
