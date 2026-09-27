"""Chroma persistence for the index plane (architecture §5.4).

Record shape: id, embedding, document, and the §4.2 metadata needed for
citations (`source_url`, `scheme_name`, `scheme_class`, `fetched_at`, `section`).
The collection is deleted and recreated on every full build, so a stale chunk can
never survive a rebuild.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.config import CHROMA_PATH, ROOT

COLLECTION_NAME = "hdfc_groww_schemes"
UPSERT_BATCH = 500

_client = None


def chroma_path() -> Path:
    """CHROMA_PATH from config; a relative path resolves against the repo root."""
    path = Path(CHROMA_PATH).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def get_client():
    """Persistent Chroma client (cosine space, telemetry off)."""
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings

        path = chroma_path()
        path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(create: bool = False):
    """Open the collection; optionally create an empty one. None if absent."""
    client = get_client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
    try:
        return client.get_collection(
            name=COLLECTION_NAME, embedding_function=None
        )
    except Exception:  # collection not built yet
        return None


def drop_collection() -> bool:
    """Delete the collection if it exists. Returns True when something was dropped."""
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        return True
    except Exception:
        return False


def _records(chunks: Sequence, vectors: Sequence[Sequence[float]]) -> dict[str, list]:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if not chunks:
        raise ValueError("nothing to index: zero chunks")
    return {
        "ids": [chunk.id for chunk in chunks],
        "embeddings": [list(vector) for vector in vectors],
        # `document` is the text the LLM will see: no embed prefix.
        "documents": [chunk.text for chunk in chunks],
        "metadatas": [chunk.metadata for chunk in chunks],
    }


def build_collection(chunks: Sequence, vectors: Sequence[Sequence[float]]) -> int:
    """Rebuild the collection from scratch. Returns the stored record count."""
    records = _records(chunks, vectors)
    client = get_client()
    drop_collection()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(chunks), UPSERT_BATCH):
        stop = start + UPSERT_BATCH
        collection.add(
            ids=records["ids"][start:stop],
            embeddings=records["embeddings"][start:stop],
            documents=records["documents"][start:stop],
            metadatas=records["metadatas"][start:stop],
        )
    return collection.count()


def collection_count() -> int:
    collection = get_collection()
    return collection.count() if collection is not None else 0
