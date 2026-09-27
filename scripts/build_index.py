"""Build the Chroma index: ingest -> chunk -> embed -> upsert (Phase 3).

Always fetches live so `fetched_at` is the real freshness stamp. The collection
is deleted and recreated, so the printed count is the whole index.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunk import chunk_documents  # noqa: E402
from src.embed import EMBED_DIM, MODEL_NAME, encode_chunks  # noqa: E402
from src.index import (  # noqa: E402
    COLLECTION_NAME,
    build_collection,
    chroma_path,
    collection_count,
)
from src.ingest import IngestError, ingest_all  # noqa: E402


def main() -> int:
    try:
        documents = ingest_all(dump_raw=True)
    except IngestError as exc:
        print(exc, file=sys.stderr)
        return 1

    chunks = chunk_documents(documents)
    print(f"documents={len(documents)} chunks={len(chunks)}")
    for document in documents:
        count = sum(1 for chunk in chunks if chunk.scheme_name == document.scheme_name)
        print(f"  {document.scheme_name:<34} chunks={count:>3}  fetched_at={document.fetched_at}")

    print(f"embedding {MODEL_NAME} (dim={EMBED_DIM}) ...")
    vectors = encode_chunks(chunks)
    stored = build_collection(chunks, vectors)

    print(f"collection={COLLECTION_NAME}")
    print(f"persist_dir={chroma_path()}")
    print(f"collection count={stored}")
    print(f"reopened count={collection_count()}")

    sample = chunks[0]
    print(
        f"sample record: id={sample.id} section={sample.section} "
        f"chars={len(sample.text)} url={sample.source_url}"
    )
    for chunk in chunks:
        if chunk.has_table:
            print(f"sample table record: id={chunk.id} section={chunk.section}")
            break

    if stored != len(chunks) or collection_count() != len(chunks):
        print(
            f"FAIL: expected {len(chunks)} records, stored={stored}, "
            f"reopened={collection_count()}",
            file=sys.stderr,
        )
        return 1
    if any(len(vector) != EMBED_DIM for vector in vectors):
        print(f"FAIL: unexpected embedding dimension", file=sys.stderr)
        return 1
    print(f"OK: {stored} records == {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
