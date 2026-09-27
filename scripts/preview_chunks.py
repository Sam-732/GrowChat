"""Inspect chunks and their MiniLM vectors; dump everything to a txt file.

Console: chunk counts per scheme, 3 sample chunks each, verification results.
Txt file (default data/raw/chunks_preview.txt, gitignored): every chunk's full
text, metadata, id, the exact embed prefix, and its embedding (dimension, norm,
head of the vector).

Vectors come from src.embed, the same helper that built the index, so the numbers
here are the ones actually stored in Chroma. Inspection only: this script never
writes to a vector store.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunk import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SEPARATORS,
    Chunk,
    chunk_documents,
)
from src.embed import MODEL_NAME, embed_prefix, encode_chunks  # noqa: E402
from src.ingest import RAW_DIR, Document, IngestError, ingest_all, scheme_slug  # noqa: E402
from src.sources import is_allowlisted  # noqa: E402

DEFAULT_OUT = RAW_DIR / "chunks_preview.txt"
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SAMPLE_PRINT_CHARS = 700
RAW_DIVIDER = "=" * 72
RAW_FIELDS = ("source_url", "scheme_name", "scheme_class", "fetched_at")


def load_from_raw() -> list[Document]:
    """Rebuild Documents from the gitignored data/raw dumps (no HTTP)."""
    documents: list[Document] = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        if path.name == DEFAULT_OUT.name:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        fields: dict[str, str] = {}
        index = 0
        for index, line in enumerate(lines):
            if line.startswith(RAW_DIVIDER):
                break
            key, _, value = line.partition(":")
            if key in RAW_FIELDS:
                fields[key] = value.strip()
        body = "\n".join(lines[index + 1 :]).strip()
        missing = [field for field in RAW_FIELDS if field not in fields]
        if missing or not body:
            print(f"skipping {path.name}: missing {missing or 'body'}", file=sys.stderr)
            continue
        documents.append(
            Document(
                source_url=fields["source_url"],
                scheme_name=fields["scheme_name"],
                scheme_class=fields["scheme_class"],
                fetched_at=fields["fetched_at"],
                text=body,
            )
        )
    if not documents:
        raise IngestError(f"No usable dumps in {RAW_DIR}. Run ingest first.")
    return documents


def embed_all(chunks: Sequence[Chunk]) -> list[list[float]] | None:
    """Embed with the production helper, or skip if the model cannot load."""
    try:
        return encode_chunks(chunks)
    except Exception as exc:  # pragma: no cover - model download / load failure
        print(f"could not embed with {MODEL_NAME} ({exc}); skipping vectors.", file=sys.stderr)
        return None


def verify(document: Document, chunks: Sequence[Chunk]) -> list[str]:
    """Checks from implementation.md Phase 2 Verify section."""
    problems: list[str] = []
    if not chunks:
        problems.append(f"{document.scheme_name}: no chunks")
    for chunk in chunks:
        if not is_allowlisted(chunk.source_url):
            problems.append(f"non-allowlisted source_url: {chunk.source_url}")
        if chunk.fetched_at != document.fetched_at:
            problems.append(f"chunk {chunk.chunk_index}: fetched_at mismatch")
        if chunk.char_count > CHUNK_SIZE:
            problems.append(
                f"chunk {chunk.chunk_index} ({chunk.section}) is {chunk.char_count} chars "
                f"> {CHUNK_SIZE}"
            )
        for line in chunk.text.splitlines():
            if line.strip().startswith("|") and not TABLE_ROW_RE.match(line):
                problems.append(f"chunk {chunk.chunk_index}: cut table row {line[:40]!r}")
    joined = "\n".join(chunk.text for chunk in chunks)
    source_rows = [line for line in document.text.splitlines() if TABLE_ROW_RE.match(line)]
    for row in source_rows:
        if row not in joined:
            problems.append(f"table row lost or split: {row[:60]!r}")
    return problems


def pick_samples(chunks: Sequence[Chunk], count: int) -> list[Chunk]:
    """Prefer a fee/load/SIP table chunk, then other tables, then long prose."""
    facts = ("expense", "exit load", "min. for", "sip", "stamp duty", "lock")
    table_chunks = [chunk for chunk in chunks if chunk.has_table]
    fact_chunks = [
        chunk
        for chunk in table_chunks
        if any(marker in chunk.text.lower() for marker in facts)
    ]
    samples: list[Chunk] = []
    for candidate in fact_chunks + table_chunks:
        if candidate not in samples:
            samples.append(candidate)
        if len(samples) == count:
            return samples
    for chunk in chunks:
        if chunk not in samples:
            samples.append(chunk)
        if len(samples) == count:
            break
    return samples


def vector_line(vector: Sequence[float] | None, index: int, full: bool) -> str:
    if vector is None:
        return ""
    norm = sum(value * value for value in vector) ** 0.5
    values = vector if full else vector[:8]
    shown = ", ".join(f"{value:+.4f}" for value in values)
    suffix = "" if full else f", ... ({len(vector) - len(values)} more)"
    return f"embedding[{index}] dim={len(vector)} l2={norm:.4f} [{shown}{suffix}]"


def build_report(
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]] | None,
    full_vectors: bool,
) -> str:
    lines = [
        "Growchat - chunk and embedding preview",
        f"generated_at (UTC): {datetime.now(timezone.utc).date().isoformat()}",
        f"chunk_size={CHUNK_SIZE} chunk_overlap={CHUNK_OVERLAP} "
        f"separators={SEPARATORS}",
        f"documents={len(documents)} chunks={len(chunks)}",
        f"embedding model: {MODEL_NAME if vectors else 'not computed'}",
        f"embedding input: embed_prefix + chunk text, e.g. '{embed_prefix('<scheme>', '<section>')}'",
        "the vector sees the prefix; the LLM sees only the chunk text (index.py)",
        f"full_vectors={full_vectors}",
        "=" * 96,
    ]
    by_scheme: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_scheme.setdefault(chunk.scheme_name, []).append(chunk)
    vector_by_id = (
        {chunk.id: vectors[position] for position, chunk in enumerate(chunks)}
        if vectors
        else {}
    )

    for document in documents:
        scheme_chunks = by_scheme.get(document.scheme_name, [])
        lines.append("")
        lines.append("#" * 96)
        lines.append(
            f"SCHEME {document.scheme_name} ({document.scheme_class}) "
            f"chars={len(document.text)} chunks={len(scheme_chunks)}"
        )
        lines.append(f"url: {document.source_url}")
        lines.append(f"fetched_at: {document.fetched_at}")
        sections: dict[str, int] = {}
        for chunk in scheme_chunks:
            sections[chunk.section] = sections.get(chunk.section, 0) + 1
        lines.append("sections: " + ", ".join(f"{k}={v}" for k, v in sections.items()))
        lines.append("-" * 96)
        lines.append(f"{'idx':>4}  {'chars':>5}  {'tbl':>3}  {'section':<34} id")
        for chunk in scheme_chunks:
            lines.append(
                f"{chunk.chunk_index:>4}  {chunk.char_count:>5}  "
                f"{'yes' if chunk.has_table else 'no':>3}  {chunk.section:<34} {chunk.id}"
            )
        for chunk in scheme_chunks:
            lines.append("")
            lines.append("-" * 96)
            lines.append(
                f"CHUNK {chunk.chunk_index} | id={chunk.id} | section={chunk.section} "
                f"| chars={chunk.char_count} | table={chunk.has_table} "
                f"| scheme={chunk.scheme_name} | fetched_at={chunk.fetched_at}"
            )
            lines.append(chunk.text)
            if vectors:
                lines.append(f"embed_prefix: {embed_prefix(chunk.scheme_name, chunk.section)!r}")
            line = vector_line(vector_by_id.get(chunk.id), chunk.chunk_index, full_vectors)
            if line:
                lines.append(line)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview Phase 2 chunks and vectors.")
    parser.add_argument("--from-raw", action="store_true", help="reuse data/raw dumps")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="txt dump path")
    parser.add_argument("--samples", type=int, default=3, help="samples per scheme")
    parser.add_argument("--no-embed", action="store_true", help="skip MiniLM vectors")
    parser.add_argument("--full-vectors", action="store_true", help="print all 384 dims")
    args = parser.parse_args(argv)

    try:
        documents = load_from_raw() if args.from_raw else ingest_all(dump_raw=True)
    except IngestError as exc:
        print(exc, file=sys.stderr)
        return 1
    chunks = chunk_documents(documents)

    vectors = None if args.no_embed else embed_all(chunks)

    problems: list[str] = []
    for document in documents:
        scheme_chunks = [
            chunk for chunk in chunks if chunk.scheme_name == document.scheme_name
        ]
        problems.extend(verify(document, scheme_chunks))

    print(f"documents={len(documents)} chunks={len(chunks)}")
    for document in documents:
        scheme_chunks = [
            chunk for chunk in chunks if chunk.scheme_name == document.scheme_name
        ]
        tables = sum(1 for chunk in scheme_chunks if chunk.has_table)
        sections = len({chunk.section for chunk in scheme_chunks})
        largest = max((chunk.char_count for chunk in scheme_chunks), default=0)
        print(
            f"  {document.scheme_name:<34} chunks={len(scheme_chunks):>3} "
            f"sections={sections:>2} tables={tables:>3} max_chars={largest:>4}"
        )
        for chunk in pick_samples(scheme_chunks, args.samples):
            snippet = chunk.text[:SAMPLE_PRINT_CHARS].replace("\n", "\n    ")
            truncated = " ..." if len(chunk.text) > SAMPLE_PRINT_CHARS else ""
            print(
                f"    - [{chunk.chunk_index}] {chunk.section} "
                f"table={chunk.has_table} chars={chunk.char_count} id={chunk.id}\n"
                f"      {snippet}{truncated}"
            )
    if vectors:
        print(f"embeddings: {len(vectors)} vectors, dim={len(vectors[0])} (normalized)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report(documents, chunks, vectors, args.full_vectors), encoding="utf-8"
    )
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("OK: allowlisted urls, fetched_at carried, no cut table rows, size limits held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
