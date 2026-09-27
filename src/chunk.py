"""Chunking for the index plane: section split, then recursive character split.

Implements architecture §5.2 (semantic sections, recursive character splitting
inside a section, tables kept whole) and §4.2 (chunk metadata).

Recursive character splitting is kept for this corpus: on the five allowlisted
pages no fee, exit-load, min-investment or holdings row is cut mid-row, so the
documented fallback to semantic splitting is not needed. `scripts/preview_chunks.py`
re-checks that on every run.

No embedding and no vector store here: this module only produces text chunks
with the metadata a later stage needs for citations.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from src.ingest import Document

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]
# `#### Exit load` style labels live *inside* a section and must not start one.
SECTION_HEADING_LEVELS = (1, 2, 3)
# Section label for the level-1 document title, so the slug stays readable.
TOP_SECTION_LABEL = "overview"

HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_ROW_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
SLUG_RE = re.compile(r"[^a-z0-9]+")
# Page-header layout: identity pills, the return ticker widget, then key:value facts.
HEADER_PERF_RE = re.compile(r"^[+-]?\d[\d.,]*\s*%")
HEADER_KV_RE = re.compile(r"^[^:\n]{2,60}:\s+\S")
# Chunks that are only a call to action carry no fact and only dilute retrieval.
CTA_CHUNK_TEXT = {
    "see all", "view all", "compare", "compare funds", "view details", "check past data",
}


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text plus the §4.2 metadata for citation."""

    id: str
    text: str
    source_url: str
    scheme_name: str
    scheme_class: str
    fetched_at: str
    section: str
    chunk_index: int

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source_url": self.source_url,
            "scheme_name": self.scheme_name,
            "scheme_class": self.scheme_class,
            "fetched_at": self.fetched_at,
            "section": self.section,
            "chunk_index": self.chunk_index,
        }

    @property
    def has_table(self) -> bool:
        return any(TABLE_ROW_RE.match(line) for line in self.text.splitlines())

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class Section:
    title: str
    level: int
    body: str


def make_chunk_id(source_url: str, section: str, chunk_index: int) -> str:
    """Stable id: hash of source_url + section + chunk_index (architecture §5.4)."""
    payload = f"{source_url}|{section}|{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def slugify(text: str) -> str:
    return SLUG_RE.sub("_", text.strip().lower()).strip("_") or "section"


def split_sections(text: str) -> list[Section]:
    """Split ingested text on its markdown headings (ingest emits h1 -> `#`)."""
    sections: list[Section] = []
    title, level, buffer = "", 0, []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) in SECTION_HEADING_LEVELS:
            if any(part.strip() for part in buffer):
                sections.append(Section(title, level, "\n".join(buffer).strip()))
            title = match.group(2).strip()
            level = len(match.group(1))
            buffer = [line]
            continue
        buffer.append(line)
    if any(part.strip() for part in buffer):
        sections.append(Section(title, level, "\n".join(buffer).strip()))
    return sections


def _is_heading_only(lines: Sequence[str]) -> bool:
    return bool(lines) and all(HEADING_RE.match(line) for line in lines)


def iter_blocks(body: str) -> list[tuple[str, list[str]]]:
    """Split a section body into ('text'|'table', lines) blocks.

    A heading starts a new text block, including the `####` in-section labels
    ("#### Exit load", "#### Investment Objective"): §5.2 treats those labelled
    parts as their own sections, so a fact is never buried behind its neighbours.
    A heading-only text block is merged into the block that follows it, so a
    heading never becomes a chunk of its own.
    """
    raw: list[tuple[str, list[str]]] = []
    buffer: list[str] = []
    kind = "text"
    for line in body.splitlines():
        is_table = bool(TABLE_ROW_RE.match(line))
        line_kind = "table" if is_table else "text"
        new_heading = bool(HEADING_RE.match(line))
        starts_block = line_kind != kind or (
            new_heading and kind == "text" and any(part.strip() for part in buffer)
        )
        if starts_block and buffer:
            raw.append((kind, buffer))
            buffer = []
        kind = line_kind
        buffer.append(line)
    if buffer:
        raw.append((kind, buffer))

    blocks: list[tuple[str, list[str]]] = []
    pending: list[str] = []
    for block_kind, lines in raw:
        lines = [line for line in lines if line.strip()]
        if not lines:
            continue
        if block_kind == "text" and _is_heading_only(lines):
            pending.extend(lines)
            continue
        blocks.append((block_kind, pending + lines))
        pending = []
    if pending:
        blocks.append(("text", pending))
    return blocks


def _clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [text[start : start + chunk_size] for start in range(0, len(text), chunk_size)]


def _atoms(text: str, separators: Sequence[str], chunk_size: int) -> list[str]:
    """Split down the separator list until every unit fits the chunk size."""
    if not text.strip():
        return []
    if not separators:
        return _hard_split(text, chunk_size)
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _atoms(text, rest, chunk_size)
    parts = text.split(sep)
    units: list[str] = []
    for index, part in enumerate(parts):
        piece = part.strip()
        if not piece:
            continue
        if len(piece) <= chunk_size:
            units.append(piece + (sep if index < len(parts) - 1 else ""))
        else:
            units.extend(_atoms(piece, rest, chunk_size))
    return units


def _overlap_window(window: Sequence[str], chunk_overlap: int) -> tuple[list[str], int]:
    kept: list[str] = []
    size = 0
    for unit in reversed(window):
        if size + len(unit) > chunk_overlap:
            break
        kept.insert(0, unit)
        size += len(unit)
    return kept, size


def _merge(units: Sequence[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    size = 0
    for unit in units:
        if window and size + len(unit) > chunk_size:
            chunks.append(_clean("".join(window)))
            window, size = _overlap_window(window, chunk_overlap)
        window.append(unit)
        size += len(unit)
    if window:
        chunks.append(_clean("".join(window)))
    return [chunk for chunk in chunks if chunk]


def split_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Sequence[str] = SEPARATORS,
) -> list[str]:
    """Recursive character split with overlap (architecture §5.2 step 2)."""
    return _merge(_atoms(text, list(separators), chunk_size), chunk_size, chunk_overlap)


def split_table(
    lines: Sequence[str],
    *,
    context: str,
    chunk_size: int = CHUNK_SIZE,
) -> list[str]:
    """Keep a markdown table in one chunk; split by row groups only if needed.

    Row groups repeat the table header, and every group after the first repeats
    the scheme + section context line, so no row is cut mid-row and no piece
    loses its caption. The budget accounts for the heading lines that stay with
    the first group, so a split table still respects `chunk_size`.
    """
    body = _clean("\n".join(lines))
    if len(body) <= chunk_size:
        return [body]

    first_row = next(
        (index for index, line in enumerate(lines) if TABLE_ROW_RE.match(line)), None
    )
    if first_row is None:
        return [body]
    leading_text = _clean("\n".join(lines[:first_row]))
    leading = lines[:first_row]
    table_lines = lines[first_row:]
    header, rest = table_lines[:1], table_lines[1:]
    if rest and SEPARATOR_ROW_RE.match(rest[0]):
        header, rest = table_lines[:2], table_lines[2:]
    else:
        # Keep a split table valid markdown: repeat header + separator, never a
        # data row, in each row group.
        header = [header[0], _separator_row(header[0])]
    rows = [line for line in rest if line.strip()]

    leading_cost = len(leading_text) + 1 if leading_text else 0
    context_cost = len(context) + 1 if context else 0
    groups: list[list[str]] = []
    current: list[str] = []
    for row in rows:
        candidate = current + [row]
        cost = leading_cost if not groups else context_cost
        if current and cost + len("\n".join(header + candidate)) > chunk_size:
            groups.append(current)
            current = [row]
        else:
            current = candidate
    if current:
        groups.append(current)

    pieces: list[str] = []
    for position, group in enumerate(groups):
        if position == 0:
            pieces.append(_clean("\n".join(leading + header + group)))
        else:
            # A single oversized row is never cut: a markdown row is one fact.
            pieces.append(_clean("\n".join([context, *header, *group])))
    return pieces


def _separator_row(header_row: str) -> str:
    columns = max(1, header_row.count("|") - 1)
    return "| " + " | ".join(["---"] * columns) + " |"


def split_header_block(lines: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Split the page-header block into identity / returns / key-facts groups.

    The header is three widgets glued together by the HTML layout: the scheme
    name with its category, risk and lock-in pills; the return ticker; and the
    `key: value` stat grid. Retrieval for "riskometer" or "lock-in" should not have
    to compete with a dozen return figures.
    """
    labels = {"identity": TOP_SECTION_LABEL, "returns": "returns_ticker", "facts": "key_facts"}
    groups: list[tuple[str, list[str]]] = []
    mode = "identity"
    current: list[str] = []
    for line in lines:
        if mode == "identity" and HEADER_PERF_RE.match(line):
            groups.append((labels[mode], current))
            current, mode = [], "returns"
        elif mode in ("identity", "returns") and HEADER_KV_RE.match(line):
            groups.append((labels[mode], current))
            current, mode = [], "facts"
        current.append(line)
    groups.append((labels[mode], current))
    return [(label, group) for label, group in groups if any(part.strip() for part in group)]


def _is_chrome_chunk(text: str) -> bool:
    """True for a chunk whose only content is a call to action."""
    body = [line for line in text.splitlines() if line.strip() and not HEADING_RE.match(line)]
    if not body:
        return True
    joined = " ".join(body).strip().strip(".:-").lower()
    return joined in CTA_CHUNK_TEXT


def _section_label(section: Section, used: set[str], scheme_name: str = "") -> str:
    base = TOP_SECTION_LABEL if section.level <= 1 else slugify(section.title)
    if scheme_name and section.level > 1:
        # "### About HDFC Large Cap Fund Direct Growth" -> "about": the label is
        # embedded in every chunk, so a label that repeats the scheme name adds
        # no signal and drowns the fact the question is actually about.
        scheme_words = set(slugify(scheme_name).split("_"))
        kept = [word for word in base.split("_") if word and word not in scheme_words]
        if kept:
            base = "_".join(kept)
    label, suffix = base, 2
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def chunk_document(
    document: Document,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Sequence[str] = SEPARATORS,
) -> list[Chunk]:
    """One ingest Document -> ordered chunks with §4.2 metadata."""
    chunks: list[Chunk] = []
    used_labels: set[str] = set()
    index = 0
    for section in split_sections(document.text):
        label = _section_label(section, used_labels, document.scheme_name)
        context = f"{document.scheme_name} | {section.title}".strip(" |")
        first_block = True
        for kind, lines in iter_blocks(section.body):
            units: list[tuple[str, str, list[str]]] = [(label, kind, lines)]
            if section.level <= 1 and first_block and kind == "text":
                # Keep the scheme/risk identity facts out of the return-ticker
                # numbers, so "riskometer" and "lock-in" questions match a short
                # chunk instead of a block of performance figures.
                units = [
                    (sub_label, "text", group)
                    for sub_label, group in split_header_block(lines)
                ]
            first_block = False
            for unit_label, unit_kind, unit_lines in units:
                if unit_kind == "table":
                    pieces = split_table(
                        unit_lines, context=context, chunk_size=chunk_size
                    )
                else:
                    pieces = split_text(
                        "\n".join(unit_lines),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=separators,
                    )
                for piece in pieces:
                    # A table chunk that starts with neither its own heading nor
                    # the context line still needs scheme + section for retrieval.
                    first = piece.splitlines()[0] if piece.strip() else ""
                    if (
                        unit_kind == "table"
                        and context
                        and not HEADING_RE.match(first)
                        and not first.startswith(context)
                    ):
                        piece = f"{context}\n{piece}"
                    if not piece.strip() or _is_chrome_chunk(piece):
                        continue
                    chunks.append(
                        Chunk(
                            id=make_chunk_id(document.source_url, unit_label, index),
                            text=piece,
                            source_url=document.source_url,
                            scheme_name=document.scheme_name,
                            scheme_class=document.scheme_class,
                            fetched_at=document.fetched_at,
                            section=unit_label,
                            chunk_index=index,
                        )
                    )
                    index += 1
    return chunks


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Sequence[str] = SEPARATORS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        )
    return chunks
