"""Public-page ingest for the closed Groww allowlist.

Implements architecture §4.2 (what is stored), §5.1 (fetch + parse rules),
§8 (public sources only) and §9 (fail visibly, per-URL `fetched_at`).

Public HTML only: one GET per allowlisted URL, no API, no OCR, no screenshots.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

if __package__ in (None, ""):  # allow `python src/ingest.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FETCH_TIMEOUT_SECONDS, HTTP_USER_AGENT, ROOT  # noqa: E402
from src.sources import Scheme, is_allowlisted, iter_schemes  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"

# Elements that never carry scheme facts.
VOID_TAGS = {
    "area", "audio", "base", "br", "canvas", "embed", "form", "iframe", "img",
    "input", "link", "meta", "object", "picture", "source", "track", "video",
}
# Page chrome that is never scheme content. `header` is kept on purpose: the
# scheme page uses it for the scheme title; the site nav bar is matched by class.
CHROME_TAGS = {
    "aside", "footer", "nav", "noscript", "script", "style", "template",
} | VOID_TAGS
CHROME_ROLES = {
    "alert", "banner", "complementary", "contentinfo", "dialog", "menu",
    "menubar", "navigation", "search", "tablist",
}
# Class/id token prefixes for site chrome on groww.in scheme pages.
CHROME_TOKEN_PREFIXES = (
    "advert", "adslot", "appdownload", "appstore", "backtotop", "breadcrumb",
    "consent", "consumer", "cookie", "drawer", "footer", "hamburger", "header20",
    "lazyload", "lne123", "loggedout", "login", "modal", "nav", "navbar",
    "promo", "rodal", "searchbar", "signup", "socialmedia", "sticky",
)
HEADING_PREFIX = {
    "h1": "# ", "h2": "## ", "h3": "### ",
    "h4": "#### ", "h5": "##### ", "h6": "###### ",
}
BLOCK_TAGS = {
    "address", "article", "blockquote", "dd", "div", "dl", "dt", "fieldset",
    "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
    "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "ul",
}
# Container children that may take part in a `key: value` row.
SIMPLE_TAGS = {
    "a", "b", "div", "em", "i", "li", "p", "small", "span", "strong", "td", "th",
    "u", "ul", "ol",
}
# Rows that are pure calls to action, not facts.
CTA_ROW_TEXT = {
    "compare", "see all", "view details", "check past data", "view all",
}
# Content root preference; first match in document order wins.
CONTENT_ROOT_SELECTORS = (
    "main",
    '[role="main"]',
    "div.pw14ContentWrapper",
    "div.layout-main",
    "body",
)
MIN_EXTRACT_CHARS = 400
KV_MAX_LEN = 80
_WS_RE = re.compile(r"\s+")
_HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
_NUMERIC_RE = re.compile(r"^[\d.,:'\s%₹$#()/-]+$")
# A line must carry at least one letter or digit to be worth indexing.
_HAS_FACT_RE = re.compile(r"[0-9A-Za-z₹]")


class IngestError(Exception):
    """A URL failed to fetch or parse. Never index error HTML or boilerplate."""


@dataclass(frozen=True)
class Document:
    """What is stored per architecture §4.2."""

    source_url: str
    scheme_name: str
    scheme_class: str
    fetched_at: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def slug(self) -> str:
        return scheme_slug(self.scheme_name)


def scheme_slug(scheme_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", scheme_name.lower()).strip("-")
    return slug or "scheme"


def fetch_html(url: str) -> str:
    """GET one allowlisted public page. Documented UA, config timeout, redirects."""
    if not is_allowlisted(url):
        raise IngestError(f"URL is not on the closed allowlist: {url}")

    headers = {
        "User-Agent": HTTP_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }
    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise IngestError(f"HTTP error fetching {url}: {exc}") from exc

    if response.status_code != 200:
        raise IngestError(
            f"Non-200 for {url}: HTTP {response.status_code}. Refusing to index."
        )
    body = response.text or ""
    if not body.strip():
        raise IngestError(f"Empty body for {url}. Refusing to index.")
    return body


def _tokens(tag: Tag) -> list[str]:
    values: list[str] = []
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    values.extend(str(c).lower() for c in classes)
    tag_id = tag.get("id")
    if tag_id:
        values.append(str(tag_id).lower())
    return values


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    if str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    style = str(tag.get("style", ""))
    return bool(_HIDDEN_STYLE_RE.search(style))


def _is_chrome(tag: Tag) -> bool:
    name = (tag.name or "").lower()
    if name in CHROME_TAGS:
        return True
    if str(tag.get("role", "")).lower() in CHROME_ROLES:
        return True
    if any(token.startswith(CHROME_TOKEN_PREFIXES) for token in _tokens(tag)):
        return True
    # A <header> with no heading is a site nav bar, not the scheme title block.
    return name == "header" and tag.find(["h1", "h2", "h3", "h4", "h5", "h6"]) is None


def strip_chrome(soup: BeautifulSoup) -> BeautifulSoup:
    """Drop nav/footer/script/style, hidden overlays, and other obvious chrome."""
    for tag in list(soup.find_all(True)):
        if tag.decomposed or not isinstance(tag, Tag):
            continue
        if _is_chrome(tag) or _is_hidden(tag):
            tag.decompose()
    return soup


def content_root(soup: BeautifulSoup) -> Tag:
    for selector in CONTENT_ROOT_SELECTORS:
        found = soup.select_one(selector)
        if isinstance(found, Tag) and found.get_text(strip=True):
            return found
    return soup.body or soup


def _cell_text(cell: Tag) -> str:
    return _WS_RE.sub(" ", cell.get_text(" ", strip=True)).replace("|", "\\|")


def _span(cell: Tag, attr: str) -> int:
    try:
        return max(1, int(str(cell.get(attr, "1"))))
    except (TypeError, ValueError):
        return 1


def _table_grid(table: Tag) -> list[list[str]]:
    """Rows of text with colspan/rowspan expanded so no cell text is lost."""
    grid: list[list[str]] = []
    carries: dict[tuple[int, int], str] = {}
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    for index, tr in enumerate(rows):
        row: list[str] = []
        # Place values carried down from a rowspan before adding this row's cells,
        # so a carried value can never overwrite a real cell.
        for key in sorted(k for k in carries if k[0] == index):
            column, text = key[1], carries[key]
            while len(row) <= column:
                row.append("")
            if not row[column]:
                row[column] = text
            del carries[key]
        for cell in tr.find_all(["th", "td"]):
            if cell.find_parent("tr") is not tr:
                continue
            text = _cell_text(cell)
            colspan = _span(cell, "colspan")
            rowspan = _span(cell, "rowspan")
            start = len(row)
            row.append(text)
            row.extend([""] * (colspan - 1))
            for extra in range(1, rowspan):
                for offset in range(colspan):
                    carries[(index + extra, start + offset)] = text
        if any(row):
            grid.append(row)
    return grid


def table_to_markdown(table: Tag) -> str:
    """HTML table -> markdown pipe table (architecture §5.1 step 3)."""
    grid = _table_grid(table)
    if not grid:
        return ""
    width = max(len(row) for row in grid)
    padded = [row + [""] * (width - len(row)) for row in grid]
    caption = _cell_text(table.find("caption")) if table.find("caption") else ""
    lines = [f"| {' | '.join(padded[0])} |", f"| {' | '.join(['---'] * width)} |"]
    for row in padded[1:]:
        joined = " | ".join(row).strip()
        if joined.replace("|", "").replace("-", "").strip().lower() in CTA_ROW_TEXT:
            continue
        lines.append(f"| {' | '.join(row)} |")
    if not lines[2:]:
        return ""
    return (f"{caption}\n" if caption else "") + "\n".join(lines)


def _replace_tables(soup: BeautifulSoup) -> int:
    count = 0
    for table in list(soup.find_all("table")):
        markdown = table_to_markdown(table)
        table.replace_with(NavigableString(f"\n\n{markdown}\n\n" if markdown else "\n\n"))
        count += 1
    return count


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _is_simple(tag: Tag) -> bool:
    if tag.name not in SIMPLE_TAGS:
        return False
    return not tag.find(BLOCK_TAGS - SIMPLE_TAGS - {"li"})


def _is_label(text: str) -> bool:
    return bool(re.match(r"[A-Za-z]", text)) and not _NUMERIC_RE.match(text)


def _key_value_pairs(el: Tag) -> list[tuple[str, str]] | None:
    """Label/value rows (e.g. the NAV / expense / min-SIP stat grid) or None.

    None means "not a key:value container"; the caller renders it normally, so a
    false negative only costs formatting, never content.
    """
    children: list[Tag] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            if child.strip():
                return None
        elif isinstance(child, Tag):
            if child.name in VOID_TAGS:
                continue
            children.append(child)
    if len(children) < 2 or len(children) % 2 or any(not _is_simple(c) for c in children):
        return None
    texts = [_norm(c.get_text(" ", strip=True)) for c in children]
    if any(not text or len(text) > KV_MAX_LEN for text in texts):
        return None
    pairs = list(zip(texts[0::2], texts[1::2]))
    if not all(_is_label(key) and not _is_label(value) for key, value in pairs):
        return None
    return pairs


class _Text:
    """Line buffer that keeps inline runs on one line and blocks apart."""

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._lines: list[str] = []

    def add(self, text: str) -> None:
        if not text:
            return
        if "\n" in text:
            self.flush()
            for line in text.splitlines():
                self.line(line)
            return
        self._buf.append(text)

    def flush(self) -> None:
        if self._buf:
            joined = _norm("".join(self._buf))
            self._buf.clear()
            if joined:
                self._lines.append(joined)

    def line(self, text: str) -> None:
        self.flush()
        stripped = text.strip()
        if stripped:
            self._lines.append(stripped)

    def result(self) -> list[str]:
        self.flush()
        return self._lines


def _render(node: Tag, out: _Text) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            if child.strip():
                out.add(str(child))
            continue
        if not isinstance(child, Tag) or child.decomposed:
            continue
        name = (child.name or "").lower()
        if name in VOID_TAGS:
            if name == "br":
                out.flush()
            continue
        if name in HEADING_PREFIX:
            title = _norm(child.get_text(" ", strip=True))
            if title:
                out.line(f"{HEADING_PREFIX[name]}{title}")
            continue
        if name in ("ul", "ol"):
            for position, item in enumerate(
                child.find_all("li", recursive=False), start=1
            ):
                marker = f"{position}." if name == "ol" else "-"
                sub = _Text()
                _render(item, sub)
                body = "\n".join(sub.result())
                if not body:
                    continue
                indented = "\n".join(f"  {line}" if line else line for line in body.splitlines())
                out.line(f"{marker} {indented}")
            continue
        pairs = _key_value_pairs(child)
        if pairs:
            for key, value in pairs:
                out.line(f"{key}: {value}")
            continue
        if name in BLOCK_TAGS:
            out.flush()
            _render(child, out)
            out.flush()
        else:
            # Inline elements often sit next to each other with no whitespace in
            # the source ("Fund benchmark" + "NIFTY 100 ..."); separate them.
            out.add(" ")
            _render(child, out)
            out.add(" ")


def html_to_text(html: str) -> str:
    """HTML -> markdown-ish text: headings, pipe tables, key: value rows."""
    soup = BeautifulSoup(html, "lxml")
    strip_chrome(soup)
    _replace_tables(soup)
    out = _Text()
    _render(content_root(soup), out)
    lines = [
        line
        for line in out.result()
        if _HAS_FACT_RE.search(line) or line.strip().startswith("|")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def write_raw(document: Document) -> Path:
    """Debug dump of the exact text that would be indexed (gitignored)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{document.slug}.txt"
    header = (
        f"source_url: {document.source_url}\n"
        f"scheme_name: {document.scheme_name}\n"
        f"scheme_class: {document.scheme_class}\n"
        f"fetched_at: {document.fetched_at}\n"
        f"chars: {document.char_count}\n"
        f"{'=' * 72}\n\n"
    )
    path.write_text(header + document.text, encoding="utf-8")
    return path


def ingest_url(scheme: Scheme) -> Document:
    url = scheme["url"]
    text = html_to_text(fetch_html(url))
    if len(text) < MIN_EXTRACT_CHARS:
        raise IngestError(
            f"Extract too short for {url}: {len(text)} chars. Possible layout change "
            "or JS-only page. Refusing to index boilerplate."
        )
    return Document(
        source_url=url,
        scheme_name=scheme["scheme_name"],
        scheme_class=scheme["scheme_class"],
        fetched_at=datetime.now(timezone.utc).date().isoformat(),
        text=text,
    )


def ingest_all(*, dump_raw: bool = True, urls: Sequence[str] | None = None) -> list[Document]:
    """Ingest the allowlist. Raises if any URL fails; never returns partial data."""
    schemes = list(iter_schemes())
    if urls is not None:
        by_url = {scheme["url"]: scheme for scheme in schemes}
        schemes = [by_url[url] for url in urls if url in by_url]
        refused = [url for url in urls if url not in by_url]
        if refused:
            raise IngestError(f"URL(s) not on the closed allowlist: {refused}")

    documents: list[Document] = []
    failures: list[str] = []
    for scheme in schemes:
        try:
            document = ingest_url(scheme)
            if dump_raw:
                write_raw(document)
            documents.append(document)
        except IngestError as exc:
            failures.append(str(exc))
    if failures:
        detail = "\n".join(f"- {message}" for message in failures)
        raise IngestError(f"Ingest failed for {len(failures)} of {len(schemes)} URLs:\n{detail}")
    return documents


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the allowlisted Groww pages.")
    parser.add_argument("--url", action="append", dest="urls", help="allowlisted URL only")
    parser.add_argument("--no-raw", action="store_true", help="skip data/raw debug dumps")
    args = parser.parse_args(argv)
    try:
        documents = ingest_all(dump_raw=not args.no_raw, urls=args.urls)
    except IngestError as exc:
        print(exc, file=sys.stderr)
        return 1
    for document in documents:
        print(
            f"{document.scheme_name:<34} {document.scheme_class:<20} "
            f"{document.char_count:>7} chars  fetched_at={document.fetched_at}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
