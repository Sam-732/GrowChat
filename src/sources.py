"""Closed allowlist of HDFC Direct-Growth Groww pages. Source of truth for ingest."""

from __future__ import annotations

from typing import Iterator, TypedDict
from urllib.parse import urldefrag, urlparse


class Scheme(TypedDict):
    scheme_class: str
    scheme_name: str
    url: str


SCHEMES: list[Scheme] = [
    {
        "scheme_class": "Large Cap",
        "scheme_name": "HDFC Large Cap Fund",
        "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    },
    {
        "scheme_class": "Flexi Cap",
        "scheme_name": "HDFC Flexi Cap Fund",
        "url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    },
    {
        "scheme_class": "ELSS",
        "scheme_name": "HDFC ELSS Tax Saver Fund",
        "url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    },
    {
        "scheme_class": "Small Cap",
        "scheme_name": "HDFC Small Cap Fund",
        "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    },
    {
        "scheme_class": "Balanced Advantage",
        "scheme_name": "HDFC Balanced Advantage Fund",
        "url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    },
]


def normalize_url(url: str) -> str:
    """Canonical form for comparison: no fragment, no trailing slash, lowercased."""
    stripped, _frag = urldefrag(url.strip())
    parsed = urlparse(stripped)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


_ALLOWLIST = {normalize_url(s["url"]) for s in SCHEMES}


def iter_schemes() -> Iterator[Scheme]:
    yield from SCHEMES


def is_allowlisted(url: str) -> bool:
    try:
        return normalize_url(url) in _ALLOWLIST
    except Exception:
        return False
