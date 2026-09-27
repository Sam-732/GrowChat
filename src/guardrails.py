"""Inbound and outbound guardrails (architecture §6.2, §6.4, §8).

No LLM, no UI, no network. `classify` runs *before* retrieval: it either passes a
message through as `factual` or returns a canned, fully static refusal string.
`validate` runs on generated text before anything is displayed.

Detection is heuristic and fails closed: a message that looks like advice, a return
forecast, an out-of-scope AMC, or personal data is refused. A `Verdict` keeps only
the classification and the in-scope scheme URL, never the user's raw text, so a
refusal cannot echo a secret (§8 zero-PII by design).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal, Sequence

from src.retrieve import detect_scheme
from src.sources import SCHEMES, is_allowlisted, normalize_url

Class = Literal["factual", "advice", "returns", "out_of_corpus", "pii"]
CLASSES: tuple[Class, ...] = ("factual", "advice", "returns", "out_of_corpus", "pii")

DISCLAIMER = "Facts-only. No investment advice."
FRESHNESS_PREFIX = "Last updated from sources:"
FRESHNESS_LINE_RE = re.compile(
    rf"^{re.escape(FRESHNESS_PREFIX)}\s+(\d{{4}}-\d{{2}}-\d{{2}})\s*$", re.MULTILINE
)
MAX_BODY_SENTENCES = 3
MAX_MATCHED = 3  # capped so a refusal is not a detector-fingerprint oracle

SCHEMES_URLS = tuple(scheme["url"] for scheme in SCHEMES)
_URL_BY_SCHEME = {scheme["scheme_name"]: scheme["url"] for scheme in SCHEMES}
# §6.2 requires an educational link even when the message names nothing in scope.
_DEFAULT_SCHEME = "HDFC Large Cap Fund"
DEFAULT_URL = _URL_BY_SCHEME[_DEFAULT_SCHEME]


def _patterns(*items: tuple[str, str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile detectors. Case-insensitive: users write "CAGR", "SBI", "Should I"."""
    return tuple((name, re.compile(raw, re.IGNORECASE)) for name, raw in items)


# --------------------------------------------------------------------------- #
# Inbound: personal identifiers (§6.2, §8)
# --------------------------------------------------------------------------- #

# Only unambiguous identifier shapes. Keywords need a value beside them so that
# "Pan-India" or a question about accounts never trips the detector, and ordinary
# financial figures are not account numbers.
_PII_PATTERNS = _patterns(
    ("email", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ("pan", r"(?<![A-Za-z0-9])[A-Za-z]{5}[0-9]{4}[A-Za-z](?![A-Za-z0-9])"),
    ("aadhaar", r"(?<![0-9])(?:[0-9]{4}[\s\-]?){2}[0-9]{4}(?![0-9])"),
    ("account_digits", r"(?<![0-9])[0-9]{8,}(?![0-9])"),
    (
        "phone",
        r"(?<![0-9])(?:\+?91[\s\-]?)?[6-9][0-9]{4}[\s\-]?[0-9]{5}(?![0-9])",
    ),
    ("upi_id", r"\b[\w.\-]{2,}@[a-z]{2,}\b"),
    (
        "secret_keyword",
        r"\b(?:otp|one[\s\-]time\s+password|verification\s+code|security\s+code"
        r"|authenticator\s+code|mpin|atm\s+pin|net\s?banking\s+password|ifsc)\b",
    ),
    (
        "identifier_keyword",
        r"\b(?:pan|aadhaar|account\s+no\.?|account\s+number|acct\s+no\.?"
        r"|folio\s+no\.?|folio\s+number|upi\s+id)\b\s*(?:is\b|[:=]|[0-9])",
    ),
)


# --------------------------------------------------------------------------- #
# Inbound: out of corpus (§6.2)
# --------------------------------------------------------------------------- #

_OTHER_AMC_NAMES = (
    "sbi", "icici", "axis", "kotak", "aditya birla", "dsp", "tata", "l&t", "nippon",
    "sundaram", "franklin", "motilal", "mirae", "quantum", "parag parikh", "canara",
    "idbi", "invesco", "hsbc", "bandhan", "pgim", "mahindra", "uti", "lic", "baroda",
    "pnb", "bajaj", "shriram", "jupiter", "whiteoak", "samco", "taurus", "navi",
    "groww",
)

# Sibling schemes, other products, and market topics. The five in-scope categories
# (large cap, flexi cap, ELSS/tax saver, small cap, balanced advantage) must never
# appear here, and no "nifty" pattern: it appears inside a published benchmark.
_OTHER_SCHEME_MARKERS = (
    r"\bmid[\s\-]?cap\b",
    r"\bsmallcap\s+250\b",
    r"\bbandhan\b",
    r"\bdividend\s+yield\b",
    r"\bvalue\s+fund\b",
    r"\bmoney\s+market\b",
    r"\bcorporate\s+bond",
    r"\bbond\s+fund\b",
    r"\bliquid\s+fund\b",
    r"\bultra\s+short\b",
    r"\bduration\s+fund\b",
    r"\bcredit\s+risk\b",
    r"\bgold\s+(?:fund|etf|trust)\b",
    r"\bbanking\s+and\s+psu\b",
    r"\binfrastructure\s+fund\b",
    r"\bpharma\b",
    r"\bconsumption\s+fund\b",
    r"\benergy\s+fund\b",
    r"\bglobal\s+(?:fund|opportunities|index)\b",
    r"\bemerging\s+market\b",
    r"\bindex\s+fund\b",
    r"\betf\b",
    r"\belss\s+index\b",
    r"\bidcw\b",
    r"\bdividend\s+option\b",
    r"\bppf\b",
    r"\bepf\b",
    r"\bnps\b",
    r"\bsavings\s+account\b",
    r"\bfixed\s+deposit\b",
    r"\bterm\s+insurance\b",
    r"\bcredit\s+card\b",
    r"\bhdfc\s+bank\b",
    r"\bstock\s+market\b",
    r"\bshare\s+price\b",
    r"\bmarket\s+news\b",
    r"\bnews\s+about\b",
    r"\bipo\b",
    r"\bgold\s+rate\b",
    r"\busd\s*/\s*inr\b",
)

_OUT_OF_CORPUS_PATTERNS = _patterns(
    ("other_amc", rf"\b(?:{'|'.join(_OTHER_AMC_NAMES)})\b"),
    *((f"other_scheme:{marker}", marker) for marker in _OTHER_SCHEME_MARKERS),
)


# --------------------------------------------------------------------------- #
# Inbound: advice and returns (§6.2)
# --------------------------------------------------------------------------- #

# §6.2 keeps procedural questions in scope, so a bare "should I" is not advice:
# only a buy/sell/hold directive is.
_ADVICE_PATTERNS = _patterns(
    (
        "buy_sell_directive",
        r"\bshould\s+(?:i|we|you)\s+(?:buy|sell|invest|hold|exit|redeem|switch"
        r"|start|pick|choose|add|park)\b|\b(?:i|we)\s+should\s+(?:buy|sell|invest|hold)\b",
    ),
    (
        "good_investment",
        r"\b(?:is|are|was)\s+(?:this|that|it)?\s*(?:a\s+|an\s+)?good\s+"
        r"(?:investment|buy|fund|scheme|option|choice)\b"
        r"|\b(?:good|bad|poor|safe|unsafe)\s+(?:investment|buy)\b",
    ),
    (
        "recommendation",
        r"\brecommend(?:ation|ations|ed|s)?\b|\bsuggest(?:ion|ions|ed)?\b"
        r"|\bmy\s+advice\b|\badvise\s+me\b",
    ),
    (
        "opinion_pick",
        r"\b(?:best|top|right|safest|fastest|cheapest)\s+"
        r"(?:fund|scheme|option|choice|investment|pick)\b"
        r"|\bwhich\s+should\s+i\b"
        r"|\bwhich\b[^.?]{0,40}\b(?:better|best|to\s+buy|to\s+invest)\b",
    ),
    ("worth_it", r"\bworth\s+(?:investing|buying|it)\b|\bis\s+it\s+worth\b"),
    (
        "portfolio",
        r"\bportfolio\s+(?:for\s+me|allocation|planner|construction|review)\b"
        r"|\bhow\s+should\s+i\s+allocate\b|\brebalance\b|\basset\s+allocation\s+for\b",
    ),
    (
        "timing",
        r"\b(?:right|good|best)\s+time\s+to\s+(?:buy|invest|enter|start)\b"
        r"|\bis\s+now\s+a\s+good\s+time\b|\btiming\s+the\s+market\b",
    ),
    (
        "amount_advice",
        r"\bhow\s+much\s+(?:money\s+)?should\s+i\b"
        r"|\bhow\s+much\s+should\s+i\s+(?:invest|put|lump)\b",
    ),
    ("switch", r"\bshould\s+i\s+switch\b|\bswitch\s+(?:from|out\s+of)\b"),
)

_RETURNS_PATTERNS = _patterns(
    ("returns_word", r"\breturns?\b|\bcagr\b|\bxirr\b|\birr\b"),
    (
        "performance",
        r"\bperformance\b|\bperform(?:ed|s|ing)?\b|\boutperform\w*\b|\bunderperform\w*\b"
        r"|\bbeat(?:s|ing)?\s+the\b|\bbest\s+perform\w*\b|\btop\s+perform\w*\b"
        r"|\bvs\.?\s+(?:the\s+)?(?:benchmark|index|nifty|sensex)\b",
    ),
    (
        "horizon_returns",
        r"\b(?:1|3|5|10)\s*(?:year|yr|month)s?\b[^.?]{0,40}\breturn"
        r"|\breturn[^.?]{0,40}\b(?:1|3|5|10)\s*(?:year|yr|month)s?\b",
    ),
    (
        "forecast",
        r"\bpredict\w*\b|\bforecast\w*\b|\bproject\w*\b|\bexpect\w*\b"
        r"|\bwill\s+(?:it|this|these|the\s+fund|these\s+funds)\b"
        r"|\bshould\s+give\b|\bgoing\s+to\s+give\b|\bhow\s+much\s+(?:will|money)\b"
        r"|\bguarantee\w*\b|\bensur(?:e|es|ing)\s+returns?\b",
    ),
    (
        "nav_forecast",
        r"\b(?:predict|forecast|expected|target|future|projected|estimated)\s+nav\b"
        r"|\bnav\s+(?:prediction|forecast|target|will|next\s+year)\b",
    ),
    # "capital gains statement" is procedural and stays in scope (§6.2).
    (
        "profit_language",
        r"(?<!capital )\bgains?\b|\bprofit\w*\b|\bmake\s+money\b|\bmoney\s+back\b"
        r"|\bdouble\s+your\b|\bmultiply\b",
    ),
    (
        "since_inception",
        r"\bsince\s+inception\b|\bfrom\s+inception\b|\ball[\s\-]time\s+returns?\b"
        r"|\bhistorical\s+returns?\b|\bpast\s+(?:1|3|5|10)[\s\-]year\b",
    ),
)


def _first_match(
    patterns: Sequence[tuple[str, re.Pattern[str]]], text: str
) -> tuple[str, ...]:
    return tuple(name for name, pattern in patterns if pattern.search(text))


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #


class Verdict(str):
    """Inbound classification; compares equal to its class name.

    `classify(text) == "advice"` holds, so a caller can branch on either the object
    or the plain class string. The raw user message is deliberately not kept.
    """

    kind: Class
    matched: tuple[str, ...]
    scheme_url: str | None

    def __new__(
        cls, kind: Class, matched: tuple[str, ...] = (), scheme_url: str | None = None
    ) -> "Verdict":
        self = super().__new__(cls, kind)
        self.kind = kind
        self.matched = matched
        self.scheme_url = scheme_url
        return self

    @property
    def needs_retrieval(self) -> bool:
        """True only for a message that may go on to retrieve and generate."""
        return self.kind == "factual"

    @property
    def response(self) -> str | None:
        """Ready user-facing refusal text, or None when retrieval may continue."""
        if self.kind == "factual":
            return None
        return inbound_message(self.kind, scheme_url=self.scheme_url)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Verdict({self.kind!r}, matched={self.matched!r})"


def classify(user_text: str) -> Verdict:
    """Classify an inbound message (§6.2). Run this before retrieve/generate.

    Precedence is safety-first: `pii` (never echo a secret) beats `out_of_corpus`
    (do not invent), which beats `advice`, which beats `returns`. Anything else is
    `factual`: in-scope is assumed until retrieval shows otherwise.
    """
    text = (user_text or "").strip()
    if not text:
        return Verdict("factual", ("empty",), None)

    # Only the in-scope URL is carried forward, and only when exactly one scheme is
    # named, so a refusal can link a relevant page without storing the message.
    scheme_url = _URL_BY_SCHEME.get(detect_scheme(text) or "")

    for kind, patterns in (
        ("pii", _PII_PATTERNS),
        ("out_of_corpus", _OUT_OF_CORPUS_PATTERNS),
        ("advice", _ADVICE_PATTERNS),
        ("returns", _RETURNS_PATTERNS),
    ):
        matched = _first_match(patterns, text)
        if matched:
            return Verdict(kind, matched[:MAX_MATCHED], scheme_url)
    return Verdict("factual", (), scheme_url)


# --------------------------------------------------------------------------- #
# Inbound canned responses
# --------------------------------------------------------------------------- #


def _newest_fetched_at() -> str | None:
    """Newest `fetched_at` in the index, or None when it is absent/unreadable."""
    try:
        from src.index import get_collection

        collection = get_collection()
        if collection is None:
            return None
        stored = collection.get(include=["metadatas"]) or {}
        dates = [
            str((meta or {}).get("fetched_at", ""))
            for meta in stored.get("metadatas") or []
        ]
        dates = [value for value in dates if value]
        return max(dates) if dates else None
    except Exception:  # no index yet: today's date is the safe default
        return None


def latest_source_date(fallback_today: date | None = None) -> str:
    """Last successful ingest date if available, else today's UTC date (ISO, §9).

    A refusal cites no retrieved chunk, so it falls back to the newest ingest and,
    with no index at all, to today. `fallback_today` overrides only that last step.
    """
    return _newest_fetched_at() or (
        fallback_today or datetime.now(timezone.utc).date()
    ).isoformat()


def inbound_message(
    kind: Class, *, scheme_url: str | None = None, date: str | None = None
) -> str:
    """Canned refusal for a non-`factual` class, with link and freshness stamp."""
    if kind not in CLASSES or kind == "factual":
        raise ValueError(f"no inbound message for class {kind!r}")
    url = scheme_url if scheme_url and is_allowlisted(scheme_url) else DEFAULT_URL
    scheme = _scheme_name_for(url)
    stamp = f"{FRESHNESS_PREFIX} {date or latest_source_date()}"

    if kind == "advice":
        body = (
            "This bot shares published facts from the HDFC scheme pages only, so it "
            "cannot advise on buying, selling, or holding a fund.\n"
            f"The published facts for {scheme} are here: {url}"
        )
    elif kind == "returns":
        body = (
            "This bot does not compute, compare, or forecast returns.\n"
            "For the official return figures and factsheet, use the documents on the "
            f"{scheme} page: {url}"
        )
    elif kind == "out_of_corpus":
        body = (
            "This bot covers only five HDFC Direct-Growth scheme pages, so it cannot "
            "answer questions about other funds, AMCs, or market news.\n"
            f"Here is one of the pages it does cover: {url}"
        )
    else:  # pii: static text only, never echoing the message
        body = (
            "This bot cannot process personal identifiers such as PAN, Aadhaar, account "
            "numbers, OTPs, email addresses, or phone numbers, and nothing from that "
            "message was stored or repeated.\n"
            f"Facts you can ask about instead are on this page: {url}"
        )
    return f"{body}\n{stamp}"


def _scheme_name_for(url: str) -> str:
    target = normalize_url(url)
    return next(
        (name for name, known in _URL_BY_SCHEME.items() if normalize_url(known) == target),
        _DEFAULT_SCHEME,
    )


def insufficient_context(url: str | None = None, date: str | None = None) -> str:
    """Template for a question the indexed pages do not answer (§6.5)."""
    target = url if url and is_allowlisted(url) else DEFAULT_URL
    scheme = _scheme_name_for(target)
    stamp = f"{FRESHNESS_PREFIX} {date or latest_source_date()}"
    return (
        f"{INSUFFICIENT_CONTEXT_LEAD}\n"
        f"The published facts for {scheme} are here: {target}\n{stamp}"
    )


def redact_secrets(text: str) -> str:
    """Mask identifier-shaped substrings before page text reaches the LLM (§8).

    Defense in depth: our pages should hold no personal data, but a layout change
    must never be able to forward one to a model provider.
    """
    redacted = text or ""
    for _name, pattern in _PII_ECHO_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted


# --------------------------------------------------------------------------- #
# Outbound validation (§6.4)
# --------------------------------------------------------------------------- #

_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+", re.IGNORECASE)
# A line holding only a citation is not a body sentence (§6.4 counts sentences).
_CITATION_LINE_RE = re.compile(
    r"^\s*(?:[-*>]\s*)?(?:\[[^\]]*\]\([^)]+\)|https?://\S+)\s*$", re.IGNORECASE
)

_ADVICE_OUTBOUND = _patterns(
    (
        "advice_phrasing",
        r"\byou\s+should\b|\bi\s+(?:would\s+)?recommend\b|\bi\s+suggest\b"
        r"|\bmy\s+advice\s+is\b|\byou\s+can\s+consider\s+investing\b",
    ),
    (
        "advice_phrasing",
        r"\bgood\s+buy\b|\bworth\s+(?:investing|buying)\b|\bdon'?t\s+miss\b"
        r"|\bmust\s+buy\b|\bconsider\s+investing\b|\bsafe\s+investment\b"
        r"|\b(?:best|ideal)\s+(?:fund|scheme|option)\s+for\s+you\b"
        r"|\binvest\s+in\s+this\b",
    ),
    (
        "invented_return_pitch",
        r"\bwill\s+(?:give|deliver|return|grow|earn)\b|\bexpected\s+returns?\b"
        r"|\bprojected\s+returns?\b|\breturn\s+target\b|\bguaranteed\b"
        r"|\bassured\s+returns?\b|\bcan\s+double\b",
    ),
)

# A percentage only counts as invented pitching when a return/performance word sits
# in the same sentence, so a published fee or NAV figure is not blocked.
_RETURN_WORDS = re.compile(
    r"\breturns?\b|\bcagr\b|\bxirr\b|\byield\b|\bperformance\b|\bprofit\w*\b|\bgains?\b"
    r"|\boutperform\w*\b|\bpredict\w*\b|\bforecast\w*\b|\bproject\w*\b|\bexpected\b"
    r"|\bwill\b|\blikely\b|\btarget\b|\bguarantee\w*\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")

# §8 support: only unambiguous identifier shapes, never the digit runs that ordinary
# financial figures produce.
_PII_ECHO_PATTERNS = tuple(
    entry for entry in _PII_PATTERNS if entry[0] in {"email", "pan", "aadhaar", "phone"}
)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of `validate`: ok, or the reasons a repair/fallback is needed."""

    ok: bool
    reasons: tuple[str, ...]
    details: tuple[str, ...]
    body_sentences: int
    urls: tuple[str, ...]
    freshness_date: str | None

    def __bool__(self) -> bool:
        return self.ok


def extract_urls(text: str) -> tuple[str, ...]:
    """Every http(s) URL in the text, trimmed of trailing punctuation."""
    return tuple(match.rstrip(".,;:!?") for match in _URL_RE.findall(text or ""))


def split_freshness_line(text: str) -> tuple[str, str | None]:
    """Split off the `Last updated from sources: [Date]` line (§6.3)."""
    matches = list(FRESHNESS_LINE_RE.finditer(text or ""))
    if not matches:
        return text or "", None
    last = matches[-1]
    body = (text[: last.start()] + text[last.end() :]).strip()
    return body, last.group(1)


def body_sentences(text: str) -> list[str]:
    """Sentences in the answer body, with the freshness line and citations removed.

    Splits on terminal punctuation followed by whitespace, so decimals (`1.03%`)
    and URLs never split a sentence. Splitting an abbreviation is possible and errs
    toward refusing, which is the safe direction.
    """
    body, _stamp = split_freshness_line(text)
    prose = "\n".join(
        line for line in body.splitlines() if not _CITATION_LINE_RE.match(line)
    )
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", prose) if part.strip()]


def validate(
    answer_text: str, allowlisted_urls: Sequence[str] | None = None
) -> ValidationResult:
    """Check a generated answer before it is displayed (§6.4).

    1. body sentence count ≤ 3, freshness line excluded
    2. at least one allowlisted URL
    3. a `Last updated from sources: [Date]` line, in final position
    4. no advice phrasing and no invented return pitching
    """
    text = answer_text or ""
    allowed = SCHEMES_URLS if allowlisted_urls is None else tuple(allowlisted_urls)
    allowed_norm = {normalize_url(url) for url in allowed}
    urls = extract_urls(text)
    cited = [url for url in urls if normalize_url(url) in allowed_norm]
    body, stamp = split_freshness_line(text)
    sentences = body_sentences(text)

    reasons: list[str] = []
    details: list[str] = []

    if len(sentences) > MAX_BODY_SENTENCES:
        reasons.append("body_sentences_exceeds_3")
        details.append(f"{len(sentences)} body sentences")

    if not cited:
        reasons.append("missing_allowlisted_url")
        details.append(f"urls present: {list(urls) or 'none'}")

    if stamp is None:
        if FRESHNESS_PREFIX in text:
            reasons.append("malformed_freshness_line")
            details.append(f"expected '{FRESHNESS_PREFIX} YYYY-MM-DD'")
        else:
            reasons.append("missing_freshness_line")
    elif not text.strip().endswith(f"{FRESHNESS_PREFIX} {stamp}"):
        reasons.append("freshness_line_not_last")
        details.append("content follows the freshness line")

    for reason, pattern in _ADVICE_OUTBOUND:
        match = pattern.search(body)
        if match:
            reasons.append(reason)
            details.append(f"{reason}: {match.group(0)!r}")

    for sentence in sentences:
        if _PERCENT_RE.search(sentence) and _RETURN_WORDS.search(sentence):
            reasons.append("invented_return_pitch")
            details.append(f"percentage in a return context: {sentence[:80]!r}")
            break

    for name, pattern in _PII_ECHO_PATTERNS:
        if pattern.search(text):
            reasons.append("pii_echo")
            details.append(f"repeats an identifier shape: {name}")
            break

    return ValidationResult(
        ok=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        details=tuple(details),
        body_sentences=len(sentences),
        urls=urls,
        freshness_date=stamp,
    )


INSUFFICIENT_CONTEXT_LEAD = (
    "I could not find that on the indexed HDFC scheme pages, so I am not answering it."
)

_FALLBACK_LEADS: dict[str, str] = {
    "body_sentences_exceeds_3": "That answer was longer than the facts-only format allows, so I am withholding it.",
    "missing_allowlisted_url": "I could not tie that answer to an indexed scheme page, so I am not giving it.",
    "missing_freshness_line": "I could not verify the source date for that answer, so I am not giving it.",
    "malformed_freshness_line": "I could not verify the source date for that answer, so I am not giving it.",
    "freshness_line_not_last": "That answer was not in the required display format, so I am withholding it.",
    "advice_phrasing": "That answer read as investment advice, which this bot does not give.",
    "invented_return_pitch": "That answer included return figures, which this bot does not provide.",
    "pii_echo": "That answer would have repeated personal data, so I am withholding it.",
    "insufficient_context": INSUFFICIENT_CONTEXT_LEAD,
}
_FALLBACK_DEFAULT = (
    "I could not verify that answer against the indexed scheme pages, so I am not giving it."
)


def template_fallback(
    reason: str, url: str | None = None, date: str | None = None
) -> str:
    """Safe replacement for an answer that failed `validate` (§6.4)."""
    lead = _FALLBACK_LEADS.get(reason, _FALLBACK_DEFAULT)
    target = url if url and is_allowlisted(url) else DEFAULT_URL
    scheme = _scheme_name_for(target)
    stamp = f"{FRESHNESS_PREFIX} {date or latest_source_date()}"
    return (
        f"{lead}\nThe published facts for {scheme} are here: {target}\n{stamp}"
    )
