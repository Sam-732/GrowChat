"""Minimal chat surface for the facts-only assistant (architecture §6.1, §3.1, §8).

All retrieval and generation logic lives in `src/pipeline.py`; this file only
renders. §8: there is no login, no file upload, and no field for a PAN, Aadhaar,
folio, account number, email address, or phone number, and the page never asks for
one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.generate import LLMError, LLMNotConfiguredError  # noqa: E402
from src.guardrails import DISCLAIMER  # noqa: E402
from src.pipeline import answer  # noqa: E402
from src.retrieve import IndexNotBuiltError  # noqa: E402
from src.sources import SCHEMES  # noqa: E402

EXAMPLES = (
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the minimum SIP amount for HDFC Small Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
)
WELCOME = (
    "Ask me **facts** about these five HDFC Mutual Fund Direct-Growth scheme pages "
    "on Groww:\n\n"
    + "\n".join(f"- {scheme['scheme_name']}" for scheme in SCHEMES)
    + "\n\nI answer only from the text on those pages, cite the page, and stamp the "
    "date it was fetched. I do not give investment advice and do not discuss returns."
)
NO_INDEX = (
    "The vector index is not built yet. Run `python scripts/build_index.py` "
    "in a terminal, then ask again."
)
NO_LLM = (
    "No LLM is configured, so a written answer cannot be produced. Copy "
    "`.env.example` to `.env`, set `LLM_API_KEY` and `LLM_MODEL`, then restart. "
    "Refusal and out-of-scope questions still work without a key."
)
_BARE_URL_RE = re.compile(r"(?<![(\[<])(https?://[^\s)\]]+)")


def with_links(text: str) -> str:
    """Make the bare URLs in an answer clickable without touching its wording."""
    return _BARE_URL_RE.sub(r"<\1>", text or "")


def ask(question: str) -> str:
    """The single call into the pipeline, with operator errors shown in the page."""
    try:
        return answer(question)
    except IndexNotBuiltError:
        return NO_INDEX
    except LLMNotConfiguredError:
        return NO_LLM
    except LLMError as exc:
        return f"The language model could not be reached: {exc}"


st.set_page_config(page_title="HDFC scheme facts assistant", layout="centered")

st.sidebar.markdown(f"**{DISCLAIMER}**")
st.sidebar.markdown(
    "Scope: five HDFC Direct-Growth Groww pages. Facts only, no advice, no returns."
)

st.title("HDFC scheme facts assistant")
st.warning(DISCLAIMER)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(WELCOME)

example = None
columns = st.columns(len(EXAMPLES))
for column, question in zip(columns, EXAMPLES):
    if column.button(question, key=f"example-{question}"):
        example = question

typed = st.chat_input("Ask a fact about expense ratio, exit load, min SIP, risk, benchmark")
question = typed or example

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        st.markdown(with_links(ask(question)))

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(with_links(message["content"]))
