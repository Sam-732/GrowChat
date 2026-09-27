"""Environment-backed settings. Secrets stay in .env, never in source files."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
TOP_K = int(os.getenv("TOP_K", "5"))
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
# Any OpenAI-compatible /chat/completions endpoint: OpenAI, Groq, Together, Ollama.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
FETCH_TIMEOUT_SECONDS = float(os.getenv("FETCH_TIMEOUT_SECONDS", "20"))

# Documented identity for public-page fetches only (architecture §5.1).
HTTP_USER_AGENT = (
    "GrowchatHDFCBot/0.1 "
    "(educational facts-only RAG; public Groww scheme pages; no PII)"
)
