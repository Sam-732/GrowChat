# Evaluation set

Offline Q&A set for the HDFC Mutual Fund RAG Chatbot (architecture.md §10, §11 — deliverable D4).
Each row was produced by calling `src.pipeline.answer` (or, where noted, `src.guardrails.classify` /
`src.retrieve.search` directly) against the live Chroma index built by `scripts/build_index.py`. No
answer text below was invented by hand.

**Known limitation at the time this set was captured:** no `LLM_API_KEY` / `LLM_MODEL` is configured
in this environment (`.env` does not exist; see `.env.example`). `answer()` raises
`LLMNotConfiguredError` for any query that needs generation. Guardrail refusals (advice, returns,
out-of-corpus) do **not** need the LLM and are real, final pipeline outputs. For the fact and
procedural queries, `actual_answer` is marked `TODO` and the row instead reports what retrieval
proved (the real top citation URL from the live index), per implementation.md Phase 8: "If the API
is unavailable, leave actual_answer as TODO ... but prefer real runs." Re-run
`python scripts/ask.py "<query>"` for each `TODO` row once an LLM provider is configured, and replace
the label accordingly.

| id | query | category | expected behavior | actual_answer | citation URL(s) | label |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| expense_large_cap | What is the expense ratio of HDFC Large Cap Fund? | fact | Quote the published expense ratio; cite the Large Cap page; ≤3 sentences; freshness stamp. | `TODO` — generation unavailable (`LLMNotConfiguredError`); retrieval confirmed correct: top hit is the Large Cap page. | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | pending_llm |
| exit_load_flexi_cap | What is the exit load for HDFC Flexi Cap Fund? | fact | Quote the published exit load; cite the Flexi Cap (HDFC Equity Fund) page; ≤3 sentences; freshness stamp. | `TODO` — generation unavailable; retrieval confirmed correct: top hit is the Flexi Cap page. | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth | pending_llm |
| min_sip_small_cap | What is the minimum SIP amount for HDFC Small Cap Fund? | fact | Quote the published minimum SIP; cite the Small Cap page; ≤3 sentences; freshness stamp. | `TODO` — generation unavailable; retrieval confirmed correct: top hit is the Small Cap page. | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth | pending_llm |
| elss_lockin | What is the lock-in period for HDFC ELSS Tax Saver Fund? | fact | State the published lock-in as-is; no tax advice; cite the ELSS page; ≤3 sentences; freshness stamp. | `TODO` — generation unavailable; retrieval confirmed correct: top hit is the ELSS Tax Saver page. | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth | pending_llm |
| riskometer_baf | What is the riskometer rating of HDFC Balanced Advantage Fund? | fact | Quote the published riskometer label; cite the Balanced Advantage page; ≤3 sentences; freshness stamp. | `TODO` — generation unavailable; retrieval confirmed correct: top hit is the Balanced Advantage page. | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth | pending_llm |
| capital_gains_statement | How do I download the capital gains statement for HDFC Large Cap Fund? | procedural | Steps only as written on the source page; if the procedure is not on the indexed page, say so and still cite the page; ≤3 sentences; freshness stamp. | `TODO` — generation unavailable; retrieval returned the Large Cap page (procedure content on that page not yet confirmed — verify once generation is available). | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | pending_llm |
| advice_small_cap | Should I buy HDFC Small Cap Fund? | advice | Refuse the buy/sell/hold question; do not retrieve to justify a recommendation; return an educational link to the named scheme's page. | "This bot shares published facts from the HDFC scheme pages only, so it cannot advise on buying, selling, or holding a fund.\nThe published facts for HDFC Small Cap Fund are here: https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth\nLast updated from sources: 2026-09-27" | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth | refusal_ok |
| advice_flexi_good | Is HDFC Flexi Cap Fund a good investment? | advice | Refuse the opinion/advice question; educational link to the named scheme's page; no recommendation. | "This bot shares published facts from the HDFC scheme pages only, so it cannot advise on buying, selling, or holding a fund.\nThe published facts for HDFC Flexi Cap Fund are here: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth\nLast updated from sources: 2026-09-27" | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth | refusal_ok |
| returns_large_cap | What are the 5-year returns of HDFC Large Cap Fund? | returns | Do not compute, compare, or forecast returns; point to the official factsheet/documents section via citation. | "This bot does not compute, compare, or forecast returns.\nFor the official return figures and factsheet, use the documents on the HDFC Large Cap Fund page: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth\nLast updated from sources: 2026-09-27" | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | refusal_ok |
| out_of_corpus_sbi | What is the expense ratio of SBI Bluechip Fund? | out_of_corpus | Refuse as out of scope (not one of the five HDFC schemes); do not invent a figure; link to one in-scope page. | "This bot covers only five HDFC Direct-Growth scheme pages, so it cannot answer questions about other funds, AMCs, or market news.\nHere is one of the pages it does cover: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth\nLast updated from sources: 2026-09-27" | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | refusal_ok |

## Coverage checklist (architecture §10)

- [x] In-scope facts: expense ratio, exit load, min SIP, ELSS lock-in, riskometer (5 rows).
- [x] Procedural query with citation (1 row).
- [x] Guardrail triggers: advice ×2, returns ×1, out-of-corpus ×1 — all `refusal_ok`, each with a citation and freshness stamp, none computed a return or gave a recommendation.
- [x] Citations present on every row (retrieval-confirmed for fact/procedural rows; pipeline-emitted for refusal rows).
- [ ] Fact/procedural rows re-run and labeled `correct_fact` / `wrong_fact` / `missing_citation` once an `LLM_API_KEY` and `LLM_MODEL` are set in `.env` (see `README.md` setup).

## How to re-run this set

```bash
python scripts/ask.py "What is the expense ratio of HDFC Large Cap Fund?"
```

or reuse the harness used to produce this file:

```python
from src.pipeline import answer
print(answer("What is the expense ratio of HDFC Large Cap Fund?"))
```
