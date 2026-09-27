# Disclaimer

> **Facts-only. No investment advice.**

This is the exact, verbatim disclaimer string required by the PRD and shown in the
app (`architecture.md` §6.1, §11). `app.py` renders it from the single
`DISCLAIMER` constant in `src/guardrails.py`, so the on-screen text and this file
cannot drift apart.

## What this bot does

- Answers factual questions about **five HDFC Mutual Fund Direct-Growth scheme
  pages** published on Groww, using only text extracted from those pages:
  HDFC Large Cap Fund, HDFC Flexi Cap Fund, HDFC ELSS Tax Saver Fund,
  HDFC Small Cap Fund, and HDFC Balanced Advantage Fund.
- Cites the page a fact came from and stamps each answer with the date that page
  was last fetched (`Last updated from sources: [Date]`).
- Refuses, by design, to recommend buying, selling, or holding anything, and will
  not compute, compare, or forecast returns.

## What this bot is not

- **Not a SEBI-registered investment advisor.** The disclaimer shown in the app is
  a product control, not a substitute for legal review.
- Not personalised: it has no view of your holdings, goals, or tax situation, and it
  asks for and stores no personal identifiers (no PAN, Aadhaar, folio, account
  number, OTP, email address, or phone number).
- Not a returns calculator or portfolio optimiser, and not a general mutual-fund
  assistant: other AMCs, other schemes, and market news are out of scope.
- Not a source of live data. Answers are only as fresh as the last successful
  ingest, and facts that exist only in a factsheet or SID PDF may be absent.
- No financial, legal, or tax advice. Consult a SEBI-registered advisor and read the
  official scheme documents before acting on anything.

## Data and privacy

The only network source is the five allowlisted public Groww pages listed in
`src/sources.py`. No account login, no file upload, and no user database: messages
are held in the browser session only and are not written to disk by this app.
