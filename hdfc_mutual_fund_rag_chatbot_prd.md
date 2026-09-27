# Product Requirements Document (PRD): HDFC Mutual Fund RAG Chatbot

## 1. Product Overview
The **HDFC Mutual Fund RAG Chatbot** is a fact-based, Retrieval-Augmented Generation (RAG) assistant designed to answer specific queries about mutual fund schemes. The chatbot operates strictly on officially published public pages, providing retail investors and customer support teams with instant, accurate, and cited information without dispensing financial advice or performance predictions.

## 2. Target Audience & Use Cases
*   **Retail Investors:** Users actively researching and comparing specific HDFC mutual fund schemes who need quick access to factual data (e.g., exit loads, minimum SIP amounts).
*   **Customer Support & Content Teams:** Internal teams requiring a quick reference tool to answer repetitive, high-volume factual questions about fund mechanics.

## 3. Scope & Data Corpus
The data corpus is strictly limited to 5 specific scheme pages from a single Asset Management Company (HDFC) hosted on Groww. No third-party blogs or external opinions are permitted in the knowledge base.

**Approved Data Sources (Direct-Growth Plans):**
1.  **Large Cap:** [HDFC Large Cap Fund](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth)
2.  **Flexi Cap:** [HDFC Flexi Cap Fund](https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth)
3.  **ELSS:** [HDFC ELSS Tax Saver Fund](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth)
4.  **Small Cap:** [HDFC Small Cap Fund](https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth)
5.  **Balanced Advantage:** [HDFC Balanced Advantage Fund](https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth)

## 4. Core Features & Functionality

### 4.1. Factual Question Answering
The bot must accurately extract and serve answers regarding:
*   Expense ratios
*   Exit loads and lock-in periods (e.g., ELSS constraints)
*   Minimum SIP / Lumpsum investment amounts
*   Riskometer ratings and scheme benchmarks
*   Procedural facts (e.g., "How to download capital-gains statements")

### 4.2. Response Guardrails
*   **Mandatory Citations:** Every single response must include at least one clear source link pointing back to the specific URL used to generate the answer.
*   **Length Constraint:** Answers must be concise and limited to **≤ 3 sentences**.
*   **Freshness Stamp:** Every answer must conclude with the tag: *"Last updated from sources: [Date]"*.
*   **Anti-Advice Mechanism:** The system must refuse opinionated, advisory, or portfolio-specific questions (e.g., "Should I buy this?", "Is this a good investment?"). The fallback response must be a polite, facts-only refusal accompanied by a relevant educational link.

### 4.3. User Interface (UI)
A minimal, lightweight frontend (app or notebook interface) featuring:
*   A clear welcome message.
*   A visible disclaimer snippet: *"Facts-only. No investment advice."*
*   3 clickable example queries to guide user interaction.

## 5. Technical Architecture (RAG Pipeline)

| Stage | Technology / Strategy | Description |
| :--- | :--- | :--- |
| **1. Data Ingestion** | Web Scraper/Loader | Extract raw text, tables, and metadata from the 5 approved URLs. |
| **2. Chunking** | Dynamic (AI-Assisted) | Cursor/AI to evaluate the ingested HTML/text and select the optimal strategy (Recursive Character or Semantic Chunking) to preserve tabular data (e.g., fee structures) and contextual boundaries. |
| **3. Embedding** | Hugging Face | `sentence-transformers/all-MiniLM-L6-v2` for generating fast, high-quality semantic vector embeddings. |
| **4. Vector Database** | ChromaDB | Local/managed vector store to house chunked embeddings for efficient similarity search and retrieval. |
| **5. Generation** | LLM Integration | Prompt-engineered LLM instructed to synthesize retrieved chunks into ≤3 sentence factual answers, appending the required citation and date stamp. |

## 6. Security, Compliance & Constraints
*   **Strictly Public Sources:** No backend API integrations, proprietary databases, or screenshots.
*   **Zero PII Data:** The system architecture must not prompt for, accept, or store Personally Identifiable Information (PII) including PAN cards, Aadhaar, account numbers, OTPs, email addresses, or phone numbers.
*   **No Performance Claims:** The LLM prompt must explicitly forbid the computation, comparison, or forecasting of historical or future returns. Queries regarding returns must be routed to the official factsheet link.

## 7. Delivery Milestones & Required Assets
To consider this milestone complete, the following deliverables must be submitted:
1.  **Working Prototype:** A live hosted link (e.g., Streamlit, Gradio, or Vercel) OR a ≤3-minute video demo if live hosting is unfeasible.
2.  **Source Roster:** A CSV or Markdown file cataloging the 5 source URLs.
3.  **Documentation (README):** Outlining setup instructions, architecture breakdown, defined scope (AMC + schemes), and any known limitations.
4.  **Evaluation Set:** A sample Q&A document containing 5–10 diverse queries, showcasing the bot's generated answers, guardrail triggers, and citation links.
5.  **Disclaimer Asset:** The exact verbatim UI disclaimer snippet utilized in the frontend.