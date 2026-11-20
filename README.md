# 💬 Ask Your Data: Natural Language to SQL with LangChain

Ask business questions in plain English and get the SQL, the result table and a short answer.
Built on a synthetic fashion e-commerce database (customers, products, orders, order items).

> Add a screenshot here: `![demo](demo.png)`

## Why this project
Business users wait on analysts for simple questions. This demo shows how an LLM can answer
them **safely and measurably**: guarded SQL, a business-definitions layer, self-repair, and an
accuracy benchmark instead of "it seems to work".

## How it works
```
Question ──► LangChain prompt (schema + sample rows + business glossary)
         ──► LLM writes SQL
         ──► Guard 1: only one SELECT/WITH, no comments, no write keywords
         ──► Guard 2: read-only SQLite connection, row cap, 5s timeout
         ──► (if SQL errors) send the error back to the LLM once to self-correct
         ──► Result table ──► LLM summarises in 1-3 sentences (only from the result)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add your free GROQ_API_KEY
python make_db.py                                     # creates ecommerce.db (synthetic data)
streamlit run app.py                                  # UI
python ask_data.py "Revenue by product category"      # or CLI
```
Other providers: set `LLM_PROVIDER=gemini` (+ `GOOGLE_API_KEY`) or `ollama` (local, no key).
Free-tier model names change; if you get "model not found", set `LLM_MODEL` to a current one.

## Evaluation
`python eval.py --sleep 2` runs 20 questions with hand-written gold SQL and compares **result
sets** (order-insensitive, rounded), not SQL text. `python eval.py --mock` checks the harness itself.

| Model | Accuracy | Retries used | Avg latency |
|---|---|---|---|
| _fill after your run_ | _x/20_ | _n_ | _s_ |

**Failure analysis** (fill in after looking at `eval_results.csv`): _which questions failed and why
(wrong join, forgot the delivered-only rule, date logic...) and what you changed in the prompt/glossary._

## Safety design
- Read-only DB connection (`mode=ro`, `PRAGMA query_only`): even a bypassed validator cannot write.
- Validator blocks multi-statement SQL, comments and DDL/DML keywords.
- Row cap and query timeout protect against runaway queries.
- Model answers only from the returned rows, to limit hallucinated numbers.

## Limitations and next steps
- Small schema fits in the prompt; large warehouses need schema retrieval (RAG over table docs).
- Metric definitions live in a text glossary; a real semantic layer (dbt metrics / Power BI semantic model) is better.
- Keyword validator is conservative; a SQL parser (e.g. `sqlglot`) would be more robust.
- Next: caching, conversation memory for follow-ups, per-user permissions.

## Files
`make_db.py` data · `ask_data.py` engine · `app.py` UI · `eval.py` benchmark
