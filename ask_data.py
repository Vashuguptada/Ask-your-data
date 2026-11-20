"""Natural language -> SQL -> answer, using LangChain with safety guards."""
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
DB_PATH = os.getenv("DB_PATH", str(Path(__file__).resolve().with_name("ecommerce.db")))
MAX_ROWS = 200
TIMEOUT_SEC = 5

# Business definitions the model cannot guess from the schema alone (a mini "semantic layer").
GLOSSARY = """- revenue = quantity * products.price * (1 - discount_pct / 100.0)
- Revenue counts ONLY orders with status = 'delivered' (exclude 'cancelled' and 'returned').
- gross profit = quantity * (products.price * (1 - discount_pct / 100.0) - products.cost), delivered orders only.
- average order value (AOV) = revenue / number of distinct delivered orders.
- Dates are ISO text (YYYY-MM-DD); use strftime() for month/year. Data covers 2024-06 to 2026-10.
- Return rate = returned orders / total orders."""

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful data analyst who writes SQLite queries.\n\n"
     "Schema:\n{schema}\n\nBusiness definitions:\n{glossary}\n\n"
     "Rules:\n- Output exactly ONE read-only SELECT (or WITH ... SELECT) statement.\n"
     "- No explanation, no markdown, no comments.\n"
     "- Use only tables and columns that exist in the schema.\n"
     "- If the requested subject is not represented in the schema, do not invent or substitute another field.\n"
     "- Use conversation history only to resolve references such as 'that', 'same period', or 'why'."),
    ("human",
     "Conversation so far (use only to understand follow-up questions):\n{history}\n\n"
     "Current question: {question}"),
])

REPAIR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You fix SQLite queries.\n\nSchema:\n{schema}\n\nBusiness definitions:\n{glossary}\n\n"
     "Return ONLY the corrected single SELECT statement. No explanation, no markdown."),
    ("human", "Question: {question}\nFailed SQL: {sql}\nError: {error}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a rigorous business data analyst. Use the schema, business definitions, conversation, "
     "SQL, and query result to answer the user's current question. Do not invent numbers or claim "
     "the dataset contains fields it does not contain. "
     "Format large numbers with thousands separators. Keep labels and values clearly separated. "
     "Write 'Jan: 670,450', never 'Jan670,450'. For three or more result rows, use a compact "
     "Markdown table or short bullet list instead of squeezing values into one sentence. "
     "Start with a direct takeaway, then explain what the returned figures support. "
     "Distinguish observed facts from interpretation. Do not state that an association proves "
     "causation, or promise what will happen in the future. If the requested answer depends on "
     "missing data, say exactly what is missing and which fields would be needed. Do not stop "
     "at 'no data' when the user is asking for a recommendation: give careful general guidance, "
     "label it as a hypothesis or proposed test, and never present it as a result from this dataset. "
     "For ad-spend questions without campaign data, explain that uplift cannot be concluded and "
     "suggest measuring spend, impressions, clicks, conversions, revenue, and a control group. "
     "Do not mention row or column counts unless asked. Do not reveal private chain-of-thought; "
     "give a concise evidence-based explanation instead. If the result is empty, explain that "
     "no matching data was found."),
    ("human", "Schema:\n{schema}\n\nBusiness definitions:\n{glossary}\n\n"
     "Conversation so far:\n{history}\n\nCurrent question: {question}\nSQL: {sql}\n"
     "Query result:\n{result}"),
])

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|truncate)\b", re.I)


def get_llm():
    """Pick the LLM from env vars. Default is Groq's free tier."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "llama-3.3-70b-versatile", temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash", temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "qwen2.5-coder:7b", temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _readonly_uri(db_path=None):
    path = Path(db_path or DB_PATH).expanduser().resolve()
    return f"{path.as_uri()}?mode=ro"


def connect_readonly(db_path=None):
    con = sqlite3.connect(_readonly_uri(db_path), uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def get_schema(db_path=None):
    with connect_readonly(db_path) as con:
        tables = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        sections = []
        for name, ddl in tables:
            quoted_name = '"' + name.replace('"', '""') + '"'
            sample = con.execute(f"SELECT * FROM {quoted_name} LIMIT 2").fetchall()
            sections.append(f"{ddl}\n-- sample rows: {sample}")
        return "\n\n".join(sections)


def clean_sql(text) -> str:
    text = getattr(text, "content", text)
    text = re.sub(r"```(?:sql)?", "", str(text), flags=re.I).strip()
    return text.rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Guard layer 1: only a single read-only SELECT/WITH statement is allowed."""
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    if ";" in sql or "--" in sql or "/*" in sql:
        raise ValueError("Multiple statements or comments are not allowed.")
    if FORBIDDEN.search(sql):
        raise ValueError("Query contains a forbidden keyword.")
    return sql


def run_query(sql: str, db_path=None) -> pd.DataFrame:
    """Guard layer 2: read-only connection, row cap and a timeout."""
    with connect_readonly(db_path) as con:
        start = time.time()
        con.set_progress_handler(lambda: 1 if time.time() - start > TIMEOUT_SEC else 0, 100000)
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        columns = _unique_column_names([d[0] for d in cur.description])
        return pd.DataFrame(rows, columns=columns)


def _format_chat_history(chat_history) -> str:
    if not chat_history:
        return "(No previous turns.)"
    lines = []
    for message in chat_history[-8:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:1200]}")
        sql = message.get("sql")
        if role == "Assistant" and sql:
            lines.append(f"Previous SQL: {str(sql)[:1500]}")
    return "\n".join(lines) or "(No previous turns.)"


def _unique_column_names(columns) -> list[str]:
    seen = {}
    unique = []
    for index, column in enumerate(columns, start=1):
        base = str(column or f"column_{index}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        unique.append(base if count == 0 else f"{base}_{count + 1}")
    return unique


def ask(question: str, llm=None, explain: bool = True, db_path=None, glossary=None,
        chat_history=None, on_stage=None, on_answer_chunk=None) -> dict:
    """Returns {sql, df, answer, error, retried}."""
    out = {"sql": None, "df": None, "answer": None, "error": None, "retried": False}
    try:
        if on_stage:
            on_stage("Reading your data")
        llm = llm or get_llm()
        schema = get_schema(db_path)
        active_glossary = glossary if glossary is not None else GLOSSARY
        ctx = {
            "schema": schema,
            "glossary": active_glossary,
            "history": _format_chat_history(chat_history),
            "question": question,
        }
        if on_stage:
            on_stage("Writing a safe query")
        sql = validate_sql(clean_sql((SQL_PROMPT | llm | StrOutputParser()).invoke(ctx)))
        out["sql"] = sql
        try:
            if on_stage:
                on_stage("Running the query")
            df = run_query(sql, db_path)
        except sqlite3.Error as e:  # self-correction: give the model its own error once
            out["retried"] = True
            if on_stage:
                on_stage("Adjusting the query")
            sql = validate_sql(clean_sql(
                (REPAIR_PROMPT | llm | StrOutputParser()).invoke({**ctx, "sql": sql, "error": str(e)})))
            out["sql"] = sql
            df = run_query(sql, db_path)
        out["df"] = df
        if explain:
            if on_stage:
                on_stage("Preparing an evidence-based answer")
            answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
            answer_ctx = {
                **ctx,
                "sql": sql,
                "result": df.head(20).to_string(index=False),
            }
            if on_answer_chunk:
                chunks = []
                for chunk in answer_chain.stream(answer_ctx):
                    text = str(chunk)
                    chunks.append(text)
                    on_answer_chunk(text)
                out["answer"] = "".join(chunks)
            else:
                out["answer"] = answer_chain.invoke(answer_ctx)
    except Exception as e:  # blocked by guard, DB error after retry, or LLM/API error
        out["error"] = f"{type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the total revenue from delivered orders?"
    res = ask(q)
    print("SQL:", res["sql"])
    print(res["df"] if res["df"] is not None else "ERROR: " + str(res["error"]))
    print("\nAnswer:", res["answer"])
