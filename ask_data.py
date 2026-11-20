"""Natural language -> SQL -> answer, using LangChain with safety guards."""
import os
import re
import sqlite3
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "ecommerce.db")
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
     "- Use only tables and columns that exist in the schema."),
    ("human", "{question}"),
])

REPAIR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You fix SQLite queries.\n\nSchema:\n{schema}\n\nBusiness definitions:\n{glossary}\n\n"
     "Return ONLY the corrected single SELECT statement. No explanation, no markdown."),
    ("human", "Question: {question}\nFailed SQL: {sql}\nError: {error}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data analyst. Answer the question in 1-3 plain sentences using ONLY the "
     "query result shown. Do not invent numbers. If the result is empty, say no data was found."),
    ("human", "Question: {question}\nSQL: {sql}\nResult:\n{result}"),
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


def connect_readonly():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def get_schema():
    with connect_readonly() as con:
        tables = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        return "\n\n".join(
            f"{ddl}\n-- sample rows: {con.execute(f'SELECT * FROM {name} LIMIT 2').fetchall()}"
            for name, ddl in tables)


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


def run_query(sql: str) -> pd.DataFrame:
    """Guard layer 2: read-only connection, row cap and a timeout."""
    with connect_readonly() as con:
        start = time.time()
        con.set_progress_handler(lambda: 1 if time.time() - start > TIMEOUT_SEC else 0, 100000)
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])


def ask(question: str, llm=None, explain: bool = True) -> dict:
    """Returns {sql, df, answer, error, retried}."""
    llm = llm or get_llm()
    schema = get_schema()
    ctx = {"schema": schema, "glossary": GLOSSARY, "question": question}
    out = {"sql": None, "df": None, "answer": None, "error": None, "retried": False}
    try:
        sql = validate_sql(clean_sql((SQL_PROMPT | llm | StrOutputParser()).invoke(ctx)))
        out["sql"] = sql
        try:
            df = run_query(sql)
        except sqlite3.Error as e:  # self-correction: give the model its own error once
            out["retried"] = True
            sql = validate_sql(clean_sql(
                (REPAIR_PROMPT | llm | StrOutputParser()).invoke({**ctx, "sql": sql, "error": str(e)})))
            out["sql"] = sql
            df = run_query(sql)
        out["df"] = df
        if explain:
            out["answer"] = (ANSWER_PROMPT | llm | StrOutputParser()).invoke(
                {"question": question, "sql": sql, "result": df.head(20).to_string(index=False)})
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
