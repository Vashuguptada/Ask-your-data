import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Ask Your Data", page_icon="💬", layout="wide")

# App aur database generator ke liye ek hi absolute path set karo.
db_dir = Path(tempfile.gettempdir()) / "ask-your-data"
db_dir.mkdir(parents=True, exist_ok=True)

db_path = db_dir / "ecommerce.db"
os.environ["DB_PATH"] = str(db_path)

# Cloud par database nahi hai toh make_db.py se banao.
if not db_path.exists():
    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "make_db.py"),
            ],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        st.error("Sample database create nahi ho paya.")
        st.code(error.stderr or str(error))
        st.stop()

# Is import ko DB_PATH set karne ke baad hi rakho.
from ask_data import ask

st.title("💬 Ask Your Data")
st.caption(
    "Natural language → SQL → answer, on a synthetic fashion "
    "e-commerce database (LangChain + SQLite)"
)

SAMPLES = [
    "What is the total revenue from delivered orders?",
    "Revenue by product category",
    "Monthly revenue in 2026",
    "Return rate by channel",
    "Top 5 products by revenue",
]

if "q" not in st.session_state:
    st.session_state.q = SAMPLES[0]

cols = st.columns(len(SAMPLES))
for col, sample in zip(cols, SAMPLES):
    if col.button(sample, use_container_width=True):
        st.session_state.q = sample

question = st.text_input("Your question", key="q")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        res = ask(question)

    if res["error"]:
        st.error(res["error"])

    if res["sql"]:
        st.subheader("Generated SQL")
        st.code(res["sql"], language="sql")
        if res["retried"]:
            st.info(
                "The first query failed, so the model corrected it once "
                "(self-repair)."
            )

    if res["df"] is not None:
        if res["answer"]:
            st.subheader("Answer")
            st.write(res["answer"])

        st.subheader("Result")
        st.dataframe(res["df"], use_container_width=True)

        df = res["df"]
        if (
            df.shape[1] == 2
            and df.shape[0] > 1
            and df.dtypes.iloc[1].kind in "if"
        ):
            st.bar_chart(df.set_index(df.columns[0]))