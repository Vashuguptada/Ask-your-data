import pathlib

import streamlit as st
from pathlib import Path
import subprocess
import sys
from ask_data import ask

st.set_page_config(page_title="Ask Your Data", page_icon="💬", layout="wide")
st.title("💬 Ask Your Data")
st.caption("Natural language → SQL → answer, on a synthetic fashion e-commerce database (LangChain + SQLite)")

SAMPLES = ["What is the total revenue from delivered orders?",
           "Revenue by product category",
           "Monthly revenue in 2026",
           "Return rate by channel",
           "Top 5 products by revenue"]

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
            st.info("The first query failed, so the model corrected it once (self-repair).")
    if res["df"] is not None:
        if res["answer"]:
            st.subheader("Answer")
            st.write(res["answer"])
        st.subheader("Result")
        st.dataframe(res["df"], use_container_width=True)
        df = res["df"]
        if df.shape[1] == 2 and df.shape[0] > 1 and df.dtypes.iloc[1].kind in "if":
            st.bar_chart(df.set_index(df.columns[0]))
