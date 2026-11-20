"""Accuracy check: 20 questions with gold SQL. Compares RESULT SETS, not SQL text.

Usage:
  python eval.py --mock            # sanity-check the harness (uses gold SQL as the 'model')
  python eval.py --sleep 2         # real run against your LLM (sleep avoids free-tier rate limits)
"""
import argparse
import time

import pandas as pd
from langchain_core.runnables import RunnableLambda

from ask_data import ask, get_llm, run_query

REV = "oi.quantity * p.price * (1 - oi.discount_pct / 100.0)"
FROM = ("FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
        "JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'delivered'")

TESTS = [
    ("How many customers do we have?", "SELECT COUNT(*) FROM customers"),
    ("How many orders were placed in total?", "SELECT COUNT(*) FROM orders"),
    ("How many orders were cancelled?", "SELECT COUNT(*) FROM orders WHERE status='cancelled'"),
    ("What is the total revenue from delivered orders?", f"SELECT ROUND(SUM({REV}),2) {FROM}"),
    ("Top 5 products by revenue", f"SELECT p.name, ROUND(SUM({REV}),2) r {FROM} GROUP BY p.product_id ORDER BY r DESC LIMIT 5"),
    ("Revenue by product category", f"SELECT p.category, ROUND(SUM({REV}),2) {FROM} GROUP BY p.category"),
    ("Which sales channel has the most orders?", "SELECT channel FROM orders GROUP BY channel ORDER BY COUNT(*) DESC LIMIT 1"),
    ("Monthly revenue in 2026", f"SELECT strftime('%Y-%m', o.order_date) m, ROUND(SUM({REV}),2) {FROM} AND o.order_date >= '2026-01-01' AND o.order_date < '2027-01-01' GROUP BY m"),
    ("What is the average order value?", f"SELECT ROUND(SUM({REV}) / COUNT(DISTINCT o.order_id),2) {FROM}"),
    ("Which city has the most customers?", "SELECT city FROM customers GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1"),
    ("How many customers placed more than 7 orders?", "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 7)"),
    ("Return rate by channel", "SELECT channel, ROUND(1.0 * SUM(status='returned') / COUNT(*), 4) FROM orders GROUP BY channel"),
    ("Top 3 customers by revenue", f"SELECT c.name, ROUND(SUM({REV}),2) r {FROM.replace('WHERE', 'JOIN customers c ON c.customer_id = o.customer_id WHERE')} GROUP BY c.customer_id ORDER BY r DESC LIMIT 3"),
    ("Average discount percentage by category", "SELECT p.category, ROUND(AVG(oi.discount_pct),2) FROM order_items oi JOIN products p ON p.product_id = oi.product_id GROUP BY p.category"),
    ("Which state generates the most revenue?", f"SELECT c.state {FROM.replace('WHERE', 'JOIN customers c ON c.customer_id = o.customer_id WHERE')} GROUP BY c.state ORDER BY SUM({REV}) DESC LIMIT 1"),
    ("What was the revenue in Q3 2026 (July to September)?", f"SELECT ROUND(SUM({REV}),2) {FROM} AND o.order_date BETWEEN '2026-07-01' AND '2026-09-30'"),
    ("Which category has the highest gross profit?", f"SELECT p.category {FROM} GROUP BY p.category ORDER BY SUM(oi.quantity * (p.price * (1 - oi.discount_pct / 100.0) - p.cost)) DESC LIMIT 1"),
    ("How many customers signed up in 2026?", "SELECT COUNT(*) FROM customers WHERE signup_date >= '2026-01-01' AND signup_date < '2027-01-01'"),
    ("How many customers have never placed an order?", "SELECT COUNT(*) FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)"),
    ("Which month had the highest revenue overall?", f"SELECT strftime('%Y-%m', o.order_date) m {FROM} GROUP BY m ORDER BY SUM({REV}) DESC LIMIT 1"),
]


def norm(df: pd.DataFrame):
    """Order-insensitive, rounding-tolerant representation of a result set."""
    return sorted(tuple(round(v, 1) if isinstance(v, float) else v for v in row)
                  for row in df.itertuples(index=False, name=None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sleep", type=float, default=0)
    args = ap.parse_args()

    rows, passed = [], 0
    for question, gold in TESTS:
        llm = RunnableLambda(lambda _p, g=gold: g) if args.mock else get_llm()
        t0 = time.time()
        res = ask(question, llm=llm, explain=False)
        secs = round(time.time() - t0, 2)
        ok = res["df"] is not None and norm(res["df"]) == norm(run_query(gold))
        passed += ok
        rows.append({"question": question, "correct": ok, "retried": res["retried"], "seconds": secs,
                     "generated_sql": res["sql"], "error": res["error"]})
        print(("PASS" if ok else "FAIL"), "-", question)
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    df.to_csv("eval_results.csv", index=False)
    print(f"\nAccuracy: {passed}/{len(TESTS)} = {passed / len(TESTS):.0%} | "
          f"retries used: {int(df.retried.sum())} | avg latency: {df.seconds.mean():.2f}s")
    print("Details saved to eval_results.csv")


if __name__ == "__main__":
    main()
