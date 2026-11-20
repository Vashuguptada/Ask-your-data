"""Ask Your Data: a small natural-language analytics workspace."""

import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

LOGGER = logging.getLogger("ask_your_data")

st.set_page_config(
    page_title="Ask Your Data",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
TEMP_ROOT = Path(tempfile.gettempdir()) / "ask-your-data"
DEMO_DIR = TEMP_ROOT / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DB = DEMO_DIR / "ecommerce.db"

# Keep the demo generator and the query engine on the same absolute path.
os.environ["DB_PATH"] = str(DEMO_DB)
if not DEMO_DB.exists():
    try:
        subprocess.run(
            [sys.executable, str(APP_DIR / "make_db.py")],
            cwd=APP_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        LOGGER.error("Demo database creation failed: %s", error.stderr or error)
        st.warning("Demo data is temporarily unavailable. Please try again later.")
        st.stop()

from ask_data import ask, connect_readonly  # noqa: E402

st.session_state.setdefault("show_welcome", True)
st.session_state.setdefault("conversation_id", uuid.uuid4().hex)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
GENERIC_GLOSSARY = (
    "No domain-specific business definitions were provided. Use the schema and "
    "column names carefully. Do not assume definitions that are not stated."
)
DEMO_PROMPTS = [
    "What is the total revenue from delivered orders?",
    "Revenue by product category",
    "Monthly revenue in 2026",
    "Return rate by channel",
    "Top 5 products by revenue",
]
FILE_PROMPTS = [
    "How many records are in the dataset?",
    "What columns and tables are available?",
    "Show the first 10 records",
    "Which groups have the highest values?",
]


def _session_workspace() -> Path:
    st.session_state.setdefault("workspace_id", uuid.uuid4().hex)
    workspace = TEMP_ROOT / "sessions" / st.session_state.workspace_id
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _table_name_from_filename(filename: str) -> str:
    name = re.sub(r"\W+", "_", Path(filename).stem, flags=re.UNICODE).strip("_").lower()
    if not name:
        name = "uploaded_data"
    if name[0].isdigit():
        name = f"dataset_{name}"
    return name


def _unique_columns(columns) -> list[str]:
    used = {}
    result = []
    for index, value in enumerate(columns, start=1):
        base = str(value).strip() or f"column_{index}"
        count = used.get(base, 0)
        used[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def persist_upload(uploaded_file) -> tuple[Path, dict]:
    """Create a session-scoped SQLite source from an uploaded CSV or SQLite file."""
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Files must be 50 MB or smaller.")

    file_name = Path(uploaded_file.name).name
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".csv", ".db", ".sqlite", ".sqlite3"}:
        raise ValueError("Choose a CSV or SQLite database file.")

    fingerprint = hashlib.sha256(file_name.encode("utf-8") + b"\0" + raw).hexdigest()
    workspace = _session_workspace()
    db_path = workspace / f"{fingerprint}.sqlite3"
    metadata = {"name": file_name, "fingerprint": fingerprint, "rows": None}
    cached_metadata = st.session_state.get("connected_metadata", {})
    if cached_metadata.get("fingerprint") == fingerprint:
        metadata = cached_metadata

    if not db_path.exists():
        if suffix == ".csv":
            try:
                frame = pd.read_csv(io.BytesIO(raw))
            except Exception as error:
                raise ValueError(f"Could not read this CSV: {error}") from error
            if frame.empty or len(frame.columns) == 0:
                raise ValueError("The CSV has no data rows or columns.")
            frame.columns = _unique_columns(frame.columns)
            table_name = _table_name_from_filename(file_name)
            try:
                with sqlite3.connect(str(db_path)) as connection:
                    frame.to_sql(table_name, connection, index=False, if_exists="replace")
            except Exception as error:
                db_path.unlink(missing_ok=True)
                raise ValueError(f"Could not prepare the CSV for analysis: {error}") from error
            metadata.update({"rows": len(frame), "table": table_name})
        else:
            if not raw.startswith(b"SQLite format 3\x00"):
                raise ValueError("This file does not look like a valid SQLite database.")
            db_path.write_bytes(raw)
            try:
                with connect_readonly(db_path) as connection:
                    tables = connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                if not tables:
                    raise ValueError("This SQLite file has no tables or views to query.")
            except Exception as error:
                db_path.unlink(missing_ok=True)
                if isinstance(error, ValueError):
                    raise
                raise ValueError(f"Could not open this SQLite file: {error}") from error

    previous_path = st.session_state.get("connected_db_path")
    if previous_path and Path(previous_path) != db_path:
        Path(previous_path).unlink(missing_ok=True)

    st.session_state["connected_db_path"] = str(db_path)
    st.session_state["connected_file_name"] = file_name
    st.session_state["connected_metadata"] = metadata
    return db_path, metadata


def list_tables(db_path: Path) -> list[str]:
    with connect_readonly(db_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def load_preview(db_path: Path, table_name: str, limit: int = 5000) -> pd.DataFrame:
    safe_name = '"' + table_name.replace('"', '""') + '"'
    with connect_readonly(db_path) as connection:
        return pd.read_sql_query(f"SELECT * FROM {safe_name} LIMIT {int(limit)}", connection)


@st.cache_data(show_spinner=False)
def load_demo_dashboard_data(db_path: str) -> pd.DataFrame:
    """Join the demo's related tables so its dashboard can analyze the full workflow."""
    query = """
        SELECT
            o.order_id,
            o.customer_id,
            o.order_date,
            o.status,
            o.channel,
            c.city,
            c.state,
            c.segment,
            p.product_id,
            p.name AS product,
            p.category,
            oi.quantity,
            oi.discount_pct,
            p.price,
            p.cost,
            CASE WHEN o.status = 'delivered'
                THEN oi.quantity * p.price * (1 - oi.discount_pct / 100.0) ELSE 0 END AS revenue,
            CASE WHEN o.status = 'delivered'
                THEN oi.quantity * (p.price * (1 - oi.discount_pct / 100.0) - p.cost) ELSE 0 END AS gross_profit
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p ON p.product_id = oi.product_id
    """
    with connect_readonly(db_path) as connection:
        frame = pd.read_sql_query(query, connection)
    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    return frame


def queue_prompt(prompt: str) -> None:
    st.session_state["pending_question"] = prompt


def start_new_chat() -> None:
    """Clear the current conversation while keeping its selected data source."""
    st.session_state["conversation"] = []
    st.session_state["conversation_id"] = uuid.uuid4().hex
    st.session_state["workspace_view"] = "Ask in chat"
    st.session_state["latest_result"] = None
    st.session_state["last_sample_prompt"] = None
    st.session_state["sample_prompt"] = None
    st.session_state.pop("pending_question", None)
    st.session_state.pop("chat_input", None)


def render_dashboard(frame: pd.DataFrame, key_prefix: str) -> None:
    """Create an automatic overview and let the user edit its chart layout."""
    if frame.empty:
        st.info("This table has no rows to visualize.")
        return

    def classify_fields(data: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
        numeric_fields = [column for column in data if pd.api.types.is_numeric_dtype(data[column])]
        id_fields = {
            column for column in data
            if re.search(r"(^|[ _-])(id|key|index|row number|record number|order number)$", str(column), flags=re.I)
            or re.search(r"(^|[ _-])(order|customer|product|row|user|account|transaction|invoice)[ _-]+(id|key|number)$", str(column), flags=re.I)
            or re.search(r"zip|postal|phone|uuid|guid", str(column), flags=re.I)
        }
        date_fields = [column for column in data if is_time_column(data, column)]
        measure_fields = [column for column in numeric_fields if column not in id_fields]
        dimension_fields = []
        for column in data:
            if column in id_fields or column in date_fields or column in measure_fields:
                continue
            distinct = data[column].nunique(dropna=True)
            name = str(column).casefold()
            descriptive = any(term in name for term in ("name", "product", "customer", "city", "state", "country"))
            if distinct > 1 and (distinct <= min(2000, max(80, int(len(data) * 0.5))) or (descriptive and distinct <= 5000)):
                dimension_fields.append(column)
        priority = ("category", "subcategory", "product name", "product", "region", "segment", "channel", "ship mode", "country", "state", "city", "department", "status")
        dimension_fields.sort(key=lambda column: (next((i for i, term in enumerate(priority) if term in str(column).casefold()), len(priority)), str(column).casefold()))
        measure_priority = ("sales", "revenue", "amount", "profit", "quantity", "units", "cost", "discount", "margin", "price")
        measure_fields.sort(key=lambda column: (next((i for i, term in enumerate(measure_priority) if term in str(column).casefold()), len(measure_priority)), str(column).casefold()))
        return numeric_fields, dimension_fields, measure_fields, date_fields

    numeric, categorical, business_numeric, date_fields = classify_fields(frame)
    id_fields = {
        column for column in frame
        if re.search(r"(^|[ _-])(id|key|index|row number|record number|order number)$", str(column), flags=re.I)
        or re.search(r"(^|[ _-])(order|customer|product|row|user|account|transaction|invoice)[ _-]+(id|key|number)$", str(column), flags=re.I)
        or re.search(r"zip|postal|phone|uuid|guid", str(column), flags=re.I)
    }

    title_key = f"{key_prefix}_title"
    st.text_input("Dashboard title", value="Dataset overview", key=title_key)
    st.subheader(st.session_state.get(title_key, "Dataset overview"))
    st.caption("Overview and charts use up to 5,000 rows from the selected table.")

    filter_fields = [column for column in frame.columns if column in categorical or column in date_fields]
    if filter_fields:
        filter_field = st.selectbox("Filter this dashboard", ["No filter", *filter_fields], key=f"{key_prefix}_filter_field")
        if filter_field != "No filter":
            filter_values = sorted(frame[filter_field].dropna().astype(str).unique().tolist())[:200]
            selected_values = st.multiselect(
                "Keep selected values (leave empty for all)",
                filter_values,
                key=f"{key_prefix}_filter_values",
            )
            if selected_values:
                frame = frame[frame[filter_field].astype(str).isin(selected_values)].copy()
                numeric, categorical, business_numeric, date_fields = classify_fields(frame)
                id_fields = {
                    column for column in frame
                    if re.search(r"(^|[ _-])(id|key|index|row number|record number|order number)$", str(column), flags=re.I)
                    or re.search(r"(^|[ _-])(order|customer|product|row|user|account|transaction|invoice)[ _-]+(id|key|number)$", str(column), flags=re.I)
                    or re.search(r"zip|postal|phone|uuid|guid", str(column), flags=re.I)
                }
                if frame.empty:
                    st.info("No rows match these filters.")
                    return

    kpi_specs_key = f"{key_prefix}_kpi_specs"
    order_field = next((column for column in frame if "order" in str(column).casefold() and re.search(r"id|number|key", str(column), re.I)), None)
    customer_field = next((column for column in frame if "customer" in str(column).casefold() and re.search(r"id|key", str(column), re.I)), None)
    sales_field = next((column for column in business_numeric if re.search(r"sales|revenue|amount", str(column), re.I)), None)
    profit_field = next((column for column in business_numeric if "profit" in str(column).casefold()), None)
    quantity_field = next((column for column in business_numeric if re.search(r"quantity|units", str(column), re.I)), None)
    kpi_choices = [*map(str, frame.columns), "Record count"]
    if sales_field and profit_field:
        kpi_choices.append("Profit margin (%)")
    if sales_field and order_field:
        kpi_choices.append("Average order value")
    if date_fields:
        kpi_choices.append("Date coverage")
    kpi_choices.extend(["Missing cells", "Fields", "Complete rows"])

    def initial_kpis() -> list[dict[str, str]]:
        definitions: list[dict[str, str]] = []
        if sales_field:
            definitions.append({"title": f"Total {sales_field}", "field": str(sales_field), "aggregation": "Sum"})
        elif business_numeric:
            field = quantity_field or business_numeric[0]
            definitions.append({"title": f"Total {field}", "field": str(field), "aggregation": "Sum"})
        if profit_field and profit_field != sales_field:
            definitions.append({"title": f"Total {profit_field}", "field": str(profit_field), "aggregation": "Sum"})
        if sales_field and profit_field:
            definitions.append({"title": "Profit margin", "field": "Profit margin (%)", "aggregation": "Formula"})
        if order_field:
            definitions.append({"title": "Unique orders", "field": str(order_field), "aggregation": "Unique count"})
        if customer_field:
            definitions.append({"title": "Unique customers", "field": str(customer_field), "aggregation": "Unique count"})
        if sales_field and order_field:
            definitions.append({"title": "Average order value", "field": "Average order value", "aggregation": "Formula"})
        if not definitions and business_numeric:
            field = business_numeric[0]
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            definitions.extend([
                {"title": f"Average {field}", "field": str(field), "aggregation": "Average"},
                {"title": f"Median {field}", "field": str(field), "aggregation": "Median"},
            ])
        if date_fields:
            definitions.append({"title": "Date coverage", "field": "Date coverage", "aggregation": "Formula"})
        if quantity_field and all(kpi["field"] != str(quantity_field) for kpi in definitions):
            definitions.append({"title": f"Total {quantity_field}", "field": str(quantity_field), "aggregation": "Sum"})
        if not definitions:
            definitions = [
                {"title": "Records analyzed", "field": "Record count", "aggregation": "Count"},
                {"title": "Missing cells", "field": "Missing cells", "aggregation": "Formula"},
            ]
        if business_numeric:
            focus_measure = sales_field or quantity_field or business_numeric[0]
            for aggregation in ("Average", "Median"):
                title = f"{aggregation} {focus_measure}"
                if len(definitions) < 6 and all(kpi["title"] != title for kpi in definitions):
                    definitions.append({"title": title, "field": str(focus_measure), "aggregation": aggregation})
        if categorical and len(definitions) < 6:
            focus_dimension = categorical[0]
            definitions.append({"title": f"Distinct {focus_dimension}", "field": str(focus_dimension), "aggregation": "Unique count"})
        for title, field, aggregation in (
            ("Records analyzed", "Record count", "Count"),
            ("Missing cells", "Missing cells", "Formula"),
            ("Complete rows", "Complete rows", "Formula"),
            ("Fields", "Fields", "Formula"),
        ):
            if len(definitions) >= 5:
                break
            if all(kpi["title"] != title for kpi in definitions):
                definitions.append({"title": title, "field": field, "aggregation": aggregation})
        return definitions[:6]

    st.session_state.setdefault(kpi_specs_key, initial_kpis())
    with st.expander("KPI cards · customize", icon=":material/monitoring:"):
        with st.container(horizontal=True):
            if st.button("Add KPI", icon=":material/add:", key=f"{key_prefix}_add_kpi", disabled=len(st.session_state[kpi_specs_key]) >= 8):
                st.session_state[kpi_specs_key].append({"title": "New KPI", "field": kpi_choices[0] if kpi_choices else "Record count", "aggregation": "Count"})
            if st.button("Remove last KPI", icon=":material/remove:", key=f"{key_prefix}_remove_kpi", disabled=len(st.session_state[kpi_specs_key]) <= 1):
                st.session_state[kpi_specs_key].pop()
        for index, spec in enumerate(st.session_state[kpi_specs_key]):
            kpi_id = f"{key_prefix}_kpi_{index}"
            field_value = spec["field"] if spec["field"] in kpi_choices else (kpi_choices[0] if kpi_choices else "Record count")
            cols = st.columns([2, 2, 1], vertical_alignment="bottom")
            spec["title"] = cols[0].text_input("KPI title", value=spec.get("title", f"KPI {index + 1}"), key=f"{kpi_id}_title")
            field_key = f"{kpi_id}_field"
            spec["field"] = cols[1].selectbox("Value", kpi_choices, index=kpi_choices.index(field_value), key=field_key)
            active_field = st.session_state.get(field_key, field_value)
            if active_field in {"Profit margin (%)", "Average order value", "Date coverage", "Missing cells", "Fields", "Complete rows"}:
                aggregation_options = ["Formula"]
            elif active_field == "Record count":
                aggregation_options = ["Count"]
            elif active_field in id_fields:
                aggregation_options = ["Unique count", "Count"]
            elif active_field in business_numeric:
                aggregation_options = ["Sum", "Average", "Median", "Minimum", "Maximum", "Count"]
            else:
                aggregation_options = ["Unique count", "Count"]
            chosen_aggregation = spec.get("aggregation", aggregation_options[0])
            spec["aggregation"] = cols[2].selectbox("Calculation", aggregation_options, index=aggregation_options.index(chosen_aggregation) if chosen_aggregation in aggregation_options else 0, key=f"{kpi_id}_aggregation")

    def kpi_value(spec: dict[str, str]) -> str:
        field, aggregation = spec["field"], spec["aggregation"]
        if field == "Record count":
            return f"{len(frame):,}"
        if field == "Missing cells":
            return f"{int(frame.isna().sum().sum()):,}"
        if field == "Fields":
            return f"{len(frame.columns):,}"
        if field == "Complete rows":
            return f"{int(frame.notna().all(axis=1).sum()):,}"
        if field == "Profit margin (%)" and sales_field and profit_field:
            sales_total = pd.to_numeric(frame[sales_field], errors="coerce").sum()
            profit_total = pd.to_numeric(frame[profit_field], errors="coerce").sum()
            return f"{profit_total / sales_total:.1%}" if sales_total else "—"
        if field == "Average order value" and sales_field and order_field:
            orders = frame[order_field].nunique(dropna=True)
            return f"{pd.to_numeric(frame[sales_field], errors='coerce').sum() / orders:,.2f}" if orders else "—"
        if field == "Date coverage" and date_fields:
            dates = pd.to_datetime(frame[date_fields[0]], errors="coerce").dropna()
            return f"{dates.min():%b %Y} – {dates.max():%b %Y}" if not dates.empty else "—"
        values = pd.to_numeric(frame[field], errors="coerce").dropna() if field in frame.columns and field in business_numeric else None
        if aggregation == "Unique count" and field in frame.columns:
            value = frame[field].nunique(dropna=True)
        elif aggregation == "Count":
            value = int(frame[field].notna().sum()) if field in frame.columns else len(frame)
        elif values is not None and not values.empty:
            value = {"Sum": values.sum(), "Average": values.mean(), "Median": values.median(), "Minimum": values.min(), "Maximum": values.max()}.get(aggregation, values.sum())
        else:
            value = 0
        return f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)

    with st.container(horizontal=True):
        for spec in st.session_state[kpi_specs_key]:
            st.metric(spec["title"] or "KPI", kpi_value(spec), border=True)

    st.subheader("Edit your dashboard")
    st.caption("Charts are suggested from this table. Change their titles, fields, and types below.")
    count_key = f"{key_prefix}_chart_count"
    st.session_state.setdefault(count_key, 8)
    controls = st.container(horizontal=True, vertical_alignment="center")
    with controls:
        if st.button("Add chart", icon=":material/add:", key=f"{key_prefix}_add_chart", disabled=st.session_state[count_key] >= 12):
            st.session_state[count_key] += 1
        if st.button("Remove last chart", icon=":material/remove:", key=f"{key_prefix}_remove_chart", disabled=st.session_state[count_key] <= 1):
            st.session_state[count_key] -= 1
        if st.button("Reset layout", icon=":material/refresh:", key=f"{key_prefix}_reset_layout"):
            st.session_state[count_key] = 8

    x_options = [*date_fields, *[column for column in categorical if column not in id_fields]]
    if not x_options:
        x_options = [column for column in frame.columns if column not in id_fields] or list(frame.columns)
    default_dimensions = list(x_options)
    default_chart_types = ["Line", "Bar", "Bar", "Pie", "Area", "Bar", "Line", "Bar"]
    export_charts: list[dict] = []
    chart_layout_specs: list[dict[str, str | None]] = []
    builder_hover_rules = []
    for index in range(st.session_state[count_key]):
        panel = f"{key_prefix}_chart_{index + 1}"
        builder_hover_rules.append(
            f".st-key-{panel}_card .st-key-{panel}_chart_options{{opacity:0;max-height:0;overflow:hidden;pointer-events:none;transition:opacity .16s ease,max-height .16s ease}}"
            f".st-key-{panel}_card:hover .st-key-{panel}_chart_options,.st-key-{panel}_card:focus-within .st-key-{panel}_chart_options{{opacity:1;max-height:5rem;overflow:visible;pointer-events:auto}}"
        )
    st.html("<style>" + "".join(builder_hover_rules) + "</style>")

    @st.dialog("Expanded chart", width="large")
    def show_builder_chart(chart_frame: pd.DataFrame, title: str, chart_type: str, x_field: str,
                           y_field: str, color_field: str | None, aggregation: str,
                           sort_order: str) -> None:
        st.subheader(title)
        build_chart(chart_frame, chart_type, x_field, y_field, color_field,
                    aggregation, min(100, len(chart_frame)), False,
                    sort_order=sort_order, height=650)

    with st.grid(2, min_column_width=400, gap="medium"):
        for index in range(st.session_state[count_key]):
            panel_key = f"{key_prefix}_chart_{index + 1}"
            with st.container(border=True, key=f"{panel_key}_card"):
                default_x = default_dimensions[index % len(default_dimensions)]
                x_field = st.selectbox(
                    "Dimension", x_options,
                    index=x_options.index(default_x),
                    key=f"{panel_key}_x",
                )
                y_options = [column for column in business_numeric if column != x_field] or ["Record count"]
                default_y = y_options[index % len(y_options)]
                y_field = st.selectbox("Measure", y_options, index=y_options.index(default_y), key=f"{panel_key}_y")
                suggested = "Line" if is_time_column(frame, x_field) else default_chart_types[index % len(default_chart_types)]
                cardinality = frame[x_field].nunique(dropna=True)
                available_chart_types = ["Bar", "Data table"]
                if is_time_column(frame, x_field):
                    available_chart_types.extend(["Line", "Area"])
                elif cardinality <= 8 and (y_field == "Record count" or pd.to_numeric(frame[y_field], errors="coerce").dropna().ge(0).all()):
                    available_chart_types.append("Pie")
                elif cardinality <= 60 and not is_time_column(frame, x_field):
                    available_chart_types.append("Line")
                suggested = suggested if suggested in available_chart_types else "Bar"
                with st.container(key=f"{panel_key}_chart_options"):
                    option_cols = st.columns([2, 2, 1], vertical_alignment="bottom")
                    chart_type = option_cols[0].selectbox(
                        "Chart type", available_chart_types,
                        index=available_chart_types.index(suggested),
                        key=f"{panel_key}_type_{hash(tuple(available_chart_types))}",
                        label_visibility="collapsed",
                    )
                    sort_order = "Highest first"
                    if chart_type in {"Bar", "Pie"}:
                        sort_order = option_cols[1].selectbox(
                            "Sort", ["Highest first", "Lowest first", "A to Z"],
                            key=f"{panel_key}_sort", label_visibility="collapsed",
                        )
                    expand_chart = option_cols[2].button(
                        "Expand", icon=":material/open_in_full:", width="stretch", key=f"{panel_key}_expand"
                    )
                title_default = f"{y_field if y_field != 'Record count' else 'Records'} by {x_field}"
                st.text_input("Chart title", value=title_default, key=f"{panel_key}_title")
                color_options = ["None", *[column for column in categorical if column != x_field]]
                color_choice = st.selectbox("Color by", color_options, key=f"{panel_key}_color")
                aggregation_options = ["Count"] if y_field == "Record count" else ["Sum", "Average", "Count", "No aggregation"]
                aggregation = st.selectbox("Aggregation", aggregation_options, key=f"{panel_key}_aggregation")
                title = st.session_state.get(f"{panel_key}_title", f"Insight {index + 1}")
                chart_layout_specs.append({
                    "title": title,
                    "dimension": str(x_field),
                    "measure": str(y_field),
                    "type": chart_type,
                    "color_by": None if color_choice == "None" else str(color_choice),
                    "aggregation": aggregation,
                })
                st.markdown(f"**{title}**")
                insight = None
                grouped_for_export = None
                if x_field in frame.columns and aggregation in {"Sum", "Average", "Count"}:
                    try:
                        if y_field == "Record count" or aggregation == "Count":
                            grouped = frame.groupby(x_field, dropna=False).size().rename("value").reset_index()
                        else:
                            values = pd.to_numeric(frame[y_field], errors="coerce")
                            method = "sum" if aggregation == "Sum" else "mean"
                            grouped = frame.assign(_insight_value=values).groupby(x_field, dropna=False)["_insight_value"].agg(method).rename("value").reset_index()
                        grouped = grouped.dropna(subset=["value"])
                        grouped_for_export = grouped
                        if len(grouped) > 1:
                            if is_time_column(frame, x_field):
                                grouped = grouped.sort_values(x_field)
                                previous, latest = float(grouped.iloc[-2]["value"]), float(grouped.iloc[-1]["value"])
                                if previous:
                                    change = (latest - previous) / abs(previous) * 100
                                    direction = "up" if change > 0 else "down" if change < 0 else "flat"
                                    insight = f"Latest {x_field.lower()} is {direction} {abs(change):.1f}% vs. the previous period."
                            else:
                                leader = grouped.nlargest(1, "value").iloc[0]
                                total = float(grouped["value"].sum())
                                share = float(leader["value"]) / total * 100 if total and aggregation in {"Sum", "Count"} else None
                                insight = f"Top {x_field.lower()}: {leader[x_field]} · {leader['value']:,.2f}"
                                if share is not None:
                                    insight += f" ({share:.1f}% of total)"
                    except (KeyError, TypeError, ValueError):
                        LOGGER.exception("Could not compute a custom dashboard insight")
                if insight:
                    st.caption(f"Insight · {insight}")
                if grouped_for_export is not None:
                    if is_time_column(frame, x_field):
                        export_frame = grouped_for_export.sort_values(x_field).head(12)
                    elif sort_order == "Lowest first":
                        export_frame = grouped_for_export.sort_values("value", ascending=True).head(12)
                    elif sort_order == "A to Z":
                        export_frame = grouped_for_export.sort_values(x_field, key=lambda values: values.astype(str).str.casefold()).head(12)
                    else:
                        export_frame = grouped_for_export.sort_values("value", ascending=False).head(12)
                    export_points = [(str(row[x_field]), float(row["value"])) for _, row in export_frame.iterrows()]
                elif y_field == "Record count":
                    counts = frame[x_field].fillna("(blank)").astype(str).value_counts().head(12)
                    export_points = [(str(label), float(value)) for label, value in counts.items()]
                elif y_field in frame.columns and pd.api.types.is_numeric_dtype(frame[y_field]):
                    export_points = [(str(row[x_field]), float(row[y_field])) for _, row in frame[[x_field, y_field]].dropna().head(12).iterrows()]
                elif y_field in frame.columns:
                    counts = frame[y_field].fillna("(blank)").astype(str).value_counts().head(12)
                    export_points = [(str(label), float(value)) for label, value in counts.items()]
                else:
                    export_points = []
                export_charts.append({"title": title, "dimension": str(x_field), "measure": str(y_field), "type": chart_type, "points": export_points})
                try:
                    build_chart(
                        frame, chart_type, x_field, y_field,
                        None if color_choice == "None" else color_choice,
                        aggregation, min(50, len(frame)), False, sort_order=sort_order,
                    )
                except Exception:
                    LOGGER.exception("Dashboard chart rendering failed")
                    st.info("This chart is unavailable. Try another field or chart type.")
                if expand_chart:
                    show_builder_chart(frame, title, chart_type, x_field, y_field,
                                       None if color_choice == "None" else color_choice,
                                       aggregation, sort_order)

    export_metrics = [(spec["title"] or "KPI", kpi_value(spec)) for spec in st.session_state[kpi_specs_key]]
    _render_dashboard_exports(key_prefix, st.session_state.get(title_key, "Dataset overview"), export_metrics, export_charts)

    save_key = f"{key_prefix}_saved"
    if st.button("Save dashboard layout", type="primary", icon=":material/save:", key=f"{key_prefix}_save"):
        st.session_state[save_key] = {
            "title": st.session_state.get(title_key, "Dataset overview"),
            "chart_count": st.session_state[count_key],
            "charts": chart_layout_specs,
            "kpis": [
                {
                    "title": spec["title"],
                    "field": spec["field"],
                    "aggregation": spec["aggregation"],
                    "value": kpi_value(spec),
                }
                for spec in st.session_state[kpi_specs_key]
            ],
        }
        st.toast("Dashboard layout saved for this session.", icon=":material/check:")
    saved_layout = st.session_state.get(save_key)
    if saved_layout:
        st.caption(f"Saved this session · {saved_layout['chart_count']} charts")
        st.download_button(
            "Download dashboard setup",
            data=json.dumps(saved_layout, indent=2).encode("utf-8"),
            file_name="ask-your-data-dashboard.json",
            mime="application/json",
            key=f"{key_prefix}_download_layout",
        )


def is_time_column(frame: pd.DataFrame, column: str) -> bool:
    values = frame[column].dropna()
    if values.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(values):
        return True
    name_hint = re.search(r"date|time|month|year|week", str(column), flags=re.I)
    if pd.api.types.is_numeric_dtype(values):
        return False
    sample = values.astype(str).head(100)
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(name_hint and parsed.notna().mean() >= 0.5) or parsed.notna().mean() >= 0.85


def recommended_chart(frame: pd.DataFrame) -> str:
    if len(frame) <= 1:
        return "Data table"
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]
    if numeric and categorical:
        return "Line" if is_time_column(frame, categorical[0]) else "Bar"
    if len(numeric) > 1:
        return "Scatter"
    if numeric:
        return "Histogram"
    if categorical:
        return "Bar"
    return "Data table"


def build_chart(frame: pd.DataFrame, chart_type: str, x_field: str, y_field: str,
                color_field: str | None, aggregation: str, top_n: int,
                show_labels: bool, sort_order: str = "Highest first", height: int = 380):
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    if chart_type == "Data table":
        st.dataframe(frame, hide_index=True, height=height, alt="Query results")
        return
    if chart_type == "Histogram":
        if x_field not in numeric:
            st.info("Choose a numeric column for the histogram.")
            return
        chart = (
            alt.Chart(frame)
            .mark_bar(color="#50D6C5", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(field=x_field, type="quantitative", bin=alt.Bin(maxbins=30), title=x_field),
                y=alt.Y(aggregate="count", type="quantitative", title="Records"),
                tooltip=[alt.Tooltip(field=x_field, type="quantitative", bin=alt.Bin(maxbins=30)),
                         alt.Tooltip(aggregate="count", type="quantitative", title="Records")],
            )
            .properties(height=height, title=f"Distribution of {x_field}")
        )
        st.altair_chart(chart, width="stretch", alt=f"Distribution of {x_field}")
        return

    if chart_type == "Scatter":
        if x_field not in numeric or y_field not in numeric or x_field == y_field:
            st.info("Choose two different numeric columns for a scatter plot.")
            return
        encodings = {
            "x": alt.X(field=x_field, type="quantitative", title=x_field),
            "y": alt.Y(field=y_field, type="quantitative", title=y_field),
            "tooltip": [alt.Tooltip(field=x_field, type="quantitative"),
                        alt.Tooltip(field=y_field, type="quantitative")],
        }
        if color_field:
            encodings["color"] = alt.Color(field=color_field, type="nominal", title=color_field)
            encodings["tooltip"].append(alt.Tooltip(field=color_field, type="nominal"))
        chart = (
            alt.Chart(frame)
            .mark_circle(size=90, opacity=0.76)
            .encode(**encodings)
            .properties(height=height, title=f"{y_field} vs. {x_field}")
        )
        st.altair_chart(chart, width="stretch", alt=f"{y_field} versus {x_field}")
        return

    if aggregation != "Count" and y_field not in numeric:
        st.info("Choose a numeric measure for this chart.")
        return

    chart_frame = frame.copy()
    temporal_x = is_time_column(chart_frame, x_field)
    if temporal_x and not pd.api.types.is_datetime64_any_dtype(chart_frame[x_field]):
        chart_frame[x_field] = pd.to_datetime(chart_frame[x_field], errors="coerce")

    group_fields = [x_field]
    if color_field and color_field != x_field:
        group_fields.append(color_field)
    value_field = y_field

    if aggregation != "No aggregation" and chart_type in {"Bar", "Line", "Area", "Pie"}:
        if aggregation == "Count":
            chart_frame = chart_frame.groupby(group_fields, dropna=False).size().reset_index(name="Records")
            value_field = "Records"
        else:
            method = "sum" if aggregation == "Sum" else "mean"
            chart_frame = (
                chart_frame.groupby(group_fields, dropna=False)[y_field]
                .agg(method)
                .reset_index()
            )

    if chart_type in {"Bar", "Pie"}:
        if sort_order == "Lowest first":
            chart_frame = chart_frame.sort_values(value_field, ascending=True).head(top_n)
        elif sort_order == "A to Z":
            chart_frame = chart_frame.sort_values(x_field, key=lambda values: values.astype(str).str.casefold()).head(top_n)
        else:
            chart_frame = chart_frame.sort_values(value_field, ascending=False).head(top_n)

    x_type = "temporal" if temporal_x else (
        "quantitative" if pd.api.types.is_numeric_dtype(chart_frame[x_field]) else "nominal"
    )
    tooltip = [
        alt.Tooltip(field=x_field, type=x_type, title=x_field),
        alt.Tooltip(field=value_field, type="quantitative", title=value_field, format=",.2f"),
    ]
    if color_field and color_field != x_field:
        tooltip.append(alt.Tooltip(field=color_field, type="nominal", title=color_field))

    if chart_type == "Pie":
        chart = (
            alt.Chart(chart_frame)
            .mark_arc(innerRadius=72)
            .encode(
                theta=alt.Theta(field=value_field, type="quantitative", title=value_field),
                color=alt.Color(field=x_field, type="nominal", title=x_field),
                tooltip=tooltip,
            )
            .properties(height=height, title=f"{value_field} by {x_field}")
        )
    else:
        mark = {"Bar": "bar", "Line": "line", "Area": "area"}[chart_type]
        chart = alt.Chart(chart_frame).mark_bar() if mark == "bar" else (
            alt.Chart(chart_frame).mark_line(point=True) if mark == "line"
            else alt.Chart(chart_frame).mark_area(opacity=0.7)
        )
        x_sort = None
        if chart_type == "Bar" and x_type == "nominal":
            x_sort = "ascending" if sort_order == "A to Z" else alt.EncodingSortField(
                field=value_field,
                order="ascending" if sort_order == "Lowest first" else "descending",
            )
        encodings = {
            "x": alt.X(field=x_field, type=x_type, title=x_field, sort=x_sort),
            "y": alt.Y(field=value_field, type="quantitative", title=value_field),
            "tooltip": tooltip,
        }
        if color_field and color_field != x_field:
            encodings["color"] = alt.Color(field=color_field, type="nominal", title=color_field)
        chart = chart.encode(**encodings).properties(height=height, title=f"{value_field} by {x_field}")
        if show_labels and chart_type == "Bar":
            labels = alt.Chart(chart_frame).mark_text(dy=-8, fontSize=11).encode(
                x=alt.X(field=x_field, type=x_type),
                y=alt.Y(field=value_field, type="quantitative"),
                text=alt.Text(field=value_field, type="quantitative", format="~s"),
            )
            chart = chart + labels

    st.altair_chart(chart, width="stretch", alt=f"{chart_type} chart of {value_field} by {x_field}")


DASH_PALETTE = ["#4C83F3", "#F04D86", "#F5B934", "#0CC9AE", "#8A63E6", "#22B8CF"]


def _make_dashboard_exports(title: str, metrics: list[tuple[str, str]], charts: list[dict]) -> tuple[bytes, bytes, bytes]:
    """Create self-contained PNG, PDF, and HTML exports for the current dashboard."""
    import base64
    import html
    from PIL import Image, ImageDraw, ImageFont

    width = 1600
    card_width = 754
    card_height = 310
    chart_rows = max(1, (min(12, len(charts)) + 1) // 2)
    height = 180 + 132 + chart_rows * (card_height + 22) + 50
    image = Image.new("RGB", (width, height), "#0B1020")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def text(value: object, limit: int = 64) -> str:
        return str(value).encode("latin-1", "replace").decode("latin-1")[:limit]

    draw.text((48, 34), text(title, 90), fill="#F3F7FF", font=font)
    draw.text((48, 78), "Ask Your Data  |  Dashboard snapshot", fill="#95A5BD", font=font)
    metric_gap = 14
    metric_count = max(1, min(8, len(metrics)))
    metric_width = (width - 96 - metric_gap * (metric_count - 1)) // metric_count
    for index, (label, value) in enumerate(metrics[:8]):
        left = 48 + index * (metric_width + metric_gap)
        draw.rounded_rectangle((left, 120, left + metric_width, 230), radius=12, fill="#151D2C", outline="#2B3548", width=2)
        draw.text((left + 18, 142), text(label, 34), fill="#A8B7CD", font=font)
        draw.text((left + 18, 184), text(value, 32), fill="#F3F7FF", font=font)

    palette = ["#4C83F3", "#F04D86", "#F5B934", "#0CC9AE", "#8A63E6", "#22B8CF"]
    chart_top = 272
    for index, chart in enumerate(charts[:12]):
        col, row = index % 2, index // 2
        left, top = 48 + col * (card_width + 44), chart_top + row * (card_height + 22)
        right, bottom = left + card_width, top + card_height
        draw.rounded_rectangle((left, top, right, bottom), radius=12, fill="#111827", outline="#2B3548", width=2)
        draw.text((left + 20, top + 18), text(chart.get("title", "Chart"), 64), fill="#F3F7FF", font=font)
        points = chart.get("points", [])[:12]
        plot_left, plot_top = left + 190, top + 58
        plot_right, plot_bottom = right - 22, bottom - 38
        if not points:
            draw.text((left + 24, top + 90), "No chart values available", fill="#A8B7CD", font=font)
            continue
        max_value = max([abs(float(point[1])) for point in points] + [1.0])
        if chart.get("type") == "Pie":
            center = (left + 270, top + 178)
            radius = 92
            total = sum(max(0.0, float(point[1])) for point in points) or 1.0
            start = 0.0
            for pindex, point in enumerate(points):
                extent = max(0.2, float(point[1])) / total * 360
                draw.pieslice((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), start, start + extent, fill=palette[pindex % len(palette)])
                start += extent
            for pindex, point in enumerate(points[:7]):
                draw.ellipse((left + 410, top + 83 + pindex * 25, left + 423, top + 96 + pindex * 25), fill=palette[pindex % len(palette)])
                draw.text((left + 432, top + 83 + pindex * 25), text(point[0], 34), fill="#D9E3F2", font=font)
            continue
        if chart.get("type") in {"Line", "Area"} and len(points) > 1:
            coords = []
            lower_value = min(float(point[1]) for point in points)
            upper_value = max(float(point[1]) for point in points)
            value_range = max(1.0, upper_value - lower_value)
            for pindex, (_, value) in enumerate(points):
                x = plot_left + pindex * (plot_right - plot_left) / max(1, len(points) - 1)
                y = plot_bottom - (float(value) - lower_value) / value_range * (plot_bottom - plot_top)
                coords.append((int(x), int(y)))
            draw.line(coords, fill=palette[index % len(palette)], width=4)
            for x, y in coords:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=palette[index % len(palette)])
            for pindex, point in enumerate(points[:6]):
                draw.text((plot_left + pindex * max(1, (plot_right - plot_left) // 6), plot_bottom + 8), text(point[0], 12), fill="#A8B7CD", font=font)
        else:
            bar_height = min(22, max(10, (plot_bottom - plot_top) // max(1, len(points))))
            row_gap = max(2, ((plot_bottom - plot_top) - bar_height * len(points)) // max(1, len(points)))
            negative_limit = min([float(point[1]) for point in points] + [0.0])
            positive_limit = max([float(point[1]) for point in points] + [0.0])
            value_span = max(1.0, positive_limit - negative_limit)
            baseline_x = plot_left + int((plot_right - plot_left) * (-negative_limit) / value_span)
            if negative_limit < 0 < positive_limit:
                draw.line((baseline_x, plot_top - 2, baseline_x, plot_bottom), fill="#536176", width=1)
            for pindex, (label, value) in enumerate(points):
                y = plot_top + pindex * (bar_height + row_gap)
                draw.text((left + 20, y + 4), text(label, 25), fill="#D9E3F2", font=font)
                bar_end = plot_left + int((float(value) - negative_limit) / value_span * (plot_right - plot_left))
                bar_start, bar_end = sorted((baseline_x, bar_end))
                draw.rounded_rectangle((bar_start, y, max(bar_start + 2, bar_end), y + bar_height), radius=4, fill=palette[index % len(palette)])
                value_x = min(plot_right - 48, max(plot_left, bar_end + 8))
                draw.text((value_x, y + 4), text(f"{float(value):,.2f}", 20), fill="#D9E3F2", font=font)

    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG", optimize=True)
    pdf_buffer = io.BytesIO()
    image.save(pdf_buffer, format="PDF", resolution=120)
    png_bytes = png_buffer.getvalue()
    html_bytes = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>body{{margin:0;padding:28px;background:#0b1020;color:#eef3fb;font:16px system-ui}}"
        "main{max-width:1600px;margin:auto}img{width:100%;height:auto;border:1px solid #2b3548;border-radius:14px}h1{font-size:1.5rem}</style>"
        f"<main><h1>{html.escape(title)}</h1><img alt='Dashboard snapshot' src='data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}'></main></html>"
    ).encode("utf-8")
    return png_bytes, pdf_buffer.getvalue(), html_bytes


def _render_dashboard_exports(key_prefix: str, title: str, metrics: list[tuple[str, str]], charts: list[dict]) -> None:
    try:
        png_export, pdf_export, html_export = _make_dashboard_exports(title, metrics, charts)
        st.subheader("Export and share")
        with st.container(horizontal=True, wrap=True):
            st.download_button("Download PNG image", png_export, file_name="ask-your-data-dashboard.png", mime="image/png", icon=":material/image:", key=f"{key_prefix}_download_png")
            st.download_button("Download PDF", pdf_export, file_name="ask-your-data-dashboard.pdf", mime="application/pdf", icon=":material/picture_as_pdf:", key=f"{key_prefix}_download_pdf")
            st.download_button("Download HTML snapshot", html_export, file_name="ask-your-data-dashboard.html", mime="text/html", icon=":material/language:", key=f"{key_prefix}_download_html")
        try:
            current_app_url = st.context.url
        except Exception:
            current_app_url = None
        if current_app_url and str(current_app_url).startswith(("http://", "https://")):
            st.link_button("Open / share this app URL", current_app_url, icon=":material/open_in_new:")
        st.caption("The HTML file is a portable snapshot. Share the app URL for a live dashboard; deploy the app first if the current address is localhost.")
    except Exception:
        LOGGER.exception("Dashboard export generation failed")
        st.info("Dashboard exports are temporarily unavailable. The dashboard itself is still ready to use.")


def _dashboard_bar(frame: pd.DataFrame, dimension: str, measure: str, title: str, color: str):
    return (
        alt.Chart(frame)
        .mark_bar(color=color, cornerRadiusEnd=5)
        .encode(
            x=alt.X(f"{measure}:Q", title=measure, axis=alt.Axis(format=",.3~s")),
            y=alt.Y(
                f"{dimension}:N",
                title=dimension,
                sort=alt.EncodingSortField(field=measure, order="descending"),
            ),
            tooltip=[
                alt.Tooltip(f"{dimension}:N", title=dimension),
                alt.Tooltip(f"{measure}:Q", title=measure, format=",.2f"),
            ],
        )
        .properties(height=280)
    )


def _dashboard_line(frame: pd.DataFrame, date_field: str, measure: str, title: str):
    return (
        alt.Chart(frame)
        .mark_line(color=DASH_PALETTE[3], point=True, strokeWidth=3)
        .encode(
            x=alt.X(f"{date_field}:T", title=date_field, axis=alt.Axis(format="%b %Y", labelAngle=-30)),
            y=alt.Y(f"{measure}:Q", title=measure, axis=alt.Axis(format=",.3~s")),
            tooltip=[
                alt.Tooltip(f"{date_field}:T", title=date_field, format="%b %Y"),
                alt.Tooltip(f"{measure}:Q", title=measure, format=",.2f"),
            ],
        )
        .properties(height=280)
    )


def _dashboard_donut(frame: pd.DataFrame, dimension: str, measure: str, title: str):
    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=70, outerRadius=125, stroke="#111827", strokeWidth=2)
        .encode(
            theta=alt.Theta(f"{measure}:Q", title=measure),
            color=alt.Color(f"{dimension}:N", title=dimension),
            tooltip=[
                alt.Tooltip(f"{dimension}:N", title=dimension),
                alt.Tooltip(f"{measure}:Q", title=measure, format=",.2f"),
            ],
        )
        .properties(height=280)
    )


def _dashboard_grid(charts: list[tuple[str, alt.Chart]], key: str) -> None:
    with st.grid(2, min_column_width=420, gap="medium", key=key):
        for title, chart in charts:
            with st.container(border=True):
                st.subheader(title)
                st.altair_chart(chart, width="stretch", alt=title)


def render_sales_dashboard(db_path: str, key_prefix: str) -> None:
    """A finished eight-chart sales dashboard for the joined demo dataset."""
    source = load_demo_dashboard_data(db_path)
    if source.empty:
        st.info("The demo dataset has no rows to analyze.")
        return

    start_date = source["order_date"].min().date()
    end_date = source["order_date"].max().date()
    with st.popover("Filters", icon=":material/filter_list:"):
        selected_range = st.date_input(
            "Order date range",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date,
            key=f"{key_prefix}_dates",
        )
        channels = st.multiselect("Sales channels", sorted(source["channel"].dropna().unique()), key=f"{key_prefix}_channels")
        categories = st.multiselect("Product categories", sorted(source["category"].dropna().unique()), key=f"{key_prefix}_categories")

    date_start, date_end = selected_range if isinstance(selected_range, tuple) else (selected_range, selected_range)
    filtered = source.loc[
        source["order_date"].dt.date.between(date_start, date_end)
    ].copy()
    if channels:
        filtered = filtered[filtered["channel"].isin(channels)]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if filtered.empty:
        st.info("No orders match these filters. Adjust the filters to see the dashboard.")
        return

    delivered = filtered[filtered["status"] == "delivered"]
    order_status = filtered[["order_id", "channel", "status"]].drop_duplicates("order_id")
    order_count = int(order_status["order_id"].nunique())
    delivered_orders = int(order_status.loc[order_status["status"] == "delivered", "order_id"].nunique())
    returned_orders = int(order_status.loc[order_status["status"] == "returned", "order_id"].nunique())
    revenue = float(delivered["revenue"].sum())
    gross_profit = float(delivered["gross_profit"].sum())
    average_order_value = revenue / delivered_orders if delivered_orders else 0
    return_rate = returned_orders / order_count if order_count else 0
    active_customers = int(delivered["customer_id"].nunique())

    with st.grid(6, min_column_width=170, gap="small"):
        st.metric("Net sales", f"₹{revenue:,.0f}", border=True)
        st.metric("Gross profit", f"₹{gross_profit:,.0f}", border=True)
        st.metric("Delivered orders", f"{delivered_orders:,}", border=True)
        st.metric("Average order value", f"₹{average_order_value:,.0f}", border=True)
        st.metric("Return rate", f"{return_rate:.1%}", border=True)
        st.metric("Active customers", f"{active_customers:,}", border=True)

    st.caption(f"Fashion commerce demo · {date_start:%d %b %Y} — {date_end:%d %b %Y} · Revenue includes delivered orders only")

    monthly = (
        delivered.assign(month=delivered["order_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["revenue"].sum()
    )
    category_sales = delivered.groupby("category", as_index=False)["revenue"].sum()
    channel_sales = delivered.groupby("channel", as_index=False)["revenue"].sum()
    product_sales = delivered.groupby("product", as_index=False)["revenue"].sum().nlargest(10, "revenue")
    category_profit = delivered.groupby("category", as_index=False)["gross_profit"].sum()
    segment_sales = delivered.groupby("segment", as_index=False)["revenue"].sum()
    city_sales = delivered.groupby("city", as_index=False)["revenue"].sum().nlargest(10, "revenue")
    order_status = order_status.assign(is_returned=(order_status["status"] == "returned").astype(int))
    return_by_channel = (
        order_status.groupby("channel", as_index=False)
        .agg(orders=("order_id", "nunique"), returns=("is_returned", "sum"))
    )
    return_by_channel["return_rate"] = return_by_channel["returns"] / return_by_channel["orders"].clip(lower=1) * 100
    status_mix = order_status.groupby("status", as_index=False)["order_id"].nunique().rename(columns={"order_id": "orders"})
    discount_sales = (
        filtered.groupby("discount_pct", as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
    )
    discount_sales["discount"] = discount_sales["discount_pct"].map(lambda value: f"{value:g}%")

    insights = []
    if not category_sales.empty:
        leader = category_sales.nlargest(1, "revenue").iloc[0]
        insights.append(f"**Top category:** {leader['category']} · ₹{leader['revenue']:,.0f}")
    if not channel_sales.empty:
        leader = channel_sales.nlargest(1, "revenue").iloc[0]
        insights.append(f"**Top channel:** {leader['channel']} · ₹{leader['revenue']:,.0f}")
    if not monthly.empty:
        leader = monthly.nlargest(1, "revenue").iloc[0]
        insights.append(f"**Best month:** {leader['month']:%b %Y} · ₹{leader['revenue']:,.0f}")
    if insights:
        st.info("  ·  ".join(insights), icon=":material/auto_awesome:")

    charts = [
        ("Monthly sales trend", _dashboard_line(monthly, "month", "revenue", "Revenue by month")),
        ("Sales by product category", _dashboard_bar(category_sales, "category", "revenue", "Net sales by category", DASH_PALETTE[0])),
        ("Sales by channel", _dashboard_bar(channel_sales, "channel", "revenue", "Net sales by channel", DASH_PALETTE[1])),
        ("Top 10 products", _dashboard_bar(product_sales, "product", "revenue", "Top products by net sales", DASH_PALETTE[2])),
        ("Profit by category", _dashboard_bar(category_profit, "category", "gross_profit", "Gross profit by category", DASH_PALETTE[3])),
        ("Returns by channel", _dashboard_bar(return_by_channel, "channel", "return_rate", "Return rate by channel (%)", DASH_PALETTE[1])),
        ("Sales by customer segment", _dashboard_bar(segment_sales, "segment", "revenue", "Net sales by customer segment", DASH_PALETTE[4])),
        ("Top 10 cities", _dashboard_bar(city_sales, "city", "revenue", "Net sales by city", DASH_PALETTE[5])),
        ("Order status mix", _dashboard_donut(status_mix, "status", "orders", "Orders by status")),
        ("Discount and sales volume", _dashboard_bar(discount_sales, "discount", "quantity", "Units by discount level · descriptive only", DASH_PALETTE[2])),
    ]
    _dashboard_grid(charts, f"{key_prefix}_chart_grid")
    with st.expander("Metric definitions and data notes"):
        st.markdown(
            "- **Net sales:** delivered order items after discounts.\n"
            "- **Gross profit:** discounted sales less product cost, for delivered orders.\n"
            "- **Average order value:** net sales divided by delivered orders.\n"
            "- **Return rate:** returned orders divided by all orders in the selected range.\n"
            "- **Discount chart:** shows observed units by discount level; it does not prove discounts caused a sales change.\n"
            "- This synthetic demo contains no advertising spend, clicks, or campaign attribution data."
        )


def render_generic_dashboard(frame: pd.DataFrame, source_name: str, key_prefix: str) -> None:
    """Build an insight-led dashboard, skipping identifier-like fields by default."""
    if frame.empty:
        st.info("This table has no rows to analyze.")
        return

    def normalized(column: object) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(column).casefold()).strip()

    def is_identifier(column: object) -> bool:
        name = normalized(column)
        series = frame[column]
        id_name = bool(re.search(r"(^| )(id|key|index|row number|record number|number)$", name)) or bool(
            re.search(r"(^| )(order|customer|product|row|user|account|transaction|invoice|product) (id|key|number)$", name)
        )
        id_name |= bool(re.search(r"(^| )(zip|postal|phone|ssn|uuid|guid)$", name))
        descriptive_name = any(term in name for term in ("name", "product", "customer", "city", "state", "country", "region", "category", "segment"))
        high_cardinality = len(frame) > 20 and series.nunique(dropna=True) / max(1, series.notna().sum()) > 0.92
        return id_name or (high_cardinality and not descriptive_name and not pd.api.types.is_datetime64_any_dtype(series))

    def semantic_score(column: object, terms: tuple[str, ...]) -> int:
        name = normalized(column)
        return max((len(term) for term in terms if term in name), default=0)

    identifier_fields = {column for column in frame.columns if is_identifier(column)}
    numeric_fields = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    measure_fields = [column for column in numeric_fields if column not in identifier_fields]
    date_fields = [column for column in frame if is_time_column(frame, column)]
    dimension_fields = []
    for column in frame.columns:
        if column in identifier_fields or column in date_fields or column in measure_fields:
            continue
        distinct = frame[column].nunique(dropna=True)
        descriptive = any(term in normalized(column) for term in ("name", "product", "customer", "city", "state", "country"))
        distinct_limit = min(2000, max(80, int(len(frame) * 0.5)))
        if distinct > 1 and (distinct <= distinct_limit or (descriptive and distinct <= 5000)):
            dimension_fields.append(column)
    # Keep ordinal/low-cardinality numeric business dimensions such as discount or rating out of the measure list.
    dimension_fields += [
        column for column in measure_fields
        if frame[column].nunique(dropna=True) <= 12 and semantic_score(column, ("discount", "rating", "score", "year", "month"))
    ]
    measure_fields = [column for column in measure_fields if column not in dimension_fields]
    dimension_priority = ("category", "subcategory", "product name", "product", "region", "segment", "channel", "ship mode", "country", "state", "city", "department", "status", "customer type", "customer")
    dimension_fields.sort(key=lambda col: (next((i for i, term in enumerate(dimension_priority) if term in normalized(col)), len(dimension_priority)), str(col).casefold()) if col in dimension_fields else (999, str(col)))
    measure_priority = ("sales", "revenue", "amount", "profit", "quantity", "units", "orders", "cost", "discount", "margin", "price")
    measure_fields.sort(key=lambda col: (next((i for i, term in enumerate(measure_priority) if term in normalized(col)), len(measure_priority)), str(col).casefold()))
    date_fields.sort(key=lambda col: (0 if any(term in normalized(col) for term in ("order date", "transaction date", "created at", "date")) else 1, str(col)))

    primary_measure = next((c for c in measure_fields if any(t in normalized(c) for t in ("sales", "revenue", "amount"))), None)
    profit_measure = next((c for c in measure_fields if "profit" in normalized(c)), None)
    quantity_measure = next((c for c in measure_fields if any(t in normalized(c) for t in ("quantity", "units"))), None)
    if primary_measure is None:
        primary_measure = profit_measure or (measure_fields[0] if measure_fields else None)
    order_id = next((c for c in frame.columns if "order" in normalized(c) and ("id" in normalized(c) or "number" in normalized(c))), None)
    customer_id = next((c for c in frame.columns if "customer" in normalized(c) and ("id" in normalized(c) or "key" in normalized(c))), None)
    entity_count = int(frame[order_id].nunique(dropna=True)) if order_id else len(frame)
    total_primary = float(pd.to_numeric(frame[primary_measure], errors="coerce").sum()) if primary_measure else None
    total_profit = float(pd.to_numeric(frame[profit_measure], errors="coerce").sum()) if profit_measure else None
    average_order = total_primary / entity_count if order_id and total_primary is not None and entity_count else None
    margin = total_profit / total_primary * 100 if total_profit is not None and total_primary else None
    customers = int(frame[customer_id].nunique(dropna=True)) if customer_id else None
    time_range = None
    if date_fields:
        parsed_dates = pd.to_datetime(frame[date_fields[0]], errors="coerce").dropna()
        if not parsed_dates.empty:
            time_range = f"{parsed_dates.min():%b %Y} – {parsed_dates.max():%b %Y}"

    kpis: list[tuple[str, str]] = []
    if primary_measure and total_primary is not None:
        label = "Total sales" if "sale" in normalized(primary_measure) else f"Total {primary_measure}"
        kpis.append((label, f"{total_primary:,.0f}"))
    if profit_measure and total_profit is not None and profit_measure != primary_measure:
        kpis.append(("Total profit", f"{total_profit:,.0f}"))
    if margin is not None:
        kpis.append(("Profit margin", f"{margin:.1f}%"))
    if order_id:
        kpis.append(("Unique orders", f"{entity_count:,}"))
    elif quantity_measure:
        kpis.append(("Units sold", f"{pd.to_numeric(frame[quantity_measure], errors='coerce').sum():,.0f}"))
    if customers is not None:
        kpis.append(("Unique customers", f"{customers:,}"))
    if average_order is not None:
        kpis.append(("Sales per order", f"{average_order:,.2f}"))
    if primary_measure and not order_id:
        values = pd.to_numeric(frame[primary_measure], errors="coerce").dropna()
        if not values.empty:
            kpis.extend([
                (f"Average {primary_measure}", f"{values.mean():,.2f}"),
                (f"Median {primary_measure}", f"{values.median():,.2f}"),
            ])
        kpis.append(("Records analyzed", f"{len(frame):,}"))
        if dimension_fields:
            focus_dimension = dimension_fields[0]
            kpis.append((f"Distinct {focus_dimension}", f"{frame[focus_dimension].nunique(dropna=True):,}"))
    if time_range:
        kpis.append(("Date coverage", time_range))
    if not kpis:
        kpis = [("Records", f"{len(frame):,}"), ("Fields", f"{len(frame.columns):,}"),
                ("Complete rows", f"{int(frame.notna().all(axis=1).sum()):,}"),
                ("Distinct entities", f"{int(frame.nunique(dropna=True).max()):,}"),
                ("Missing cells", f"{int(frame.isna().sum().sum()):,}")]
    with st.container(horizontal=True):
        for label, value in kpis[:6]:
            st.metric(label, value, border=True)
    field_summary = ", ".join([*(str(c) for c in dimension_fields[:4]), *(str(c) for c in measure_fields[:3])])
    st.caption(f"{source_name} · automatic insights use {len(frame):,} rows · prioritized fields: {field_summary or 'record-level summary only'}")

    chart_specs: list[dict] = []
    used_dimensions: set[object] = set()
    monthly_trend: pd.DataFrame | None = None
    if date_fields:
        date_col = date_fields[0]
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        bucket = "Month"
        temporal = pd.DataFrame({"period": dates.dt.to_period("M").dt.to_timestamp()})
        if primary_measure:
            temporal["value"] = pd.to_numeric(frame[primary_measure], errors="coerce")
            trend = temporal.dropna().groupby("period", as_index=False)["value"].sum()
            trend_title = f"{primary_measure} trend by month"
            trend_type = "line"
        else:
            trend = temporal.dropna().groupby("period", as_index=False).size().rename(columns={"size": "value"})
            trend_title = "Records over time"
            trend_type = "line"
        if len(trend) > 1:
            monthly_trend = trend
            chart_specs.append({"title": trend_title, "data": trend, "x": "period", "y": "value", "kind": trend_type, "types": ["line", "bar", "area"], "sort": False, "color": DASH_PALETTE[3]})

    secondary_dimension_charts: list[dict] = []
    if primary_measure:
        for dimension in dimension_fields:
            if dimension in identifier_fields or dimension in date_fields:
                continue
            values_count = frame[dimension].nunique(dropna=True)
            if values_count < 2 or values_count > 80:
                continue
            aggregation = frame.groupby(dimension, dropna=False)[primary_measure].sum(min_count=1).rename("value").reset_index()
            aggregation[dimension] = aggregation[dimension].fillna("(blank)").astype(str)
            aggregation = aggregation.dropna(subset=["value"]).nlargest(12, "value")
            if len(aggregation) < 2:
                continue
            is_product = "product" in normalized(dimension) or "name" in normalized(dimension)
            chart_types = ["bar", "line"]
            if not is_product and len(aggregation) <= 8 and not aggregation["value"].lt(0).any():
                chart_types.append("pie")
            chart_title = f"Top products by {primary_measure}" if is_product else f"{primary_measure} by {dimension}"
            chart_specs.append({"title": chart_title, "data": aggregation, "x": dimension, "y": "value", "kind": "bar", "types": chart_types, "sort": True, "color": DASH_PALETTE[len(chart_specs) % len(DASH_PALETTE)]})
            used_dimensions.add(dimension)
            if profit_measure and profit_measure != primary_measure:
                profit_by_dimension = frame.groupby(dimension, dropna=False)[profit_measure].sum(min_count=1).rename("value").reset_index()
                profit_by_dimension[dimension] = profit_by_dimension[dimension].fillna("(blank)").astype(str)
                profit_by_dimension = profit_by_dimension.dropna(subset=["value"]).nlargest(12, "value")
                if len(profit_by_dimension) > 1:
                    profit_chart_types = ["bar", "line"]
                    if not is_product and len(profit_by_dimension) <= 8 and not profit_by_dimension["value"].lt(0).any():
                        profit_chart_types.append("pie")
                    secondary_dimension_charts.append({"title": f"{profit_measure} by {dimension}", "data": profit_by_dimension, "x": dimension, "y": "value", "kind": "bar", "types": profit_chart_types, "sort": True, "color": DASH_PALETTE[len(chart_specs) % len(DASH_PALETTE)]})
            if len(chart_specs) >= 8:
                break

    if primary_measure and len(chart_specs) < 8:
        relation_measure = profit_measure if profit_measure and profit_measure != primary_measure else quantity_measure
        scatter_data = frame[[primary_measure, relation_measure]].apply(pd.to_numeric, errors="coerce").dropna().head(2000) if relation_measure else pd.DataFrame()
        if len(scatter_data) >= 3:
            scatter_data = scatter_data.rename(columns={primary_measure: "x_value", relation_measure: "y_value"})
            chart_specs.append({"title": f"{primary_measure} vs {relation_measure}", "data": scatter_data, "x": "x_value", "y": "y_value", "kind": "scatter", "types": ["scatter"], "sort": False, "color": DASH_PALETTE[5]})

    for extra_chart in secondary_dimension_charts:
        if len(chart_specs) >= 8:
            break
        # Use a second metric only for one strong business dimension to avoid a wall of near-duplicate charts.
        if extra_chart["x"] == next(iter(used_dimensions), None):
            chart_specs.append(extra_chart)

    if not chart_specs:
        usable = next((c for c in dimension_fields if frame[c].nunique(dropna=True) > 1), None)
        if usable:
            frequency = frame[usable].fillna("(blank)").astype(str).value_counts().head(12).rename_axis(usable).reset_index(name="value")
            chart_specs.append({"title": f"Records by {usable}", "data": frequency, "x": usable, "y": "value", "kind": "bar", "types": ["bar", "pie"], "sort": True, "color": DASH_PALETTE[0]})

    if not chart_specs:
        st.info("This table has no chartable business measures or categories. Try another table, or ask a question in chat.")
        return

    insight_lines = []
    if primary_measure:
        for spec in chart_specs:
            dimension = spec["x"]
            if dimension not in frame.columns or spec["y"] != "value" or spec["data"].empty:
                continue
            leader = spec["data"].nlargest(1, "value").iloc[0]
            denominator = pd.to_numeric(frame[primary_measure], errors="coerce").sum()
            if spec["title"] == f"{primary_measure} by {dimension}" and denominator:
                share = float(leader["value"]) / float(denominator) * 100
                insight_lines.append(f"**Leading {dimension.lower()}:** {leader[dimension]} · {leader['value']:,.2f} ({share:.1f}% of {primary_measure.lower()}).")
                break
    if monthly_trend is not None and len(monthly_trend) >= 2:
        previous_value = float(monthly_trend.iloc[-2]["value"])
        latest_value = float(monthly_trend.iloc[-1]["value"])
        if previous_value:
            change = (latest_value - previous_value) / abs(previous_value) * 100
            direction = "up" if change > 0 else "down" if change < 0 else "flat"
            insight_lines.append(f"**Latest month:** {direction} {abs(change):.1f}% vs. the previous month ({latest_value:,.2f}).")
    if insight_lines:
        st.info("  ·  ".join(insight_lines), icon=":material/auto_awesome:")

    hover_rules = []
    for index in range(min(8, len(chart_specs))):
        card = f"{key_prefix}_smart_{index}_card"
        actions = f"{key_prefix}_smart_{index}_actions"
        hover_rules.append(
            f".st-key-{card} .st-key-{actions}{{opacity:0;max-height:0;overflow:hidden;pointer-events:none;transition:opacity .16s ease,max-height .16s ease}}"
            f".st-key-{card}:hover .st-key-{actions},.st-key-{card}:focus-within .st-key-{actions}{{opacity:1;max-height:5rem;overflow:visible;pointer-events:auto}}"
        )
    st.html("<style>" + "".join(hover_rules) + "</style>")

    def display_chart(spec: dict, chart_key: str, height: int = 330, allow_expand: bool = True) -> None:
        chart_data = spec["data"].copy().reset_index(drop=True)
        x_name, y_name = spec["x"], spec["y"]
        has_type_picker = len(spec["types"]) > 1
        has_sort = bool(spec["sort"])
        sort_options = ["Largest first", "Smallest first", "A to Z"]
        control_weights = [2 for _ in range(int(has_type_picker) + int(has_sort) + int(allow_expand))]
        with st.container(key=f"{chart_key}_actions"):
            control_cols = st.columns(control_weights, vertical_alignment="bottom") if control_weights else []
            control_index = 0
            chart_type = spec["kind"]
            if has_type_picker:
                chart_type = control_cols[control_index].selectbox("Chart type", spec["types"], index=spec["types"].index(spec["kind"]), key=f"{chart_key}_type", label_visibility="collapsed")
                control_index += 1
            sort_choice = None
            if has_sort:
                sort_choice = control_cols[control_index].selectbox("Sort", sort_options, key=f"{chart_key}_sort", label_visibility="collapsed")
                control_index += 1
            expand_chart = control_cols[control_index].button("Expand", key=f"{chart_key}_expand", icon=":material/open_in_full:", width="stretch") if allow_expand else False
        if sort_choice == "Largest first":
            chart_data = chart_data.sort_values(y_name, ascending=False)
        elif sort_choice == "Smallest first":
            chart_data = chart_data.sort_values(y_name, ascending=True)
        elif sort_choice == "A to Z":
            chart_data = chart_data.sort_values(x_name, key=lambda values: values.astype(str).str.casefold())

        x_type = "time" if pd.api.types.is_datetime64_any_dtype(chart_data[x_name]) else "category"
        series = {"type": chart_type, "encode": {"x": x_name, "y": y_name}, "name": str(spec["title"]), "universalTransition": True, "itemStyle": {"color": spec["color"]}}
        if chart_type == "line":
            series["smooth"] = True
            series["showSymbol"] = len(chart_data) < 30
        if chart_type == "area":
            series["type"] = "line"
            series["areaStyle"] = {"opacity": 0.22}
            series["smooth"] = True
        option = {
            "animationDuration": 450,
            "tooltip": {"trigger": "item" if chart_type in ("pie", "scatter") else "axis", "valueFormatter": "compact"},
            "dataset": {"source": chart_data},
            "xAxis": {"type": x_type, "name": str(x_name), "nameLocation": "middle", "nameGap": 28},
            "yAxis": {"type": "value", "name": str(y_name), "nameLocation": "middle", "nameGap": 48, "axisLabel": {"formatter": "{value}"}},
            "grid": {"left": 62, "right": 20, "top": 35, "bottom": 58, "containLabel": True},
            "toolbox": {"show": False, "right": 8, "feature": {"dataView": {"show": True, "readOnly": True}, "magicType": {"show": has_type_picker and chart_type != "pie", "type": [x for x in ("line", "bar") if x in spec["types"]]}, "restore": {"show": True}, "saveAsImage": {"show": True}}},
            "series": [series],
        }
        if chart_type == "pie":
            option["xAxis"] = {"show": False}
            option["yAxis"] = {"show": False}
            option["grid"] = {"show": False}
            option["legend"] = {"orient": "vertical", "right": 8, "top": "middle"}
            option["series"] = [{"type": "pie", "radius": ["38%", "68%"], "encode": {"itemName": x_name, "value": y_name}, "name": str(spec["title"]), "label": {"show": True, "formatter": "{d}%"}, "itemStyle": {"borderColor": "#0B1020", "borderWidth": 2}}]
        elif chart_type == "scatter":
            option["xAxis"] = {"type": "value", "name": str(x_name), "nameLocation": "middle", "nameGap": 28}
            option["series"][0]["symbolSize"] = 9
            option["series"][0]["itemStyle"] = {"color": spec["color"], "opacity": 0.65}
        if x_type == "time" and chart_type in ("line", "area"):
            option["dataZoom"] = [{"type": "inside"}, {"type": "slider", "height": 18, "bottom": 8}]
            option["toolbox"]["feature"]["dataZoom"] = {"show": True, "yAxisIndex": "none"}
        st.echarts_chart(option, height=height, key=f"{chart_key}_chart", alt=str(spec["title"]))
        return expand_chart

    @st.dialog("Expanded chart", width="large")
    def show_expanded_chart(chart_spec: dict, chart_key: str) -> None:
        st.subheader(chart_spec["title"])
        display_chart(chart_spec, f"{chart_key}_large", height=650, allow_expand=False)

    grid_columns = 2 if len(chart_specs) > 1 else 1
    with st.grid(grid_columns, min_column_width=420, gap="medium", key=f"{key_prefix}_smart_grid"):
        for index, spec in enumerate(chart_specs[:8]):
            card_key = f"{key_prefix}_smart_{index}"
            with st.container(border=True, key=f"{card_key}_card"):
                st.subheader(spec["title"])
                if display_chart(spec, card_key):
                    show_expanded_chart(spec, card_key)

    export_charts = []
    for spec in chart_specs[:8]:
        points_frame = spec["data"].copy()
        if spec["y"] in points_frame.columns:
            if spec["kind"] not in {"line", "area"}:
                points_frame = points_frame.sort_values(spec["y"], ascending=False)
            points_frame = points_frame.head(12)
            points = [(str(row[spec["x"]]), float(row[spec["y"]])) for _, row in points_frame.iterrows()]
        else:
            points = []
        export_charts.append({"title": spec["title"], "dimension": str(spec["x"]), "measure": str(spec["y"]), "type": "Pie" if spec["kind"] == "pie" else "Line" if spec["kind"] in {"line", "area"} else "Bar", "points": points})
    _render_dashboard_exports(key_prefix, f"{source_name} · overview", kpis, export_charts)


def render_visualization(frame: pd.DataFrame, key_prefix: str = "chart") -> None:
    st.subheader("Chart builder")
    recommendation = recommended_chart(frame)
    st.caption(f"Suggested view: {recommendation}")
    chart_options = ["Auto", "Bar", "Line", "Area", "Scatter", "Pie", "Histogram", "Data table"]
    selection = st.selectbox("Chart type", chart_options, key=f"{key_prefix}_type")
    chart_type = recommendation if selection == "Auto" else selection

    if chart_type == "Data table":
        st.dataframe(frame, hide_index=True, alt="Results in a data table")
        return

    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]

    if chart_type == "Histogram":
        if not numeric:
            st.info("Choose a chart type that works with these results.")
            return
        x_field = st.selectbox("Value", numeric, key=f"{key_prefix}_hist_x_{hash(tuple(numeric))}")
        try:
            build_chart(frame, chart_type, x_field, "", None, "No aggregation", len(frame), False)
        except Exception:
            LOGGER.exception("Histogram rendering failed")
            st.info("This chart could not be displayed. Try another chart type or field.")
        return

    if chart_type == "Scatter":
        if len(numeric) < 2:
            st.info("A scatter chart needs two numeric columns. Try a bar, line, or table view.")
            return
        x_field = st.selectbox("X-axis", numeric, key=f"{key_prefix}_scatter_x_{hash(tuple(numeric))}")
        y_options = [column for column in numeric if column != x_field]
        y_field = st.selectbox("Y-axis", y_options, key=f"{key_prefix}_scatter_y_{hash(tuple(y_options))}")
        color_options = ["None", *categorical]
        color_choice = st.selectbox("Color / series", color_options, key=f"{key_prefix}_scatter_color_{hash(tuple(color_options))}")
        try:
            build_chart(
                frame, chart_type, x_field, y_field,
                None if color_choice == "None" else color_choice,
                "No aggregation", len(frame), False,
            )
        except Exception:
            LOGGER.exception("Scatter chart rendering failed")
            st.info("This chart could not be displayed. Try another chart type or field.")
        return

    x_options = list(frame.columns)
    default_x = next(
        (column for column in x_options if is_time_column(frame, column)),
        next((column for column in x_options if column in categorical), x_options[0]),
    )
    x_field = st.selectbox(
        "X-axis / dimension",
        x_options,
        index=x_options.index(default_x),
        key=f"{key_prefix}_x_{hash(tuple(x_options))}",
    )
    y_options = [column for column in numeric if column != x_field]
    measure_options = y_options or ["Record count"]
    y_field = st.selectbox(
        "Y-axis / measure",
        measure_options,
        key=f"{key_prefix}_y_{hash(tuple(measure_options))}",
    )
    color_options = ["None", *[column for column in categorical if column != x_field]]
    color_choice = st.selectbox(
        "Color / series",
        color_options,
        key=f"{key_prefix}_color_{hash(tuple(color_options))}",
    )
    color_field = None if color_choice == "None" else color_choice

    aggregation_options = ["Count"] if y_field == "Record count" else [
        "Sum", "Average", "Count", "No aggregation"
    ]
    aggregation = st.selectbox(
        "Aggregation",
        aggregation_options,
        key=f"{key_prefix}_aggregation_{hash((chart_type, tuple(aggregation_options)))}",
    )
    show_labels = st.toggle("Show bar labels", key=f"{key_prefix}_labels") if chart_type == "Bar" else False
    top_n = len(frame)
    top_max = min(50, len(frame))
    if chart_type in {"Bar", "Pie"} and top_max > 1:
        top_n = st.slider(
            "Limit to top categories",
            min_value=1,
            max_value=top_max,
            value=min(20, top_max),
            key=f"{key_prefix}_top_n_{len(frame)}",
        )

    try:
        build_chart(frame, chart_type, x_field, y_field, color_field, aggregation, top_n, show_labels)
    except Exception:
        LOGGER.exception("Chart rendering failed")
        st.info("This chart could not be displayed. Try another chart type or field.")


def render_welcome_page() -> None:
    with st.grid(2, min_column_width=400, gap="large", vertical_alignment="center"):
        with st.container():
            st.caption("NATURAL-LANGUAGE DATA ANALYSIS · BUILT BY VASHUGUPTADA")
            st.title("Your data has answers. Just ask.", icon=":material/insights:")
            st.markdown(
                "Turn everyday questions into clear answers, readable SQL, and charts you can explore. "
                "Start with sample data or bring your own CSV / SQLite file."
            )
            st.markdown("A conversational analytics project by **Vashuguptada** · [View the GitHub repo](https://github.com/Vashuguptada/Ask-your-data)")
            with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
                if st.button(
                    "Try the demo",
                    type="primary",
                    icon=":material/play_arrow:",
                    width="content",
                    wrap=False,
                    key="welcome_try_demo",
                ):
                    st.session_state["source_mode"] = "Demo data"
                    st.session_state["show_welcome"] = False
                    st.rerun()
                if st.button(
                    "Connect your data",
                    icon=":material/upload_file:",
                    width="content",
                    wrap=False,
                    key="welcome_connect_data",
                ):
                    st.session_state["source_mode"] = "Connect a file"
                    st.session_state["show_welcome"] = False
                    st.rerun()

        with st.container():
            with st.container(border=True):
                st.caption("FROM QUESTION TO INSIGHT")
                st.markdown("### What was our best month?")
                st.markdown(
                    "<div style='height:150px;display:flex;align-items:end;gap:12px;padding:14px 8px 4px;'>"
                    "<div style='height:35%;flex:1;background:linear-gradient(180deg,#50d6c5,#176d69);border-radius:10px 10px 3px 3px'></div>"
                    "<div style='height:52%;flex:1;background:linear-gradient(180deg,#50d6c5,#176d69);border-radius:10px 10px 3px 3px'></div>"
                    "<div style='height:43%;flex:1;background:linear-gradient(180deg,#50d6c5,#176d69);border-radius:10px 10px 3px 3px'></div>"
                    "<div style='height:78%;flex:1;background:linear-gradient(180deg,#8c7bff,#5446c9);border-radius:10px 10px 3px 3px'></div>"
                    "<div style='height:61%;flex:1;background:linear-gradient(180deg,#50d6c5,#176d69);border-radius:10px 10px 3px 3px'></div>"
                    "<div style='height:92%;flex:1;background:linear-gradient(180deg,#50d6c5,#176d69);border-radius:10px 10px 3px 3px'></div></div>",
                    unsafe_allow_html=True,
                )
                st.caption("Ask → SQL → answer → interactive chart")

    st.subheader("What you can do")
    feature_cards = [
        (
            "Ask naturally",
            "Ask business questions in everyday language and continue with follow-ups in the same chat.",
            ":material/chat:",
        ),
        (
            "See how it answered",
            "Review the generated read-only SQL alongside the answer and result table.",
            ":material/visibility:",
        ),
        (
            "Explore visually",
            "Get an automatic chart, then adjust its type, axes, grouping, and aggregation.",
            ":material/insert_chart:",
        ),
        (
            "Bring a dataset",
            "Use the built-in sample or upload a CSV / SQLite file to explore your own data.",
            ":material/database:",
        ),
    ]
    with st.grid(4, min_column_width=230, gap="medium"):
        for title, description, icon in feature_cards:
            with st.container(border=True, height="stretch"):
                st.subheader(title, icon=icon)
                st.write(description)

    st.subheader("A simple workflow")
    workflow = [
        ("01 · Choose a source", "Use the sample data or upload a CSV / SQLite file."),
        ("02 · Ask a question", "Ask for a trend, comparison, total, or a follow-up."),
        ("03 · Explore the result", "Read the answer, inspect SQL, and adjust the chart."),
    ]
    with st.grid(3, min_column_width=250, gap="medium"):
        for title, description in workflow:
            with st.container(border=True, height="stretch"):
                st.markdown(f"**{title}**")
                st.caption(description)

    with st.container(border=True):
        st.subheader("Example questions")
        st.markdown(
            "- What is the total revenue from delivered orders?\n"
            "- Which product categories are growing month over month?\n"
            "- Compare the return rate across sales channels."
        )

    st.info(
        "CSV and SQLite file uploads are supported. For AI answers, the schema, sample rows, "
        "conversation, question, and results can be sent to Groq after the user acknowledges "
        "the notice. Avoid confidential data.",
        icon=":material/lock_open:",
    )
    st.caption("Powered by Streamlit, LangChain, SQLite, and Groq.")
    st.caption("Appearance: choose Light or Dark from the top-right menu → Settings → Theme.")


if st.session_state["show_welcome"]:
    with st.sidebar:
        st.title("Ask Your Data", icon=":material/insights:")
        st.caption("A conversational workspace for exploring data.")
    render_welcome_page()
    st.stop()


with st.sidebar:
    st.title("Workspace", icon=":material/space_dashboard:")
    if st.button("Welcome page", icon=":material/home:", key="back_to_welcome"):
        st.session_state["show_welcome"] = True
        st.rerun()
    if st.button("New chat", icon=":material/add_comment:", width="stretch", key="sidebar_new_chat"):
        start_new_chat()
        st.rerun()
    st.divider()
    st.caption("Choose a source to start exploring.")
    source_mode = st.segmented_control(
        "Data source",
        ["Demo data", "Connect a file"],
        default="Demo data",
        key="source_mode",
        width="stretch",
    )

    uploaded_file = None
    business_rules = ""
    source_path = None
    source_name = "Fashion commerce demo"
    source_metadata = {"rows": None, "name": source_name}
    source_signature = str(DEMO_DB)

    if source_mode == "Connect a file":
        previous_path = st.session_state.get("connected_db_path")
        if previous_path and Path(previous_path).exists():
            source_path = Path(previous_path)
            source_name = st.session_state.get("connected_file_name", "Uploaded data")
            source_metadata = st.session_state.get("connected_metadata", {"rows": None})
            source_signature = str(source_path)
    else:
        old_upload = st.session_state.get("connected_db_path")
        if old_upload:
            Path(old_upload).unlink(missing_ok=True)
        st.session_state["connected_db_path"] = None
        st.caption("Using the built-in fashion commerce sample data.")
        source_path = DEMO_DB

    if source_mode == "Connect a file" and source_path is None:
        st.divider()
        st.caption("Choose Demo data to remove the active temporary upload from the server.")

    st.divider()
    st.caption("Ask Your Data | prototype")


with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
    st.title("Ask your data", icon=":material/insights:")
    if st.button("New chat", icon=":material/add_comment:", width="content", wrap=False, key="main_new_chat"):
        start_new_chat()
        st.rerun()
st.markdown("Ask follow-ups, explore your results, and build a dashboard from your data.")

if source_mode == "Connect a file":
    with st.container(border=True):
        st.subheader("Connect your data", icon=":material/upload_file:")
        uploaded_file = st.file_uploader(
            "Choose a CSV or SQLite file",
            type=["csv", "db", "sqlite", "sqlite3"],
            help="CSV becomes one queryable table. SQLite files keep their existing tables.",
            key="workspace_data_upload",
        )
        st.caption("Uploads up to 50 MB · CSV and SQLite supported")
        st.info(
            "The file is temporarily copied to this app's server. For AI answers, table and column names, "
            "sample rows, definitions, conversation, question, and query results may be sent to Groq. "
            "Avoid confidential data.",
            icon=":material/lock_open:",
        )
        business_rules = st.text_area(
            "Business definitions (optional)",
            placeholder="Example: Net revenue excludes refunded orders.",
            height=90,
            key="business_rules",
        )
        if uploaded_file is not None:
            try:
                source_path, source_metadata = persist_upload(uploaded_file)
                source_name = source_metadata["name"]
                source_signature = str(source_path)
                st.success(f"Connected: {source_name}", icon=":material/check_circle:")
            except Exception:
                LOGGER.exception("Uploaded data could not be connected")
                st.warning("This file could not be connected. Use a valid CSV or SQLite file.")
        elif source_path is not None:
            st.success(f"Connected: {source_name}", icon=":material/check_circle:")

if source_path is not None:
    try:
        tables = list_tables(source_path)
    except sqlite3.Error as error:
        LOGGER.exception("Could not inspect the selected database: %s", error)
        tables = []
        st.warning("This data source could not be opened. Choose another file or use Demo data.")

    st.caption(f"Data source: {source_name}")

    view = st.segmented_control(
        "Workspace view",
        ["Ask in chat", "Dashboard", "Data preview", "Build your own"],
        default="Ask in chat",
        key="workspace_view",
    )
    st.caption("Switch between light and dark in the top-right menu: Settings → Theme.")

    if view == "Data preview":
        with st.container(border=True):
            st.subheader("Explore your tables")
            if tables:
                selected_table = st.selectbox("Table", tables, key="preview_table")
                try:
                    preview = load_preview(source_path, selected_table, limit=100)
                    st.caption(f"Preview: {selected_table}")
                    st.dataframe(preview, hide_index=True, alt=f"Preview of {selected_table}")
                except Exception:
                    LOGGER.exception("Data preview failed for table %s", selected_table)
                    st.info("This table preview is unavailable. Try another table.")
            else:
                st.info("No queryable tables were found. Connect another file or use Demo data.")

    elif view == "Dashboard":
        with st.container(border=True):
            st.subheader("Sales performance dashboard", icon=":material/space_dashboard:")
            st.caption("A complete overview of sales, products, channels, customers, and returns.")
            try:
                if source_mode == "Demo data":
                    render_sales_dashboard(str(DEMO_DB), "sales_dashboard")
                elif tables:
                    dashboard_table = st.selectbox(
                        "Dashboard data",
                        tables,
                        key=f"dashboard_table_{hashlib.sha256(source_signature.encode()).hexdigest()[:10]}",
                    )
                    dashboard_frame = load_preview(source_path, dashboard_table, limit=5000)
                    table_token = hashlib.sha256(f"{source_signature}|{dashboard_table}".encode()).hexdigest()[:10]
                    render_generic_dashboard(dashboard_frame, dashboard_table, f"auto_dashboard_{table_token}")
                else:
                    st.info("No tables are available. Connect a CSV or SQLite file, or choose Demo data.")
            except Exception:
                LOGGER.exception("Automatic dashboard generation failed")
                st.info("This dashboard could not be generated. Try another table or data source.")

    elif view == "Build your own":
        with st.container(border=True):
            st.subheader("Build your own dashboard", icon=":material/dashboard_customize:")
            st.caption("Start with eight chart cards, then choose each title, dimension, measure, chart type, and aggregation.")
            if source_mode == "Demo data":
                editable_frame = load_demo_dashboard_data(str(DEMO_DB))
                render_dashboard(editable_frame, "custom_demo_dashboard")
            elif tables:
                selected_table = st.selectbox(
                    "Choose a table to build from",
                    tables,
                    key=f"builder_table_{hashlib.sha256(source_signature.encode()).hexdigest()[:10]}",
                )
                editable_frame = load_preview(source_path, selected_table, limit=5000)
                table_token = hashlib.sha256(f"custom|{source_signature}|{selected_table}".encode()).hexdigest()[:10]
                render_dashboard(editable_frame, f"custom_dashboard_{table_token}")
            else:
                st.info("Connect a file first, or select Demo data from the workspace source menu.")

    else:
        if st.session_state.get("conversation_source") != source_signature:
            st.session_state["conversation_source"] = source_signature
            st.session_state["conversation"] = []
            st.session_state["conversation_id"] = uuid.uuid4().hex
            st.session_state["latest_result"] = None
            st.session_state["last_sample_prompt"] = None
            st.session_state["sample_prompt"] = None
            st.session_state["allow_groq_processing"] = False
            st.session_state.pop("pending_question", None)
            st.session_state.pop("chat_input", None)

        st.session_state.setdefault("conversation", [])
        prompt_options = DEMO_PROMPTS if source_mode == "Demo data" else FILE_PROMPTS
        selected_prompt = None
        if not st.session_state["conversation"]:
            selected_prompt = st.pills(
                "Try a question",
                prompt_options,
                selection_mode="single",
                key="sample_prompt",
            )
            if selected_prompt and selected_prompt != st.session_state.get("last_sample_prompt"):
                st.session_state["last_sample_prompt"] = selected_prompt
                st.session_state["pending_question"] = selected_prompt

        if source_mode == "Connect a file":
            approved_data_processing = st.checkbox(
                "I'm allowed to send this data and conversation to Groq for analysis.",
                key="allow_groq_processing",
            )
        else:
            approved_data_processing = True

        if not st.session_state["conversation"]:
            with st.container(border=True):
                st.subheader("Start with a question")
                st.caption("Try a suggested question above, or ask anything about the connected data.")

        for turn_index, message in enumerate(st.session_state["conversation"], start=1):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    response_number = turn_index // 2
                    conversation_id = st.session_state["conversation_id"][:10]
                    turn_prefix = f"{conversation_id}_{response_number}"
                    visual_key = f"show_visualization_{turn_prefix}"
                    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                        if message.get("df") is not None and st.button(
                            "Visualize" if not st.session_state.get(visual_key) else "Hide chart",
                            icon=":material/insert_chart:",
                            width="content",
                            wrap=False,
                            key=f"visualize_button_{turn_prefix}",
                        ):
                            st.session_state[visual_key] = not st.session_state.get(visual_key, False)
                        if message.get("sql"):
                            with st.popover("SQL", icon=":material/code:"):
                                st.code(message["sql"], language="sql")
                    if message.get("df") is not None:
                        with st.expander(f"View results for turn {response_number}"):
                            st.dataframe(message["df"], hide_index=True, alt="Query results")
                            st.download_button(
                                "Download CSV",
                                data=message["df"].to_csv(index=False).encode("utf-8"),
                                file_name=f"ask-your-data-turn-{response_number}.csv",
                                mime="text/csv",
                                key=f"download_results_{turn_prefix}",
                            )
                        if st.session_state.get(visual_key):
                            with st.container(border=True):
                                render_visualization(message["df"], key_prefix=f"turn_{turn_prefix}")
                    if message.get("retried"):
                        st.caption("The query was adjusted once before results were returned.")

        question = st.chat_input("Ask a follow-up question...", key="chat_input")
        pending_question = st.session_state.pop("pending_question", None)
        question = pending_question or question

        if question:
            if source_mode == "Connect a file" and not approved_data_processing:
                st.session_state["pending_question"] = question
                st.info("Confirm the data-sharing notice before asking about this file.")
            else:
                conversation = st.session_state["conversation"]
                history = [dict(message) for message in conversation]
                conversation.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                active_glossary = None
                if source_mode == "Connect a file":
                    active_glossary = business_rules.strip() or GENERIC_GLOSSARY

                with st.chat_message("assistant"):
                    progress = st.status(":shimmer[Reading your data]", type="compact")
                    answer_slot = st.empty()
                    streamed_answer = []

                    def update_stage(label: str) -> None:
                        progress.update(label=f":shimmer[{label}]", state="running")

                    def show_answer_chunk(chunk: str) -> None:
                        streamed_answer.append(chunk)
                        answer_slot.markdown("".join(streamed_answer))

                    try:
                        result = ask(
                            question,
                            db_path=str(source_path),
                            glossary=active_glossary,
                            chat_history=history,
                            on_stage=update_stage,
                            on_answer_chunk=show_answer_chunk,
                        )
                    except Exception:
                        LOGGER.exception("Question processing failed")
                        result = {
                            "sql": None,
                            "df": None,
                            "answer": None,
                            "error": "request_failed",
                            "retried": False,
                        }
                    progress.update(label="Analysis complete", state="complete")
                    if result.get("error"):
                        LOGGER.error("Question processing returned an error: %s", result["error"])
                        if result.get("df") is not None:
                            answer = "I found results. You can review the table and chart below."
                        else:
                            answer = "I couldn't complete that request. Try rephrasing your question and ask again."
                        answer_slot.empty()
                        answer_slot.markdown(answer)
                    else:
                        answer = result.get("answer") or "I found results for your question."
                        if not streamed_answer:
                            answer_slot.markdown(answer)

                conversation.append({
                    "role": "assistant",
                    "content": answer,
                    "sql": result.get("sql"),
                    "retried": result.get("retried", False),
                    "df": result.get("df"),
                })
                st.session_state["latest_result"] = {
                    "source": source_signature,
                    **result,
                }
                st.rerun()

else:
    st.caption("Upload a file above to open the analytics workspace.")
