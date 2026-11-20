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

    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]
    business_numeric = [
        column for column in numeric
        if not re.search(r"(^id$|_id$|^id_|zip|postal|phone|index)", str(column), flags=re.I)
    ]

    title_key = f"{key_prefix}_title"
    st.text_input("Dashboard title", value="Dataset overview", key=title_key)
    st.subheader(st.session_state.get(title_key, "Dataset overview"))
    st.caption("Overview and charts use up to 5,000 rows from the selected table.")

    filter_fields = [column for column in frame.columns if column in categorical or is_time_column(frame, column)]
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
                numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
                categorical = [column for column in frame if column not in numeric]
                if frame.empty:
                    st.info("No rows match these filters.")
                    return

    metrics = [
        ("Rows in sample", f"{len(frame):,}"),
        ("Fields", f"{len(frame.columns):,}"),
        ("Numeric measures", f"{len(numeric):,}"),
        ("Dimensions", f"{len(categorical):,}"),
        ("Missing cells", f"{int(frame.isna().sum().sum()):,}"),
    ]
    if categorical:
        metrics.append((f"Unique {categorical[0]}", f"{frame[categorical[0]].nunique(dropna=True):,}"))
    if business_numeric:
        measure = business_numeric[0]
        if re.search(r"revenue|sales|amount|profit|total", measure, flags=re.I):
            metrics.append((f"Total {measure}", f"{frame[measure].sum():,.2f}"))
        else:
            metrics.append((f"Average {measure}", f"{frame[measure].mean():,.2f}"))
    with st.container(horizontal=True):
        for label, value in metrics:
            st.metric(label, value, border=True)

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

    x_options = list(frame.columns)
    default_dimensions = [column for column in x_options if is_time_column(frame, column)]
    default_dimensions.extend(column for column in x_options if column in categorical and column not in default_dimensions)
    default_dimensions.extend(column for column in x_options if column not in default_dimensions)
    chart_types = ["Bar", "Line", "Area", "Pie", "Data table"]
    default_chart_types = ["Line", "Bar", "Bar", "Pie", "Area", "Bar", "Line", "Bar"]
    with st.grid(2, min_column_width=400, gap="medium"):
        for index in range(st.session_state[count_key]):
            panel_key = f"{key_prefix}_chart_{index + 1}"
            with st.container(border=True):
                default_x = default_dimensions[index % len(default_dimensions)]
                x_field = st.selectbox(
                    "Dimension", x_options,
                    index=x_options.index(default_x),
                    key=f"{panel_key}_x",
                )
                y_options = [column for column in numeric if column != x_field] or ["Record count"]
                default_y = y_options[index % len(y_options)]
                y_field = st.selectbox("Measure", y_options, index=y_options.index(default_y), key=f"{panel_key}_y")
                suggested = "Line" if is_time_column(frame, x_field) else default_chart_types[index % len(default_chart_types)]
                chart_type = st.selectbox(
                    "Chart type", chart_types,
                    index=chart_types.index(suggested),
                    key=f"{panel_key}_type",
                )
                title_default = f"{y_field if y_field != 'Record count' else 'Records'} by {x_field}"
                st.text_input("Chart title", value=title_default, key=f"{panel_key}_title")
                color_options = ["None", *[column for column in categorical if column != x_field]]
                color_choice = st.selectbox("Color by", color_options, key=f"{panel_key}_color")
                aggregation_options = ["Count"] if y_field == "Record count" else ["Sum", "Average", "Count", "No aggregation"]
                aggregation = st.selectbox("Aggregation", aggregation_options, key=f"{panel_key}_aggregation")
                title = st.session_state.get(f"{panel_key}_title", f"Insight {index + 1}")
                st.markdown(f"**{title}**")
                try:
                    build_chart(
                        frame, chart_type, x_field, y_field,
                        None if color_choice == "None" else color_choice,
                        aggregation, min(50, len(frame)), False,
                    )
                except Exception:
                    LOGGER.exception("Dashboard chart rendering failed")
                    st.info("This chart is unavailable. Try another field or chart type.")

    save_key = f"{key_prefix}_saved"
    if st.button("Save dashboard layout", type="primary", icon=":material/save:", key=f"{key_prefix}_save"):
        st.session_state[save_key] = {
            "title": st.session_state.get(title_key, "Dataset overview"),
            "chart_count": st.session_state[count_key],
            "charts": [
                {
                    "title": st.session_state.get(f"{key_prefix}_chart_{index + 1}_title", f"Insight {index + 1}"),
                    "dimension": st.session_state.get(f"{key_prefix}_chart_{index + 1}_x"),
                    "measure": st.session_state.get(f"{key_prefix}_chart_{index + 1}_y"),
                    "type": st.session_state.get(f"{key_prefix}_chart_{index + 1}_type"),
                    "color_by": st.session_state.get(f"{key_prefix}_chart_{index + 1}_color"),
                    "aggregation": st.session_state.get(f"{key_prefix}_chart_{index + 1}_aggregation"),
                }
                for index in range(st.session_state[count_key])
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
                show_labels: bool):
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    if chart_type == "Data table":
        st.dataframe(frame, hide_index=True, alt="Query results")
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
            .properties(height=380, title=f"Distribution of {x_field}")
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
            .properties(height=380, title=f"{y_field} vs. {x_field}")
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
            .properties(height=380, title=f"{value_field} by {x_field}")
        )
    else:
        mark = {"Bar": "bar", "Line": "line", "Area": "area"}[chart_type]
        chart = alt.Chart(chart_frame).mark_bar() if mark == "bar" else (
            alt.Chart(chart_frame).mark_line(point=True) if mark == "line"
            else alt.Chart(chart_frame).mark_area(opacity=0.7)
        )
        x_sort = (
            alt.EncodingSortField(field=value_field, order="descending")
            if chart_type == "Bar" and x_type == "nominal" else None
        )
        encodings = {
            "x": alt.X(field=x_field, type=x_type, title=x_field, sort=x_sort),
            "y": alt.Y(field=value_field, type="quantitative", title=value_field),
            "tooltip": tooltip,
        }
        if color_field and color_field != x_field:
            encodings["color"] = alt.Color(field=color_field, type="nominal", title=color_field)
        chart = chart.encode(**encodings).properties(height=380, title=f"{value_field} by {x_field}")
        if show_labels and chart_type == "Bar":
            labels = alt.Chart(chart_frame).mark_text(dy=-8, fontSize=11).encode(
                x=alt.X(field=x_field, type=x_type),
                y=alt.Y(field=value_field, type="quantitative"),
                text=alt.Text(field=value_field, type="quantitative", format="~s"),
            )
            chart = chart + labels

    st.altair_chart(chart, width="stretch", alt=f"{chart_type} chart of {value_field} by {x_field}")


DASH_PALETTE = ["#4C83F3", "#F04D86", "#F5B934", "#0CC9AE", "#8A63E6", "#22B8CF"]


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
    """An eight-view overview for a user-connected table, with no chart controls exposed."""
    if frame.empty:
        st.info("This table has no rows to analyze.")
        return
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    dimensions = [column for column in frame if column not in numeric]
    date_field = next((column for column in frame if is_time_column(frame, column)), None)
    business_numeric = [
        column for column in numeric
        if not re.search(r"(^id$|_id$|^id_|zip|postal|phone|index)", str(column), flags=re.I)
    ]

    unique_field = dimensions[0] if dimensions else frame.columns[0]
    with st.grid(6, min_column_width=165, gap="small"):
        st.metric("Rows in preview", f"{len(frame):,}", border=True)
        st.metric("Fields", f"{len(frame.columns):,}", border=True)
        st.metric("Numeric fields", f"{len(numeric):,}", border=True)
        st.metric("Dimensions", f"{len(dimensions):,}", border=True)
        st.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}", border=True)
        st.metric(f"Unique {unique_field}", f"{frame[unique_field].nunique(dropna=True):,}", border=True)
    st.caption(f"{source_name} · overview uses up to 5,000 rows from the selected table")

    charts: list[tuple[str, alt.Chart]] = []
    category = dimensions[0] if dimensions else None
    if category:
        records = frame[category].fillna("(empty)").astype(str).value_counts().head(12).rename_axis("value").reset_index(name="records")
        charts.append((f"Records by {category}", _dashboard_bar(records, "value", "records", f"Record frequency by {category}", DASH_PALETTE[0])))

    missing = frame.isna().sum().rename_axis("field").reset_index(name="missing")
    charts.append(("Missing values by field", _dashboard_bar(missing, "field", "missing", "Missing values by field", DASH_PALETTE[1])))
    unique_counts = frame.nunique(dropna=True).rename_axis("field").reset_index(name="unique_values")
    charts.append(("Distinct values by field", _dashboard_bar(unique_counts, "field", "unique_values", "Distinct values by field", DASH_PALETTE[4])))

    dtype_counts = frame.dtypes.astype(str).value_counts().rename_axis("type").reset_index(name="fields")
    charts.append(("Field types", _dashboard_donut(dtype_counts, "type", "fields", "Dataset field types")))

    if business_numeric:
        measure = business_numeric[0]
        distribution = (
            alt.Chart(frame.dropna(subset=[measure]))
            .mark_bar(color=DASH_PALETTE[2], cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(f"{measure}:Q", bin=alt.Bin(maxbins=24), title=measure),
                y=alt.Y("count():Q", title="Records"),
                tooltip=[alt.Tooltip(f"{measure}:Q", bin=alt.Bin(maxbins=24)), alt.Tooltip("count():Q", title="Records")],
            )
            .properties(height=280)
        )
        charts.append((f"{measure} distribution", distribution))

        if category:
            grouped = frame.groupby(category, dropna=False)[measure].mean().nlargest(12).rename_axis("value").reset_index(name="average")
            charts.append((f"Average {measure} by {category}", _dashboard_bar(grouped, "value", "average", f"Average {measure} by {category}", DASH_PALETTE[3])))
        else:
            summary = pd.DataFrame({"statistic": ["Minimum", "Median", "Average", "Maximum"], "value": [frame[measure].min(), frame[measure].median(), frame[measure].mean(), frame[measure].max()]})
            charts.append((f"{measure} summary", _dashboard_bar(summary, "statistic", "value", f"Summary statistics for {measure}", DASH_PALETTE[3])))
    else:
        if category:
            text_lengths = frame[category].dropna().astype(str).str.len().rename("characters").to_frame()
            length_chart = (
                alt.Chart(text_lengths)
                .mark_bar(color=DASH_PALETTE[2], cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(x=alt.X("characters:Q", bin=alt.Bin(maxbins=20), title="Characters per value"),
                        y=alt.Y("count():Q", title="Records"),
                        tooltip=[alt.Tooltip("characters:Q", bin=alt.Bin(maxbins=20)), alt.Tooltip("count():Q", title="Records")])
                .properties(height=280)
            )
            charts.append((f"Text length · {category}", length_chart))
        else:
            value_frequency = frame[unique_field].fillna("(empty)").astype(str).value_counts().head(12).rename_axis("value").reset_index(name="records")
            charts.append((f"Top values in {unique_field}", _dashboard_bar(value_frequency, "value", "records", f"Top values in {unique_field}", DASH_PALETTE[2])))

    if date_field:
        date_frame = frame[[date_field]].copy()
        date_frame[date_field] = pd.to_datetime(date_frame[date_field], errors="coerce")
        by_month = date_frame.dropna().assign(month=lambda data: data[date_field].dt.to_period("M").dt.to_timestamp()).groupby("month").size().rename("records").reset_index()
        if not by_month.empty:
            charts.append(("Records over time", _dashboard_line(by_month, "month", "records", "Records by month")))

    if len(numeric) > 1:
        x_metric, y_metric = numeric[:2]
        scatter = (
            alt.Chart(frame.dropna(subset=[x_metric, y_metric]).head(2000))
            .mark_circle(size=65, opacity=0.7, color=DASH_PALETTE[5])
            .encode(
                x=alt.X(f"{x_metric}:Q", title=x_metric),
                y=alt.Y(f"{y_metric}:Q", title=y_metric),
                tooltip=[alt.Tooltip(f"{x_metric}:Q"), alt.Tooltip(f"{y_metric}:Q")],
            )
            .properties(height=280)
        )
        charts.append(("Measure relationship", scatter))
    elif category:
        shares = frame[category].fillna("(empty)").astype(str).value_counts().head(8).rename_axis("value").reset_index(name="records")
        charts.append((f"{category} composition", _dashboard_donut(shares, "value", "records", f"Share by {category}")))
    elif business_numeric:
        measure = business_numeric[0]
        box_data = frame[[measure]].dropna().assign(group="All records")
        box_chart = (
            alt.Chart(box_data)
            .mark_boxplot(color=DASH_PALETTE[5], extent="min-max")
            .encode(x=alt.X(f"{measure}:Q", title=measure), y=alt.Y("group:N", title="Dataset"))
            .properties(height=280)
        )
        charts.append((f"Spread of {measure}", box_chart))
    else:
        chars = frame.select_dtypes(include=["object", "string"]).astype("string").apply(lambda column: column.str.len().mean()).dropna()
        if not chars.empty:
            length_frame = chars.rename_axis("field").reset_index(name="average_length")
            charts.append(("Text length by field", _dashboard_bar(length_frame, "field", "average_length", "Average text length by field", DASH_PALETTE[5])))

    if len(charts) < 8:
        completeness = pd.DataFrame({
            "field": [str(column) for column in frame.columns],
            "non_empty": [int(frame[column].notna().sum()) for column in frame.columns],
        })
        charts.append(("Non-empty records by field", _dashboard_bar(completeness, "field", "non_empty", "Non-empty records by field", DASH_PALETTE[0])))
    if len(charts) < 8:
        missing_rate = pd.DataFrame({
            "field": [str(column) for column in frame.columns],
            "missing_percent": [float(frame[column].isna().mean() * 100) for column in frame.columns],
        })
        charts.append(("Missing rate by field", _dashboard_bar(missing_rate, "field", "missing_percent", "Missing values (%)", DASH_PALETTE[1])))
    if len(charts) < 8:
        distribution = frame.dtypes.astype(str).value_counts().rename_axis("type").reset_index(name="fields")
        charts.append(("Column type mix", _dashboard_donut(distribution, "type", "fields", "Column type mix")))
    _dashboard_grid(charts[:8], f"{key_prefix}_chart_grid")


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
