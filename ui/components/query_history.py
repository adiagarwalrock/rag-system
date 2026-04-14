from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from ui.components.auth import get_api
from ui.components.layout import render_page_shell
from ui.components.utils import get_client_options


def _truncate_text(text: str | None, limit: int = 100) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _format_dt(value: str | None) -> str:
    if not value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _status_icon(status: str) -> str:
    if status == "completed":
        return ":material/check_circle:"
    if status == "failed":
        return ":material/error:"
    if status == "running":
        return ":material/progress_activity:"
    return ":material/help:"


def render_query_history():
    render_page_shell(
        "Review historical queries.",
        "Inspect persisted query logs with latency, retrieval, and conflict summaries.",
        "Query History",
        icon="history",
    )

    api = get_api()
    client_options, client_names = get_client_options()
    client_name_by_id = {client_id: name for name, client_id in client_options.items()}

    status_options = {
        "All statuses": "all",
        "Completed": "completed",
        "Failed": "failed",
        "Running": "running",
    }

    filter_cols = st.columns([2, 1, 2, 1])
    with filter_cols[0]:
        selected_client_name = st.selectbox(
            "Client workspace",
            ["All clients"] + client_names,
            key="query_history_client_filter",
            help="Filter history to one client or review all client workspaces.",
        )
    with filter_cols[1]:
        selected_status_label = st.selectbox(
            "Status",
            list(status_options.keys()),
            key="query_history_status_filter",
        )
    with filter_cols[2]:
        search_text = st.text_input(
            "Question text search",
            key="query_history_search_filter",
            placeholder="Search in question text...",
        )
    with filter_cols[3]:
        page_size = st.selectbox(
            "Rows",
            [25, 50, 100],
            index=1,
            key="query_history_page_size",
        )

    selected_client_id = (
        client_options[selected_client_name]
        if selected_client_name != "All clients"
        else None
    )
    selected_status = status_options[selected_status_label]
    normalized_search = search_text.strip()

    filter_signature = (
        selected_client_id,
        selected_status,
        normalized_search,
        page_size,
    )
    if st.session_state.get("query_history_filter_signature") != filter_signature:
        st.session_state.query_history_filter_signature = filter_signature
        st.session_state.query_history_page = 0

    current_page = st.session_state.get("query_history_page", 0)
    offset = current_page * page_size

    try:
        payload = api.list_query_history(
            client_id=selected_client_id,
            status=selected_status,
            search_text=normalized_search,
            limit=page_size,
            offset=offset,
        )
    except Exception as exc:
        st.error(f"Unable to load query history: {exc}")
        return

    rows = payload.get("rows", [])
    total = int(payload.get("total", 0))
    has_prev = current_page > 0
    has_next = (current_page + 1) * page_size < total

    control_cols = st.columns([1, 1, 1, 3])
    with control_cols[0]:
        if st.button(
            "Previous",
            icon=":material/arrow_back:",
            disabled=not has_prev,
            key="query_history_prev",
            width="stretch",
        ):
            st.session_state.query_history_page = max(0, current_page - 1)
            st.rerun()
    with control_cols[1]:
        if st.button(
            "Next",
            icon=":material/arrow_forward:",
            disabled=not has_next,
            key="query_history_next",
            width="stretch",
        ):
            st.session_state.query_history_page = current_page + 1
            st.rerun()
    with control_cols[2]:
        if st.button(
            "Refresh",
            icon=":material/refresh:",
            key="query_history_refresh",
            width="stretch",
        ):
            st.session_state.pop("clients", None)
            st.rerun()

    if total == 0:
        st.info("No query history recorded yet.")
        return

    start_row = offset + 1
    end_row = min(offset + len(rows), total)
    st.caption(f"Showing {start_row}-{end_row} of {total} queries")

    if not rows:
        st.info("No rows match the current filters.")
        return

    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "created_at": _format_dt(row.get("created_at")),
                "client": client_name_by_id.get(
                    row.get("client_id"), row.get("client_id", "unknown")
                ),
                "user_id": row.get("user_id"),
                "question": _truncate_text(row.get("question"), 90),
                "status": row.get("status"),
                "latency_ms": row.get("latency_ms"),
                "retrieval_count": row.get("retrieval_count", 0),
                "conflict_count": row.get("conflict_count", 0),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    st.subheader("Query details")
    for row in rows:
        title = (
            f"{_format_dt(row.get('created_at'))} | "
            f"{_truncate_text(row.get('question'), 70)}"
        )
        with st.expander(title, icon=_status_icon(row.get("status", ""))):
            metric_cols = st.columns(4)
            metric_cols[0].metric("Status", row.get("status", "unknown"))
            metric_cols[1].metric(
                "Latency",
                (
                    f"{row.get('latency_ms')} ms"
                    if row.get("latency_ms") is not None
                    else "n/a"
                ),
            )
            metric_cols[2].metric("Retrieved", str(row.get("retrieval_count", 0)))
            metric_cols[3].metric("Conflicts", str(row.get("conflict_count", 0)))

            st.caption(f"Query ID: {row.get('query_id')}")
            st.caption(
                "Client: "
                + client_name_by_id.get(row.get("client_id"), row.get("client_id", ""))
            )
            st.caption(f"User ID: {row.get('user_id')}")

            st.markdown("**Question**")
            st.write(row.get("question") or "n/a")

            st.markdown("**Answer**")
            st.write(row.get("answer") or "No answer stored.")
