from __future__ import annotations

from datetime import datetime

import streamlit as st

from ui.components.layout import render_page_shell
from ui.components.utils import (
    QUERY_HISTORY_CACHE_KEY,
    bump_cache_revision,
    get_client_options,
    get_query_history,
)


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


def render_query_history():
    render_page_shell(
        "Review historical queries.",
        "Use explicit filters to avoid unnecessary reloads and inspect details only when needed.",
        "Query History",
        icon="history",
    )

    client_options, client_names = get_client_options()
    status_options = {
        "All statuses": "all",
        "Completed": "completed",
        "Failed": "failed",
        "Running": "running",
    }

    st.session_state.setdefault("query_history_page", 0)
    st.session_state.setdefault(
        "query_history_applied",
        {
            "client_name": "All clients",
            "status_label": "All statuses",
            "search_text": "",
            "page_size": 50,
        },
    )

    with st.form("query_history_filters"):
        filter_cols = st.columns([2, 1, 2, 1], vertical_alignment="bottom")
        client_labels = ["All clients"] + client_names
        default_client_label = st.session_state["query_history_applied"]["client_name"]
        default_client_index = (
            client_labels.index(default_client_label)
            if default_client_label in client_labels
            else 0
        )
        with filter_cols[0]:
            selected_client_name = st.selectbox(
                "Client workspace",
                client_labels,
                index=default_client_index,
            )
        with filter_cols[1]:
            selected_status_label = st.selectbox(
                "Status",
                list(status_options.keys()),
                index=max(
                    0,
                    (
                        list(status_options.keys()).index(
                            st.session_state["query_history_applied"]["status_label"]
                        )
                        if st.session_state["query_history_applied"]["status_label"]
                        in status_options
                        else 0
                    ),
                ),
            )
        with filter_cols[2]:
            search_text = st.text_input(
                "Question text search",
                value=st.session_state["query_history_applied"]["search_text"],
                placeholder="Search in question text...",
            )
        with filter_cols[3]:
            page_size = st.selectbox(
                "Rows",
                [25, 50, 100],
                index=(
                    [25, 50, 100].index(
                        st.session_state["query_history_applied"]["page_size"]
                    )
                    if st.session_state["query_history_applied"]["page_size"]
                    in [25, 50, 100]
                    else 1
                ),
            )

        apply_filters = st.form_submit_button(
            "Apply filters",
            type="primary",
            icon=":material/filter_alt:",
            width="stretch",
        )

    if apply_filters:
        st.session_state["query_history_applied"] = {
            "client_name": selected_client_name,
            "status_label": selected_status_label,
            "search_text": search_text.strip(),
            "page_size": page_size,
        }
        st.session_state["query_history_page"] = 0
        st.rerun()

    applied = st.session_state["query_history_applied"]
    if (
        applied["client_name"] != "All clients"
        and applied["client_name"] not in client_options
    ):
        applied["client_name"] = "All clients"
        st.session_state["query_history_applied"] = applied
    selected_client_id = (
        client_options[applied["client_name"]]
        if applied["client_name"] != "All clients"
        else None
    )
    selected_status = status_options[applied["status_label"]]
    current_page = st.session_state.get("query_history_page", 0)
    offset = current_page * applied["page_size"]

    try:
        payload = get_query_history(
            client_id=selected_client_id,
            status=selected_status,
            search_text=applied["search_text"],
            limit=applied["page_size"],
            offset=offset,
        )
    except Exception as exc:
        st.error(f"Unable to load query history: {exc}")
        return

    rows = payload.get("rows", [])
    total = int(payload.get("total", 0))
    if total == 0:
        st.info("No query history recorded yet.")
        return

    has_prev = current_page > 0
    has_next = (current_page + 1) * applied["page_size"] < total
    control_cols = st.columns([1, 1, 1, 3], vertical_alignment="center")
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
            bump_cache_revision(QUERY_HISTORY_CACHE_KEY)
            st.rerun()

    start_row = offset + 1
    end_row = min(offset + len(rows), total)
    st.caption(f"Showing {start_row}-{end_row} of {total} queries")

    client_name_by_id = {client_id: name for name, client_id in client_options.items()}
    for row in rows:
        with st.container(border=True):
            st.markdown(f"**{_truncate_text(row.get('question'), 140)}**")
            top_cols = st.columns([0.32, 0.18, 0.24, 0.26], vertical_alignment="center")
            top_cols[0].caption(_format_dt(row.get("created_at")))
            top_cols[1].caption(f"Status: {row.get('status', 'unknown')}")
            top_cols[2].caption(
                f"Latency: {row.get('latency_ms')} ms"
                if row.get("latency_ms") is not None
                else "Latency: n/a"
            )
            top_cols[3].caption(
                f"Retrieved: {row.get('retrieval_count', 0)} · Conflicts: {row.get('conflict_count', 0)}"
            )

            st.caption(
                "Client: "
                + client_name_by_id.get(row.get("client_id"), row.get("client_id", ""))
            )
            with st.expander("Answer details", expanded=False):
                st.write(row.get("answer") or "No answer stored.")
                st.caption(f"Query ID: {row.get('query_id')}")
