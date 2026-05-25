from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ui.components.api_client import get_api
from ui.components.layout import render_page_shell

KNOWN_STATUSES = {"ok", "partial", "error", "skipped", "unknown"}


def render_status() -> None:
    render_page_shell(
        "Runtime status.",
        "Current parser, vector store, database, and AI provider diagnostics.",
        "Status",
        icon="health_and_safety",
    )

    refresh_cols = st.columns([0.8, 0.2], vertical_alignment="center")
    refresh_cols[0].caption("Cached for 45 seconds.")
    if refresh_cols[1].button(
        "Refresh",
        icon=":material/refresh:",
        width="stretch",
        key="runtime_status_refresh",
    ):
        _get_runtime_status.clear()
        st.rerun()

    try:
        status = _get_runtime_status()
    except Exception as exc:
        st.error(f"Failed to load runtime status: {type(exc).__name__}: {exc}")
        return

    _render_overview(status.get("overall", {}))
    st.divider()
    _render_parser_section(status.get("parsers", {}))
    st.divider()
    _render_qdrant_section(status.get("qdrant", {}))
    st.divider()
    _render_database_section(status.get("database", {}))
    st.divider()
    _render_ai_section(status.get("ai", {}))


@st.cache_data(ttl=45, show_spinner=False)
def _get_runtime_status() -> dict[str, Any]:
    return get_api().get_runtime_status()


def _render_overview(overall: dict[str, Any]) -> None:
    cols = st.columns(4)
    items = [
        ("Database", overall.get("database")),
        ("Qdrant", overall.get("qdrant")),
        ("AI provider", overall.get("ai")),
        ("Parsers", overall.get("parsers")),
    ]
    for col, (label, value) in zip(cols, items):
        col.metric(label, _status_label(value))


def _render_parser_section(parser_status: dict[str, Any]) -> None:
    st.subheader("Parsers")
    rows = []
    for parser in parser_status.get("parsers", []):
        rows.append(
            {
                "name": parser.get("name"),
                "configured": _yes_no(parser.get("configured")),
                "enabled": _yes_no(parser.get("enabled")),
                "priority": parser.get("priority"),
                "detail": parser.get("detail"),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No parser status returned.")

    priority = parser_status.get("routing_priority") or []
    st.caption(f"Routing priority: {' -> '.join(priority)}")


def _render_qdrant_section(qdrant_status: dict[str, Any]) -> None:
    st.subheader("Qdrant")
    cols = st.columns(4)
    cols[0].metric("Status", _status_label(qdrant_status.get("status")))
    cols[1].metric("URL", str(qdrant_status.get("url") or "-"))
    cols[2].metric(
        "Document collection",
        str(qdrant_status.get("document_collection") or "-"),
    )
    cols[3].metric(
        "Chat collection",
        str(qdrant_status.get("chat_history_collection") or "-"),
    )

    _render_status_message(
        qdrant_status, partial_fallback="Qdrant check partially succeeded."
    )

    rows = qdrant_status.get("collections") or []
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No Qdrant collections returned.")


def _render_database_section(database_status: dict[str, Any]) -> None:
    st.subheader("Database")
    target = database_status.get("target") or {}
    cols = st.columns(4)
    cols[0].metric("Status", _status_label(database_status.get("status")))
    cols[1].metric("Mode", str(database_status.get("mode") or "-"))
    cols[2].metric("Dialect", str(database_status.get("dialect") or "-"))
    cols[3].metric("Target", _database_target_label(target))

    _render_status_message(database_status)

    if isinstance(target, dict) and len(target) > 1:
        rows = [{"field": key, "value": value or "-"} for key, value in target.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_ai_section(ai_status: dict[str, Any]) -> None:
    st.subheader("AI provider")
    cols = st.columns(4)
    cols[0].metric("Status", _status_label(ai_status.get("status")))
    cols[1].metric("Provider", str(ai_status.get("provider") or "-"))
    cols[2].metric("API mode", str(ai_status.get("api_mode") or "-"))
    cols[3].metric("Key valid", _yes_no(ai_status.get("key_valid")))

    model_cols = st.columns(3)
    model_cols[0].metric("LLM model", str(ai_status.get("llm_model") or "-"))
    model_cols[1].metric(
        "Embedding model",
        str(ai_status.get("embedding_model") or "-"),
    )
    model_cols[2].metric(
        "Embedding dimensions",
        str(ai_status.get("embedding_dimensions") or "-"),
    )

    initialization = ai_status.get("initialization") or {}
    models_api = ai_status.get("models_api") or {}
    checks = [
        {
            "check": "Provider initialization",
            "status": _status_label(initialization.get("status")),
            "result": initialization.get("message") or "-",
        },
        {
            "check": "Models API",
            "status": _status_label(models_api.get("status")),
            "result": models_api.get("message") or "-",
        },
        {
            "check": "LLM model listed",
            "status": _yes_no(models_api.get("llm_model_available")),
            "result": ai_status.get("llm_model") or "-",
        },
        {
            "check": "Embedding model listed",
            "status": _yes_no(models_api.get("embedding_model_available")),
            "result": ai_status.get("embedding_model") or "-",
        },
        {
            "check": "Model count",
            "status": str(models_api.get("model_count", 0)),
            "result": "models returned",
        },
    ]
    st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")


def _render_status_message(
    status_dict: dict[str, Any],
    *,
    partial_fallback: str = "",
) -> None:
    message = status_dict.get("message")
    status = status_dict.get("status")
    if status == "error":
        st.error(message or "Check failed.")
    elif status == "partial" and partial_fallback:
        st.warning(message or partial_fallback)
    elif message:
        st.caption(str(message))


def _status_label(value: Any) -> str:
    return _normalize_status(value).upper()


def _normalize_status(value: Any) -> str:
    status = str(value or "unknown").strip().lower()
    return status if status in KNOWN_STATUSES else "unknown"


def _yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _database_target_label(target: Any) -> str:
    if not isinstance(target, dict):
        return "-"
    if target.get("path"):
        return str(target["path"])
    if target.get("account"):
        database = target.get("database") or "-"
        schema = target.get("schema") or "-"
        return f"{target['account']} / {database}.{schema}"
    return "-"
