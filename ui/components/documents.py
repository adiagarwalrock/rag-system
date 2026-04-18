from __future__ import annotations

from datetime import datetime

import streamlit as st

from app.core.config import settings
from ui.components.api_client import get_api
from ui.components.layout import render_page_shell
from ui.components.utils import (
    DOCUMENTS_CACHE_KEY,
    QUERY_HISTORY_CACHE_KEY,
    bump_cache_revision,
    get_client_options,
    get_documents,
)


def _format_dt(value: str | None) -> str:
    if not value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def _status_chip(status: str) -> tuple[str, str]:
    normalized = (status or "").lower()
    if normalized in {"indexed", "completed"}:
        return "Ready", ":material/check_circle:"
    if normalized in {"queued", "processing", "running"}:
        return "In progress", ":material/progress_activity:"
    if normalized in {"failed", "deleting_failed"}:
        return "Needs attention", ":material/error:"
    if normalized in {"deleted"}:
        return "Deleted", ":material/delete:"
    return normalized or "unknown", ":material/help:"


@st.dialog("Confirm Deletion")
def confirm_delete_dialog(api, doc_id: str, doc_name: str):
    st.warning(
        "Are you sure you want to delete this document? This removes the raw file and all vector references."
    )
    st.caption(f"Document: `{doc_name}`")
    if st.button("Yes, delete document", type="primary", width="stretch"):
        with st.spinner("Deleting document..."):
            try:
                api.delete_document(doc_id, hard=True)
                bump_cache_revision(DOCUMENTS_CACHE_KEY)
                bump_cache_revision(QUERY_HISTORY_CACHE_KEY)
                st.success("Document deleted.")
            except Exception as e:
                st.error(f"Failed to delete document: {e}")
        st.rerun()


@st.dialog("Document Details")
def show_document_details_dialog(api, doc_id: str, doc_name: str):
    st.caption(f"Document: `{doc_name}`")
    try:
        status = api.get_document_status(doc_id)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Status", status.get("status", "?"))
        metric_cols[1].metric("Vector points", status.get("vector_point_count", 0))
        metric_cols[2].metric("Version", status.get("version_label") or "-")

        if err := status.get("error_message"):
            st.error(err)
        if family := status.get("version_group"):
            st.caption(f"Document family: {family}")
    except Exception as exc:
        st.error(f"Failed to load document details: {exc}")


def _render_activity(docs: list[dict]):
    active_docs = [
        doc
        for doc in docs
        if doc.get("status") in {"queued", "processing", "running", "failed"}
    ]
    if not active_docs:
        st.info("No active ingestion jobs right now.")
        return

    st.caption("Active ingestion jobs")
    for doc in active_docs:
        label, icon = _status_chip(doc.get("status", ""))
        with st.container(border=True):
            st.markdown(f"**{doc.get('name', 'Unknown document')}**")
            st.badge(label, icon=icon, color="blue")
            st.caption(f"Uploaded: {_format_dt(doc.get('created_at'))}")


def render_documents():
    render_page_shell(
        "Add source material for chat.",
        "Queue uploads instantly, monitor indexing in background, and manage document lifecycle.",
        "Documents",
        icon="folder",
    )
    api = get_api()

    client_options, client_names = get_client_options()
    if not client_names:
        st.info("Create a client before uploading documents.")
        return

    st.session_state.setdefault("documents_active_client_name", client_names[0])
    if st.session_state["documents_active_client_name"] not in client_names:
        st.session_state["documents_active_client_name"] = client_names[0]
    st.session_state.setdefault(
        "documents_active_client_id",
        client_options[st.session_state["documents_active_client_name"]],
    )
    st.session_state["documents_active_client_id"] = client_options[
        st.session_state["documents_active_client_name"]
    ]
    if st.session_state.get("documents_workspace_selector") not in client_names:
        st.session_state["documents_workspace_selector"] = st.session_state[
            "documents_active_client_name"
        ]
    selected_name = st.selectbox(
        "Client workspace",
        client_names,
        key="documents_workspace_selector",
    )
    if selected_name != st.session_state["documents_active_client_name"]:
        st.session_state["documents_active_client_name"] = selected_name
        st.session_state["documents_active_client_id"] = client_options[selected_name]
        st.rerun()

    active_client_name = st.session_state["documents_active_client_name"]
    active_client_id = st.session_state["documents_active_client_id"]
    st.caption(f"Active workspace: **{active_client_name}**")

    upload_tab, activity_tab, library_tab = st.tabs(["Upload", "Activity", "Library"])

    with upload_tab:
        with st.container(border=True):
            st.markdown("#### :material/upload_file: Upload files")
            st.caption("Supported formats: PDF, DOCX, PPTX.")
            uploaded_files = st.file_uploader(
                "Drop source files here",
                accept_multiple_files=True,
                type=["pdf", "docx", "pptx"],
                key="document_uploads",
                label_visibility="collapsed",
            )

            if st.button(
                "Queue ingestion",
                type="primary",
                icon=":material/upload:",
                key="upload_and_ingest",
                width="stretch",
            ):
                if not uploaded_files:
                    st.error("Select at least one file to upload.")
                else:
                    queued = 0
                    failed = 0
                    with st.status("Queueing uploads...", expanded=True):
                        for uploaded_file in uploaded_files:
                            st.write(f":material/description: {uploaded_file.name}")
                            try:
                                content = uploaded_file.read()
                                api.upload_document(
                                    active_client_id,
                                    uploaded_file.name,
                                    content,
                                )
                                queued += 1
                            except Exception as exc:
                                failed += 1
                                st.error(f"{uploaded_file.name} failed: {exc}")

                    bump_cache_revision(DOCUMENTS_CACHE_KEY)
                    if queued:
                        st.success(f"Queued {queued} document(s) for background indexing.")
                    if failed:
                        st.warning(f"{failed} document(s) failed to queue.")
                    st.rerun()

    with activity_tab:
        with st.container(border=True):
            st.markdown("#### :material/sync: Ingestion activity")

            @st.fragment(run_every=f"{settings.UI_POLL_INTERVAL_SECONDS}s")
            def _live_activity_fragment():
                latest_docs = get_documents(active_client_id, force_refresh=True)
                _render_activity(latest_docs)

            _live_activity_fragment()

    with library_tab:
        docs = get_documents(active_client_id)
        with st.container(border=True):
            st.markdown(f"#### :material/folder: Document library // {len(docs)}")
            if st.button(
                "Refresh library",
                icon=":material/refresh:",
                width="stretch",
                key="refresh_documents",
            ):
                bump_cache_revision(DOCUMENTS_CACHE_KEY)
                st.rerun()

            if not docs:
                st.info("No documents uploaded for this client yet.")
                return

            for doc in docs:
                status_label, icon = _status_chip(doc.get("status", ""))
                with st.container(border=True):
                    header_cols = st.columns([0.6, 0.4], vertical_alignment="center")
                    with header_cols[0]:
                        st.markdown(f"**{doc['name']}**")
                        st.caption(
                            f"{doc.get('file_type', '')} · Uploaded {_format_dt(doc.get('created_at'))}"
                        )
                    with header_cols[1]:
                        st.badge(status_label, icon=icon, color="blue")

                    action_cols = st.columns(3)
                    with action_cols[0]:
                        if st.button(
                            "Details",
                            icon=":material/info:",
                            key=f"details_{doc['id']}",
                            width="stretch",
                        ):
                            show_document_details_dialog(api, doc["id"], doc["name"])
                    with action_cols[1]:
                        if st.button(
                            "Retry",
                            icon=":material/refresh:",
                            key=f"retry_{doc['id']}",
                            width="stretch",
                            disabled=doc.get("status")
                            not in {"failed", "indexed", "completed"},
                        ):
                            with st.spinner("Retrying ingestion..."):
                                try:
                                    api.retry_document_ingestion(doc["id"])
                                    bump_cache_revision(DOCUMENTS_CACHE_KEY)
                                    st.success("Ingestion retried successfully.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Retry failed: {e}")
                    with action_cols[2]:
                        if st.button(
                            "Delete",
                            icon=":material/delete:",
                            key=f"delete_{doc['id']}",
                            width="stretch",
                        ):
                            confirm_delete_dialog(api, doc["id"], doc["name"])
