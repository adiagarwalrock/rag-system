import pandas as pd
import streamlit as st

from ui.components.auth import get_api
from ui.components.layout import get_current_user_id, render_page_shell
from ui.components.utils import get_client_options


@st.dialog("Confirm Deletion")
def confirm_delete_dialog(api, doc_id):
    st.warning(
        "Are you sure you want to delete this document? This removes the raw file and all vector references."
    )
    if st.button("Yes, delete document", type="primary", width="stretch"):
        with st.spinner("Deleting document..."):
            try:
                api.delete_document(doc_id, hard=True, user_id=get_current_user_id())
                st.success("Document deleted.")
            except Exception as e:
                st.error(f"Failed to delete document: {e}")
        st.rerun()


def render_documents():
    render_page_shell(
        "Add source material for chat.",
        "Upload client files, confirm ingestion status, then return to Chat for source-grounded answers.",
        "Documents",
        icon="folder",
    )
    api = get_api()

    client_options, client_names = get_client_options()
    if not client_names:
        st.info("Create a client before uploading documents.")
        return

    toolbar = st.columns([0.72, 0.28], vertical_alignment="bottom")
    with toolbar[0]:
        selected_name = st.selectbox(
            "Client workspace", client_names, key="documents_client"
        )
        selected_client_id = client_options[selected_name]
    with toolbar[1]:
        if st.button(
            "Refresh documents",
            icon=":material/refresh:",
            key="refresh_documents",
            width="stretch",
        ):
            st.rerun()

    upload_tab, library_tab = st.tabs(["Upload files", "Document library"])

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
                "Upload and ingest",
                type="primary",
                icon=":material/upload:",
                key="upload_and_ingest",
                width="stretch",
            ):
                if not uploaded_files:
                    st.error("Select at least one file to upload.")
                else:
                    progress = st.progress(0)
                    with st.status("Ingesting documents...", expanded=True) as status:
                        failures = 0
                        for i, uploaded_file in enumerate(uploaded_files):
                            st.write(f":material/description: {uploaded_file.name}")
                            try:
                                content = uploaded_file.read()
                                result = api.upload_document(
                                    selected_client_id,
                                    uploaded_file.name,
                                    content,
                                    user_id=get_current_user_id(),
                                )
                                st.success(
                                    f"{uploaded_file.name} ingested as {result.get('status', 'done')}."
                                )
                            except Exception as exc:
                                failures += 1
                                st.error(f"{uploaded_file.name} failed: {exc}")
                            progress.progress((i + 1) / len(uploaded_files))

                        status.update(
                            label=(
                                "Ingestion completed with errors."
                                if failures
                                else "Ingestion complete."
                            ),
                            state="error" if failures else "complete",
                            expanded=bool(failures),
                        )

    with library_tab:
        with st.container(border=True):
            st.markdown("#### :material/folder: Document library")
            try:
                docs = api.list_documents(selected_client_id)
                if not docs:
                    st.info("No documents uploaded for this client yet.")
                    return

                df = pd.DataFrame(docs)
                status_counts = df["status"].fillna("unknown").value_counts().to_dict()

                with st.container(border=True):
                    stats = st.columns(3)
                    stats[0].metric(":material/description: Documents", len(docs))
                    ready = status_counts.get("completed", 0) + status_counts.get(
                        "indexed", 0
                    )
                    proc = status_counts.get("processing", 0) + status_counts.get(
                        "running", 0
                    )
                    stats[1].metric(":material/check_circle: Ready", ready)
                    stats[2].metric(":material/sync: Processing", proc)

                cols = ["name", "file_type", "status", "document_family", "created_at"]
                display_df = df[[c for c in cols if c in df.columns]]

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                    key="documents_table",
                    column_config={
                        "name": st.column_config.TextColumn("Document"),
                        "file_type": st.column_config.TextColumn("Type"),
                        "status": st.column_config.TextColumn("Status"),
                        "document_family": st.column_config.TextColumn("Family"),
                        "created_at": st.column_config.TextColumn("Uploaded"),
                    },
                )

                with st.expander(":material/info: Check document details"):
                    doc_options = {d["name"]: d["id"] for d in docs}
                    selected_doc = st.selectbox(
                        "Document",
                        list(doc_options.keys()),
                        key="document_status_select",
                    )
                    selected_doc_id = doc_options[selected_doc]

                    status = api.get_document_status(selected_doc_id)
                    with st.container(border=True):
                        c1, c2, c3 = st.columns(3)
                        c1.metric(":material/flag: Status", status.get("status", "?"))
                        c2.metric(
                            ":material/reorder: Vector points",
                            status.get("vector_point_count", 0),
                        )
                        c3.metric(
                            ":material/history: Version",
                            status.get("version_label") or "-",
                        )

                        if err := status.get("error_message"):
                            st.error(err)

                        if (is_current := status.get("is_current_version")) is not None:
                            text = (
                                "Yes"
                                if is_current
                                else "No, this version was superseded"
                            )
                            icon = (
                                ":material/check_circle:"
                                if is_current
                                else ":material/cancel:"
                            )
                            st.info(f"{icon} Current version: {text}")

                        if family := status.get("version_group"):
                            st.caption(
                                f":material/family_restroom: Document family: {family}"
                            )

                    action_cols = st.columns(2)
                    with action_cols[0]:
                        if status.get("status") in [
                            "failed",
                            "indexed",
                            "completed",
                            "deleted",
                        ]:
                            if st.button(
                                "Retry Ingestion",
                                icon=":material/refresh:",
                                type="primary",
                                width="stretch",
                                key=f"retry_{selected_doc_id}",
                            ):
                                with st.spinner("Retrying ingestion..."):
                                    try:
                                        api.retry_document_ingestion(
                                            selected_doc_id,
                                            user_id=get_current_user_id(),
                                        )
                                        st.success("Ingestion retried successfully.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Retry failed: {e}")

                    with action_cols[1]:
                        if st.button(
                            "Delete Document",
                            icon=":material/delete:",
                            width="stretch",
                            key=f"delete_{selected_doc_id}",
                        ):
                            confirm_delete_dialog(api, selected_doc_id)

            except Exception as e:
                msg = str(e)
                if "Connection" in msg:
                    st.warning("Cannot connect to backend.")
                else:
                    st.error(f"Failed to load documents: {msg}")
