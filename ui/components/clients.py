from __future__ import annotations

from datetime import datetime

import streamlit as st

from ui.components.api_client import get_api
from ui.components.layout import render_page_shell
from ui.components.utils import (
    CLIENTS_CACHE_KEY,
    DOCUMENTS_CACHE_KEY,
    QUERY_HISTORY_CACHE_KEY,
    bump_cache_revision,
    get_clients,
)


def _format_dt(value: str | None) -> str:
    if not value:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


@st.dialog("Confirm Client Deletion")
def confirm_client_delete_dialog(api, client_id: str, client_name: str):
    st.error(
        "Deleting this client permanently removes its documents, vector data, and query history."
    )
    st.caption(f"Client workspace: `{client_name}`")
    st.caption("Type `DELETE` to confirm.")

    confirmation = st.text_input(
        "Confirmation",
        placeholder="DELETE",
        key=f"delete_client_confirmation_{client_id}",
    )

    if st.button(
        "Delete client workspace",
        type="primary",
        icon=":material/delete_forever:",
        width="stretch",
        disabled=confirmation != "DELETE",
        key=f"confirm_delete_client_{client_id}",
    ):
        with st.spinner("Deleting client workspace..."):
            try:
                api.delete_client(client_id)
                bump_cache_revision(CLIENTS_CACHE_KEY)
                bump_cache_revision(DOCUMENTS_CACHE_KEY)
                bump_cache_revision(QUERY_HISTORY_CACHE_KEY)
                st.session_state.pop("chat_history", None)
                st.session_state.pop("chat_history_by_client", None)
                for key in list(st.session_state.keys()):
                    if key.startswith("query_active_session_"):
                        st.session_state.pop(key, None)
                st.success(f"{client_name} deleted.")
            except Exception as e:
                st.error(f"Failed to delete client: {e}")
        st.rerun()


def render_clients():
    render_page_shell(
        "Manage workspaces.",
        "Create a client workspace, then upload documents and query within that scope.",
        "Clients",
        icon="domain",
    )
    api = get_api()

    with st.container(border=True):
        st.markdown("#### :material/add_business: New client")

        try:
            embedding_model_entries = api.list_embedding_models()
        except Exception:
            embedding_model_entries = []

        with st.form("create_client_form", clear_on_submit=True):
            client_name = st.text_input("Client name")
            client_desc = st.text_area(
                "Description", placeholder="Optional context", height=94
            )

            if embedding_model_entries:
                # API returns models already sorted openai → gemini → other
                model_ids = [e["id"] for e in embedding_model_entries]
                model_meta = {e["id"]: e for e in embedding_model_entries}

                selected_embedding_model = st.selectbox(
                    "Embedding model",
                    options=model_ids,
                    format_func=lambda x: model_meta[x].get("display_name") or x,
                    help="Documents for this client are indexed and queried with this model.",
                )

                if selected_embedding_model and selected_embedding_model in model_meta:
                    m = model_meta[selected_embedding_model]
                    provider_label = (m.get("provider") or "").upper()
                    dims = m.get("dimensions")
                    hint = f"{provider_label} · {dims:,}d" if dims else provider_label
                    st.caption(hint)
            else:
                selected_embedding_model = None

            if st.form_submit_button(
                "Create client",
                type="primary",
                icon=":material/add:",
                width="stretch",
            ):
                if not client_name.strip():
                    st.error("Enter a client name.")
                else:
                    try:
                        result = api.create_client(
                            client_name.strip(),
                            client_desc.strip(),
                            embedding_model=selected_embedding_model,
                        )
                        bump_cache_revision(CLIENTS_CACHE_KEY)
                        st.success(f"{result['name']} created.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create client: {e}")

    st.markdown("#### :material/list_alt: Existing clients")
    toolbar = st.columns([0.75, 0.25], vertical_alignment="bottom")
    with toolbar[1]:
        if st.button(
            "Refresh",
            icon=":material/refresh:",
            width="stretch",
            key="refresh_clients",
        ):
            bump_cache_revision(CLIENTS_CACHE_KEY)
            st.rerun()

    try:
        clients = get_clients()
    except Exception as e:
        msg = str(e)
        if "Connection" in msg:
            st.warning("Cannot connect to backend. Is the server running?")
        else:
            st.error(f"Failed to load clients: {msg}")
        return

    if not clients:
        st.info("No clients found. Create one to start uploading documents.")
        return

    st.caption(f"Client workspaces: **{len(clients)}**")

    for client in clients:
        with st.container(border=True):
            header_cols = st.columns([0.75, 0.25], vertical_alignment="center")
            with header_cols[0]:
                st.markdown(f"**{client['name']}**")
                st.caption(client.get("description") or "No description")
            with header_cols[1]:
                if st.button(
                    "Delete",
                    icon=":material/delete:",
                    width="stretch",
                    key=f"delete_client_{client['id']}",
                ):
                    confirm_client_delete_dialog(
                        api=api,
                        client_id=client["id"],
                        client_name=client["name"],
                    )

            info_cols = st.columns(3)
            info_cols[0].caption(f"Created: {_format_dt(client.get('created_at'))}")
            info_cols[1].caption(f"ID: `{client['id']}`")
            embed = client.get("embedding_model") or "default"
            info_cols[2].caption(f"Embed: `{embed}`")
