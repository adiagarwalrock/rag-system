import pandas as pd
import streamlit as st

from ui.components.auth import get_api
from ui.components.layout import render_page_shell


def render_clients():
    render_page_shell(
        "Manage workspaces.",
        "Each client is a separate retrieval scope for document upload and chat.",
        "Clients",
        icon="domain",
    )
    api = get_api()

    create_tab, list_tab = st.tabs(["Create client", "Existing clients"])

    with create_tab:
        with st.container(border=True):
            st.markdown("#### :material/add_business: New client")
            with st.form("create_client_form", clear_on_submit=True):
                client_name = st.text_input("Client name")
                client_desc = st.text_area(
                    "Description", placeholder="Optional context"
                )

                if st.form_submit_button(
                    "Create client",
                    type="primary",
                    icon=":material/add:",
                    width="stretch",
                ):
                    if not client_name:
                        st.error("Enter a client name.")
                    else:
                        try:
                            result = api.create_client(client_name.strip(), client_desc)
                            st.session_state.pop("clients", None)
                            st.success(f"{result['name']} created.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create client: {e}")

    with list_tab:
        with st.container(border=True):
            st.markdown("#### :material/list_alt: Existing clients")
            try:
                clients = api.list_clients()
                if not clients:
                    st.info(
                        "No clients found. Create one to start uploading documents."
                    )
                    return

                df = pd.DataFrame(clients)
                st.metric("Client workspaces", len(clients))

                cols = ["name", "description", "created_at", "id"]
                display_df = df[[c for c in cols if c in df.columns]]

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                    key="clients_table",
                    column_config={
                        "name": st.column_config.TextColumn("Client"),
                        "description": st.column_config.TextColumn("Description"),
                        "created_at": st.column_config.TextColumn("Created"),
                        "id": st.column_config.TextColumn("ID"),
                    },
                )
                st.session_state.clients = clients
            except Exception as e:
                msg = str(e)
                if "Connection" in msg:
                    st.warning("Cannot connect to backend. Is the server running?")
                else:
                    st.error(f"Failed to load clients: {msg}")
