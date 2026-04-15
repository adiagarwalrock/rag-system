import pandas as pd
import streamlit as st

from ui.components.auth import get_api
from ui.components.layout import render_page_shell


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
                st.success(f"{client_name} deleted.")
                st.session_state.pop("clients", None)
                st.session_state.pop("chat_history", None)
                st.session_state.pop("chat_history_by_client", None)
                st.session_state.pop("documents_client", None)
                st.session_state.pop("query_history_client_filter", None)
                st.session_state.pop("query_history_filter_signature", None)
            except Exception as e:
                st.error(f"Failed to delete client: {e}")
        st.rerun()


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

                with st.expander(":material/delete_forever: Delete a client"):
                    delete_options = {c["name"]: c["id"] for c in clients}
                    selected_name = st.selectbox(
                        "Client workspace to delete",
                        list(delete_options.keys()),
                        key="delete_client_select",
                    )
                    st.warning(
                        "This action is irreversible. All documents and vector indexes for this client will be removed."
                    )
                    if st.button(
                        "Delete selected client",
                        icon=":material/delete:",
                        type="primary",
                        width="stretch",
                        key="delete_selected_client",
                    ):
                        confirm_client_delete_dialog(
                            api=api,
                            client_id=delete_options[selected_name],
                            client_name=selected_name,
                        )
            except Exception as e:
                msg = str(e)
                if "Connection" in msg:
                    st.warning("Cannot connect to backend. Is the server running?")
                else:
                    st.error(f"Failed to load clients: {msg}")
