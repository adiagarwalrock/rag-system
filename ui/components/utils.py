import streamlit as st

from ui.components.auth import get_api


def get_client_options():
    """Get client list for dropdowns with session state caching."""
    clients = st.session_state.get("clients", [])
    if not clients:
        try:
            api = get_api()
            clients = api.list_clients()
            st.session_state.clients = clients
        except Exception:
            return {}, []

    options = {c["name"]: c["id"] for c in clients}
    return options, list(options.keys())
