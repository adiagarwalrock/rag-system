from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from ui.lib.api import VecteraCore


@st.cache_resource
def _create_api() -> "VecteraCore":
    # Lazy import to avoid eager DB/network module import at module load.
    from ui.lib.api import VecteraCore

    return VecteraCore()


def get_api() -> "VecteraCore":
    """Get or create the API client."""
    api = _create_api()
    required_methods = (
        "list_query_history",
        "delete_client",
        "list_chat_sessions",
        "create_chat_session",
        "list_chat_messages",
        "clear_chat_session",
        "get_runtime_status",
    )
    if not all(hasattr(api, method_name) for method_name in required_methods):
        _create_api.clear()
        api = _create_api()
    return api
