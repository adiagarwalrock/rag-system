from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from frontend.lib.api import RAGCore


@st.cache_resource
def _create_api() -> "RAGCore":
    # Lazy import to avoid eager DB/network module import at module load.
    from frontend.lib.api import RAGCore

    return RAGCore()


def get_api() -> "RAGCore":
    """Get or create the API client."""
    api = _create_api()
    required_methods = (
        "list_query_history",
        "delete_client",
        "list_chat_sessions",
        "create_chat_session",
        "list_chat_messages",
        "clear_chat_session",
    )
    if not all(hasattr(api, method_name) for method_name in required_methods):
        _create_api.clear()
        api = _create_api()
    return api
