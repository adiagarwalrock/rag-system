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
    if not hasattr(api, "list_query_history") or not hasattr(api, "delete_client"):
        _create_api.clear()
        api = _create_api()
    return api
