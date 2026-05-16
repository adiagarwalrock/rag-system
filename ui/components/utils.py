import streamlit as st

from ui.components.api_client import get_api

CLIENTS_CACHE_KEY = "clients_cache_revision"
DOCUMENTS_CACHE_KEY = "documents_cache_revision"
QUERY_HISTORY_CACHE_KEY = "query_history_cache_revision"


def _cache_revision(key: str) -> int:
    return int(st.session_state.get(key, 0))


def bump_cache_revision(key: str) -> None:
    st.session_state[key] = _cache_revision(key) + 1


@st.cache_data(show_spinner=False, ttl=20)
def _cached_clients(_revision: int) -> list[dict]:
    api = get_api()
    return api.list_clients()


@st.cache_data(show_spinner=False, ttl=10)
def _cached_documents(client_id: str, _revision: int) -> list[dict]:
    api = get_api()
    return api.list_documents(client_id)


@st.cache_data(show_spinner=False, ttl=8)
def _cached_query_history(
    client_id: str | None,
    status: str | None,
    search_text: str,
    limit: int,
    offset: int,
    _revision: int,
) -> dict:
    api = get_api()
    return api.list_query_history(
        client_id=client_id,
        status=status,
        search_text=search_text,
        limit=limit,
        offset=offset,
    )


def get_clients(force_refresh: bool = False) -> list[dict]:
    if force_refresh:
        bump_cache_revision(CLIENTS_CACHE_KEY)
    return _cached_clients(_cache_revision(CLIENTS_CACHE_KEY))


def get_client_options(force_refresh: bool = False) -> tuple[dict[str, str], list[str]]:
    try:
        clients = get_clients(force_refresh=force_refresh)
    except Exception:
        return {}, []

    options = {c["name"]: c["id"] for c in clients}
    return options, list(options.keys())


def get_documents(client_id: str, force_refresh: bool = False) -> list[dict]:
    if force_refresh:
        bump_cache_revision(DOCUMENTS_CACHE_KEY)
    return _cached_documents(client_id, _cache_revision(DOCUMENTS_CACHE_KEY))


def get_query_history(
    *,
    client_id: str | None,
    status: str | None,
    search_text: str,
    limit: int,
    offset: int,
    force_refresh: bool = False,
) -> dict:
    if force_refresh:
        bump_cache_revision(QUERY_HISTORY_CACHE_KEY)
    return _cached_query_history(
        client_id,
        status,
        search_text,
        limit,
        offset,
        _cache_revision(QUERY_HISTORY_CACHE_KEY),
    )
