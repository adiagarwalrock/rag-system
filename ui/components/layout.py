import streamlit as st

from app.core.config import settings


def auth_enabled() -> bool:
    return settings.AUTH_ENABLED


def get_current_user_id() -> str:
    if not auth_enabled():
        return "dev-user"
    return st.session_state.get("token") or "internal"


def require_auth():
    if not auth_enabled():
        st.session_state.authenticated = True
        st.session_state.token = "dev-user"
        st.session_state.user_email = "dev@localhost"
        return

    if not st.session_state.get("authenticated"):
        # Lazy import to avoid eager auth/api/db import chain during app startup.
        from ui.components.auth import render_login

        render_login()
        st.stop()


def render_account_sidebar():
    st.sidebar.caption(":material/account_circle: Account")

    if not auth_enabled():
        st.sidebar.caption("Auth disabled: dev mode")

    if settings.is_openai_api_key_placeholder:
        st.sidebar.warning(
            "OpenAI API key missing or placeholder. Runtime startup validation will fail until this is configured.",
            icon=":material/warning:",
        )

    user_email = st.session_state.get("user_email")
    if user_email:
        st.sidebar.caption(user_email)

    if auth_enabled():

        if st.sidebar.button(
            "Log out",
            icon=":material/logout:",
            key="account_logout",
            width="stretch",
        ):
            st.session_state.authenticated = False
            st.session_state.pop("token", None)
            st.session_state.pop("chat_history", None)
            st.session_state.pop("chat_history_by_client", None)
            st.rerun()


def render_page_shell(
    title: str,
    subtitle: str,
    label: str | None = None,
    icon: str = "article",
):
    if label:
        st.caption(f":material/{icon}: {label.upper()}")
    st.title(title)
    st.write(subtitle)
    st.divider()
