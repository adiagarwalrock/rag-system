import streamlit as st

from app.core.config import settings
from ui.lib.api import VecteraCore


@st.cache_resource
def _create_api() -> VecteraCore:
    return VecteraCore()


def get_api() -> VecteraCore:
    """Get or create the API client."""
    api = _create_api()
    # Streamlit can retain a cached instance across hot reloads while the class
    # definition has changed. Rebuild once if expected methods are missing.
    if not hasattr(api, "list_query_history") or not hasattr(api, "delete_client"):
        _create_api.clear()
        api = _create_api()
    if st.session_state.get("token"):
        api.token = st.session_state.token
    return api


def render_login():
    if not settings.AUTH_ENABLED:
        st.session_state.authenticated = True
        st.session_state.token = "dev-user"
        st.session_state.user_email = "dev@localhost"
        st.rerun()

    left, right = st.columns([0.52, 0.48], vertical_alignment="center")

    with left:
        st.caption(":material/search: RAG-System")
        st.title("Chat with trusted client documents.")
        st.write(
            "Sign in to upload source files, ask questions, and review cited "
            "answers with conflict detection."
        )

    with right:
        with st.container(border=True):
            st.markdown("#### :material/login: Sign in")
            st.caption("Use the seeded local admin account for development.")

            with st.form("login_form"):
                email = st.text_input(
                    "Email",
                    placeholder="admin@user.local",
                    key="login_email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                )
                submit = st.form_submit_button(
                    "Sign in",
                    type="primary",
                    icon=":material/login:",
                    width="stretch",
                )

                if submit:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                        return

                    api = get_api()
                    try:
                        data = api.login(email, password)
                        st.session_state.authenticated = True
                        st.session_state.token = data["access_token"]
                        st.session_state.user_email = email
                        st.rerun()
                    except Exception as e:
                        error_msg = str(e)
                        if "401" in error_msg or "Incorrect" in error_msg:
                            st.error("Invalid email or password.")
                        elif "Connection" in error_msg:
                            st.error(
                                "Cannot connect to backend. Is the server running?"
                            )
                        else:
                            st.error(f"Login failed: {error_msg}")
