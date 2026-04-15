import streamlit as st

from app.core.config import settings


def render_runtime_sidebar() -> None:
    st.sidebar.caption(":material/tune: Runtime")
    if settings.is_openai_api_key_placeholder:
        st.sidebar.warning(
            "OpenAI API key missing or placeholder. Runtime startup validation will fail until this is configured.",
            icon=":material/warning:",
        )


def render_page_shell(
    title: str,
    subtitle: str,
    label: str | None = None,
    icon: str = "article",
) -> None:
    if label:
        st.caption(f":material/{icon}: {label.upper()}")
    st.title(title)
    st.write(subtitle)
    st.divider()
