import streamlit as st

from app.core.ai_provider import initialize_ai_provider
from app.core.config import validate_runtime_settings
from ui.components.layout import render_account_sidebar, require_auth

st.set_page_config(
    page_title="RAG-System",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Define pages for navigation
pg = st.navigation(
    [
        st.Page(
            "ui/pages/1_Chat.py", title="Chat", icon=":material/chat:", default=True
        ),
        st.Page(
            "ui/pages/6_Query_History.py",
            title="Query History",
            icon=":material/history:",
        ),
        st.Page(
            "ui/pages/2_Documents.py", title="Documents", icon=":material/upload_file:"
        ),
        st.Page("ui/pages/3_Clients.py", title="Clients", icon=":material/domain:"),
        st.Page("ui/pages/4_Quality.py", title="Quality", icon=":material/monitoring:"),
        st.Page(
            "ui/pages/5_Qdrant_Inspector.py",
            title="Qdrant Inspector",
            icon=":material/dataset:",
        ),
    ]
)


def main():
    validate_runtime_settings()
    initialize_ai_provider()
    require_auth()
    render_account_sidebar()
    pg.run()


if __name__ == "__main__":
    main()
