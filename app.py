import streamlit as st

from ui.components.layout import render_account_sidebar, require_auth

st.set_page_config(
    page_title="RAG-System",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_home():
    st.caption(":material/search: SOURCE-GROUNDED DOCUMENT CHAT")
    st.title("Ask questions across client documents.")
    st.write(
        "Searches the selected client workspace, answers from uploaded "
        "files, and keeps citations and conflicts visible for review."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.subheader(":material/domain: 1. Pick a client")
            st.write("Scope every answer to one client workspace.")
            st.page_link(
                "ui/pages/3_Clients.py",
                label="Open Clients",
                icon=":material/arrow_forward:",
            )
    with col2:
        with st.container(border=True):
            st.subheader(":material/upload_file: 2. Upload documents")
            st.write("Add PDFs, DOCX files, or PPTX decks.")
            st.page_link(
                "ui/pages/2_Documents.py",
                label="Open Documents",
                icon=":material/arrow_forward:",
            )
    with col3:
        with st.container(border=True):
            st.subheader(":material/chat: 3. Chat with sources")
            st.write("Review latency, citations, and conflicts.")
            st.page_link(
                "ui/pages/1_Chat.py", label="Open Chat", icon=":material/arrow_forward:"
            )
    st.divider()
    col4, _, _, _ = st.columns(4)
    with col4:
        with st.container(border=True):
            st.subheader(":material/monitoring: 4. Measure quality")
            st.write("Run retrieval and ingestion evals, then inspect live telemetry.")
            st.page_link(
                "ui/pages/4_Quality.py",
                label="Open Quality",
                icon=":material/arrow_forward:",
            )
    st.info(
        "Keep answers source-grounded: upload documents first, then ask from Chat.",
        icon=":material/info:",
    )


# Define pages for navigation
pg = st.navigation(
    [
        st.Page(
            "ui/pages/1_Chat.py", title="Chat", icon=":material/chat:", default=True
        ),
        st.Page(
            "ui/pages/2_Documents.py", title="Documents", icon=":material/upload_file:"
        ),
        st.Page("ui/pages/3_Clients.py", title="Clients", icon=":material/domain:"),
        st.Page("ui/pages/4_Quality.py", title="Quality", icon=":material/monitoring:"),
    ]
)


def main():
    require_auth()
    render_account_sidebar()
    pg.run()


if __name__ == "__main__":
    main()
