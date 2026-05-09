import streamlit as st

from ui.components.documents import render_documents

st.set_page_config(page_title="Documents", page_icon="V", layout="wide")

render_documents()
