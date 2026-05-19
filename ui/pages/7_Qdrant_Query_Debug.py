import streamlit as st

from ui.components.qdrant_query_debug import render_qdrant_query_debug

st.set_page_config(page_title="Qdrant Query Debug", page_icon="V", layout="wide")
render_qdrant_query_debug()
