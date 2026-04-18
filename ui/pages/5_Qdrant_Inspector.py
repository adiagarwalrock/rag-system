import streamlit as st

from ui.components.qdrant_inspector import render_qdrant_inspector

st.set_page_config(page_title="Qdrant Inspector", page_icon="V", layout="wide")

render_qdrant_inspector()
