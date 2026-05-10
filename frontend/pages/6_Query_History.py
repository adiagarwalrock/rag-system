import streamlit as st

from frontend.components.query_history import render_query_history

st.set_page_config(page_title="Query History", page_icon="V", layout="wide")

render_query_history()
