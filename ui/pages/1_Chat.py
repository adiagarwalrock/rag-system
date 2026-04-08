import streamlit as st

from ui.components.query import render_query

st.set_page_config(page_title="Chat", page_icon="V", layout="wide")

render_query()
