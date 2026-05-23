import streamlit as st

from ui.components.status import render_status

st.set_page_config(page_title="Status", page_icon="V", layout="wide")

render_status()
