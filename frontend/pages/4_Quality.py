import streamlit as st

from frontend.components.quality import render_quality

st.set_page_config(page_title="Quality", page_icon="V", layout="wide")

render_quality()
