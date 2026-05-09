import streamlit as st

from ui.components.clients import render_clients

st.set_page_config(page_title="Clients", page_icon="V", layout="wide")

render_clients()
