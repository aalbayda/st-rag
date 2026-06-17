from __future__ import annotations

import streamlit as st

from frontend.pages_chat import render_chat

st.set_page_config(
    page_title="Ask Bob",
    page_icon=":material/forum:",
    layout="wide",
)

render_chat
