import os

import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="MediExplain", layout="wide")

Synthetic_App = st.Page(
    os.path.join(_ROOT, "app_synthetic", "synthetic_app.py"),
    title="Synthetic App",
)

chat_app = st.Page(
    os.path.join(_ROOT, "app_synthetic", "chat_app.py"),
    title="MediExplain Chatbot",
)

validator_app = st.Page(
    os.path.join(_ROOT, "app_synthetic", "validator", "validator_app.py"),
    title="Validator Console",
)

pages = {
    "Home": [
        Synthetic_App,
        chat_app,
        validator_app,
    ]
}

st.navigation(pages).run()
