import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Synthetic Main Page", layout="wide")

Synthetic_App = st.Page(
    "/workspaces/mediexplain/app_synthetic/synthetic_app.py",
    title="Synthetic App"
)

chat_app = st.Page(
    "/workspaces/mediexplain/app_synthetic/chat_app.py",
    title="MediExplain Chatbot"
)

validator_app = st.Page(
    "/workspaces/mediexplain/app_synthetic/validator/validator_app.py",
    title="Validator Console",
)

pages = {
    "Home": [
        Synthetic_App,
        chat_app,
        validator_app,   # ← ADD THIS LINE
    ]
}

st.navigation(pages).run()
