import streamlit as st

# No need for sys.path hacks here – Streamlit Cloud runs from repo root

st.set_page_config(page_title="MediExplain", layout="wide")

# Use **relative paths from the repo root**:
# these must match exactly how they appear in GitHub
Synthetic_App = st.Page(
    "app_synthetic/synthetic_app.py",
    title="Synthetic App",
)

chat_app = st.Page(
    "app_synthetic/chat_app.py",
    title="MediExplain Chatbot",
)

validator_app = st.Page(
    "app_synthetic/validator/validator_app.py",
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
