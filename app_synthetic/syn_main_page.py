import streamlit as st

st.set_page_config(page_title="Synthetic Main Page", layout="wide")

Synthetic_App = st.Page("/workspaces/mediexplain/app_synthetic/synthetic_app.py", title="Synthetic App")
chat_app = st.Page("/workspaces/mediexplain/app_synthetic/chat_app.py", title="MediExplain Chatbot")

pages = {"Home": [Synthetic_App, chat_app]}

st.navigation(pages).run()
