import os
import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="MediExplain – AI Medical Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS into the nav shell (sidebar always rendered by entry script)
from styles import inject_global_css
inject_global_css()

# ── Sidebar brand header ────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:16px 8px 20px; border-bottom:1px solid rgba(0,212,255,0.1); margin-bottom:12px;">
  <div style="font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700;
              background:linear-gradient(135deg,#F1F5F9,#00D4FF);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              background-clip:text; margin-bottom:4px;">
    🧬 MediExplain
  </div>
  <div style="font-size:11px; color:#475569; font-family:'Inter',sans-serif;
              letter-spacing:1px; text-transform:uppercase;">
    AI Medical Intelligence
  </div>
</div>
""", unsafe_allow_html=True)

# ── Page definitions ────────────────────────────────────────────
home_page = st.Page(
    os.path.join(_ROOT, "home.py"),
    title="Home",
    icon="🏠",
    default=True,
)

Synthetic_App = st.Page(
    os.path.join(_ROOT, "app_synthetic", "synthetic_app.py"),
    title="Synthetic Generator",
    icon="🧬",
)

chat_app = st.Page(
    os.path.join(_ROOT, "app_synthetic", "chat_app.py"),
    title="Medical Chatbot",
    icon="💬",
)

validator_app = st.Page(
    os.path.join(_ROOT, "app_synthetic", "validator", "validator_app.py"),
    title="Validator Console",
    icon="🔬",
)

pages = {
    "": [home_page],
    "Platform": [Synthetic_App, chat_app, validator_app],
}

st.navigation(pages).run()
