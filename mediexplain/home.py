"""
MediExplain Home / Landing Page
Uses native Streamlit layout with simple single-line HTML only.
"""
import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from styles import inject_global_css

inject_global_css()

# ── HERO ─────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; color:#00D4FF; font-size:11px; letter-spacing:3px;"
    " font-weight:600; text-transform:uppercase; margin-bottom:4px;'>"
    "CLINICAL AI PLATFORM</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='text-align:center; font-family:Space Grotesk,sans-serif;"
    " background:linear-gradient(135deg,#F1F5F9 0%,#00D4FF 60%,#0066FF 100%);"
    " -webkit-background-clip:text; -webkit-text-fill-color:transparent;"
    " background-clip:text; font-size:clamp(36px,5vw,64px); margin:8px 0;'>"
    "MediExplain</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#94A3B8; font-size:18px; margin:0 0 8px;'>"
    "AI-Powered Medical Intelligence Platform</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:13px;'>"
    "19 Specialized AI Bots &nbsp;·&nbsp; RAG-Enhanced Retrieval &nbsp;·&nbsp;"
    " Clinical-Grade PDF &nbsp;·&nbsp; Real-Time Safety Labeling</p>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

# ── FEATURE CARDS ─────────────────────────────────────────────────
c1, c2, c3 = st.columns(3, gap="medium")

CARD_STYLE = (
    "border:1px solid rgba(0,212,255,0.18); border-radius:14px;"
    " background:rgba(255,255,255,0.025); padding:24px 20px;"
)

with c1:
    st.markdown(
        f"<div style='{CARD_STYLE}'>",
        unsafe_allow_html=True,
    )
    st.markdown("### 🧬 Synthetic Generator")
    st.markdown(
        "Generate complete, clinically realistic patient records end-to-end — "
        "demographics, labs, vitals, clinical notes, prescriptions, billing, and a full PDF."
    )
    st.markdown(
        "<span style='color:#00D4FF;font-size:12px;'>19 Bots</span>"
        " &nbsp; <span style='color:#64748B;font-size:12px;'>PDF Export</span>"
        " &nbsp; <span style='color:#64748B;font-size:12px;'>Safety Audit</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown(
        f"<div style='{CARD_STYLE}'>",
        unsafe_allow_html=True,
    )
    st.markdown("### 💬 Medical Chatbot")
    st.markdown(
        "Upload any medical report PDF and ask questions in plain language. "
        "Specialist bots cover labs, medications, care plans, and emotional support — "
        "powered by RAG retrieval and long-term memory."
    )
    st.markdown(
        "<span style='color:#6366F1;font-size:12px;'>RAG-Enhanced</span>"
        " &nbsp; <span style='color:#64748B;font-size:12px;'>7 Specialist Bots</span>"
        " &nbsp; <span style='color:#64748B;font-size:12px;'>Memory</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown(
        f"<div style='{CARD_STYLE}'>",
        unsafe_allow_html=True,
    )
    st.markdown("### 🔬 Validator Console")
    st.markdown(
        "Developer-facing diagnostics console for inspecting retrieval quality, "
        "routing decisions, safety guardrails, bot outputs, and Q&A history. "
        "Full pipeline introspection in one view."
    )
    st.markdown(
        "<span style='color:#10B981;font-size:12px;'>Diagnostics</span>"
        " &nbsp; <span style='color:#64748B;font-size:12px;'>RAG Audit</span>"
        " &nbsp; <span style='color:#64748B;font-size:12px;'>Safety Trace</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ── ARCHITECTURE ──────────────────────────────────────────────────
st.markdown(
    "<div style='border:1px solid rgba(0,212,255,0.12); border-radius:12px;"
    " background:rgba(255,255,255,0.02); padding:20px 24px;'>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#00D4FF; font-size:11px; font-weight:600; letter-spacing:2px;"
    " text-transform:uppercase; margin:0 0 10px;'>PLATFORM ARCHITECTURE</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "**PDF / Input** &nbsp;→&nbsp; **Router Bot** &nbsp;→&nbsp; **RAG Retrieval**"
    " &nbsp;→&nbsp; **Specialist Bot** &nbsp;→&nbsp; **Safety Check** &nbsp;→&nbsp; **Response**"
)
st.markdown(
    "<p style='color:#475569; font-size:12px; margin:4px 0 0;'>"
    "Intent detection · Vector search · Domain expertise · Clinical guardrails · Plain-language output</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── QUICK START ───────────────────────────────────────────────────
st.markdown("### Quick Start")

q1, q2, q3 = st.columns(3, gap="medium")
with q1:
    with st.container(border=True):
        st.markdown("**1 · Set API Key**")
        st.caption("Add `OPENAI_API_KEY` to `.streamlit/secrets.toml`")
with q2:
    with st.container(border=True):
        st.markdown("**2 · Choose a Module**")
        st.caption("Select Synthetic Generator, Chatbot, or Validator from the sidebar")
with q3:
    with st.container(border=True):
        st.markdown("**3 · Generate or Chat**")
        st.caption("Run the full pipeline or upload a medical PDF and start asking questions")

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# ── DISCLAIMER ────────────────────────────────────────────────────
st.warning(
    "**Research & Educational Use Only** — MediExplain generates **synthetic, fictional** "
    "medical records for research and educational purposes only. All patient data is entirely "
    "AI-generated. This platform is **not a substitute for professional medical advice**, "
    "diagnosis, or treatment. Always consult a qualified healthcare professional.",
    icon="⚠️",
)
