"""
Global CSS styles for MediExplain.
Call inject_global_css() at the top of every page script.
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── BASE ─────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0A0F1E !important;
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

/* Background radial glow */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 15% 40%, rgba(0,212,255,0.04) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 15%, rgba(0,102,255,0.05) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
}

/* ── TYPOGRAPHY ───────────────────────────────────────── */
h1, h2, h3,
[data-testid="stHeading"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #F1F5F9 !important;
}

p, li, span, div {
    font-family: 'Inter', sans-serif;
}

/* ── SIDEBAR ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(6, 10, 22, 0.98) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.08) !important;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

[data-testid="stSidebarNav"] a[aria-selected="true"] span {
    color: #00D4FF !important;
}

/* ── HIDE STREAMLIT CHROME ────────────────────────────── */
#MainMenu          { visibility: hidden; }
footer             { visibility: hidden; }
.stDeployButton    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }

/* ── SCROLLBAR ────────────────────────────────────────── */
::-webkit-scrollbar              { width: 4px; height: 4px; }
::-webkit-scrollbar-track        { background: #0A0F1E; }
::-webkit-scrollbar-thumb        { background: rgba(0,212,255,0.25); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover  { background: #00D4FF; }

/* ── BUTTONS ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,102,255,0.08)) !important;
    border: 1px solid rgba(0,212,255,0.35) !important;
    color: #00D4FF !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(0,102,255,0.18)) !important;
    border-color: #00D4FF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,102,255,0.2)) !important;
    border-color: #00D4FF !important;
}

/* ── INPUTS ───────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
}

/* ── SELECTBOX / RADIO ────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,212,255,0.18) !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}

.stRadio > div { gap: 8px; }
.stRadio label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 8px;
    padding: 6px 14px !important;
    color: #94A3B8 !important;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stRadio label:has(input:checked) {
    background: rgba(0,212,255,0.1) !important;
    border-color: #00D4FF !important;
    color: #00D4FF !important;
}

/* ── METRICS ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,212,255,0.3);
    background: rgba(0,212,255,0.03);
}
[data-testid="stMetricValue"] {
    color: #00D4FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

/* ── TABS ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
    border: 1px solid rgba(0,212,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.1) !important;
    color: #00D4FF !important;
}

/* ── EXPANDERS ────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.015) !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(0,212,255,0.25) !important;
}

/* ── ALERTS ───────────────────────────────────────────── */
[data-testid="stAlert"][data-baseweb="notification"][aria-label="Success"] {
    background: rgba(16,185,129,0.07) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"][data-baseweb="notification"][aria-label="Error"] {
    background: rgba(239,68,68,0.07) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"][data-baseweb="notification"][aria-label="Warning"] {
    background: rgba(245,158,11,0.07) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"][data-baseweb="notification"][aria-label="Info"] {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
}

/* ── DATAFRAME ────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,255,0.1) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── SPINNER ──────────────────────────────────────────── */
.stSpinner > div > div {
    border-top-color: #00D4FF !important;
}

/* ── DIVIDER ──────────────────────────────────────────── */
hr { border-color: rgba(0,212,255,0.1) !important; }

/* ── COLUMNS ──────────────────────────────────────────── */
[data-testid="column"] { padding: 0 8px; }

/* ── DOWNLOAD BUTTON ──────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(0,212,255,0.1)) !important;
    border: 1px solid rgba(16,185,129,0.4) !important;
    color: #10B981 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(0,212,255,0.2)) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(16,185,129,0.25) !important;
}

/* ── ANIMATIONS ───────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 6px rgba(0,212,255,0.5); }
    50%      { box-shadow: 0 0 18px rgba(0,212,255,0.9), 0 0 36px rgba(0,212,255,0.2); }
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
</style>
"""


def inject_global_css() -> None:
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
