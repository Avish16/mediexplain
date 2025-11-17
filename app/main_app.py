import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
from app.safety.consent import consent_check
from app.router import RouterBot   # NEW LINE

st.set_page_config(page_title="MediExplain", layout="centered")

st.title("MediExplain – Medical Information Simplifier")

# Consent
if not consent_check():
    st.warning("Please check the box above to continue.")
    st.stop()

router = RouterBot()   # NEW LINE

user_input = st.text_area("Paste any medical text or question:")

if st.button("Explain"):
    if user_input.strip() == "":
        st.error("Please enter text.")
    else:
        response = router.route(user_input)
        st.write("### Response")
        st.write(response)
