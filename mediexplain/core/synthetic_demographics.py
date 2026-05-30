import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


# ------------------------------------------------------------
# OPENAI CLIENT
# ------------------------------------------------------------
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and st is not None:
        api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


# Lazy singleton
_client_cache = None


def _client():
    global _client_cache
    if _client_cache is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and st is not None:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to .streamlit/secrets.toml or set the env var."
            )
        _client_cache = OpenAI(api_key=api_key)
    return _client_cache


# ------------------------------------------------------------
# DEMOGRAPHICS BOT (PLAIN TEXT MODE)
# ------------------------------------------------------------
def generate_demographics_llm(age: int, gender: str) -> str:
    """
    Generates synthetic patient demographics in plain text (NO JSON).
    Structured with headers for easy downstream use.
    """

    prompt = f"""
You are generating synthetic medical demographics for a fictional patient.

STRICT RULES:
- Output ONLY PLAIN TEXT.
- NO JSON.
- NO brackets, braces, or code fence formatting.
- Use section headers exactly as shown below.
- Write realistic, US-style demographic details.

OUTPUT FORMAT (TEXT):

===== PATIENT DEMOGRAPHICS =====
Name:
Age:
Gender:
Ethnicity:
MRN:
Address:
Phone:
Email:

===== INSURANCE =====
Provider:
Insurance ID:

===== EMERGENCY CONTACT =====
Name:
Relationship:
Phone:

===== SOCIAL HISTORY =====
Occupation:
Living Situation:

Now fill the template with realistic **synthetic** details.
Age: {age}
Gender: {gender}
"""

    response = _client().responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=500,
    )

    raw = response.output_text.strip()
    return raw
