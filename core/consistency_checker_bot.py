import json
import os
import re
from openai import OpenAI

try:
    import streamlit as st
except:
    st = None


def _get_client():
    api_key = os.getenv("OPENAI_API_KEY") or (st.secrets["OPENAI_API_KEY"] if st else None)
    return OpenAI(api_key=api_key)


client = _get_client()


def _clean(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    return re.sub(r"[\x00-\x1f\x7f]", " ", text)


def check_consistency_llm(patient_record: dict) -> dict:
    """
    Uses GPT to detect contradictions across the full patient record.
    """

    record_str = json.dumps(patient_record, ensure_ascii=False)[:12000]  # limit to avoid overflow

    prompt = f"""
You are a senior clinical auditor reviewing a full synthetic EMR.
Your task is to identify inconsistencies and contradictions.

Examples:
- Diagnosis does not match pathology.
- Lab values impossible for age.
- Medications not indicated for diagnosis.
- Timeline inconsistent with radiology or procedures.
- Procedures happening before diagnosis.
- Vitals contradict labs.
- Pathology type not consistent with imaging.

OUTPUT ONLY VALID JSON:

{{
  "consistency_report": {{
    "errors": ["list major contradictions"],
    "warnings": ["list minor contradictions"],
    "suggested_fixes": ["suggest simple corrections to fix issues"]
  }}
}}
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=1500
    )

    raw = _clean(resp.output_text)
    out = json.loads(raw)
    return out
