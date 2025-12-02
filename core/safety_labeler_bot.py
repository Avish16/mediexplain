import json
import os
import re
from openai import OpenAI

try:
    import streamlit as st
except:
    st = None


def _client():
    api_key = os.getenv("OPENAI_API_KEY") or (st.secrets["OPENAI_API_KEY"] if st else None)
    return OpenAI(api_key=api_key)


client = _client()


def _clean(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    return re.sub(r"[\x00-\x1f\x7f]", " ", text)


def label_safety_llm(patient_record: dict) -> dict:
    """
    Safety Labeler Bot assigns risk tags.
    """

    rec = json.dumps(patient_record, ensure_ascii=False)[:15000]

    prompt = f"""
You are a clinical safety analyst. Review this full EMR and label ANY high-risk content.

Label risks related to:
- Medications
- Drug interactions
- Abnormal labs
- Abnormal vitals
- Radiology red flags
- Pathology red flags
- Procedures with complications
- Diagnosis severity
- Timeline acute events

OUTPUT ONLY VALID JSON:

{{
 "safety_labels": {{
    "medication_risks": [...],
    "interaction_risks": [...],
    "lab_risks": [...],
    "vital_risks": [...],
    "pathology_risks": [...],
    "radiology_risks": [...],
    "procedure_risks": [...],
    "global_warnings": [...]
 }}
}}
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2000
    )

    raw = _clean(resp.output_text)
    return json.loads(raw)
