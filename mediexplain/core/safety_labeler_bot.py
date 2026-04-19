import json
import os
import re
from openai import OpenAI

try:
    import streamlit as st
except:
    st = None


# -------------------------------------------------------------------
# CLIENT SETUP
# -------------------------------------------------------------------
def _client():
    api_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if st else None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found.")
    return OpenAI(api_key=api_key)

client = _client()


# -------------------------------------------------------------------
# CLEAN JSON HELPERS
# -------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """
    Minimal cleaning without altering actual JSON content.
    """
    text = text.replace("```json", "").replace("```", "").strip()
    return re.sub(r"[\x00-\x1f\x7f]", " ", text)


def _safe_json_extract(text: str):
    """
    Safely extract JSON from mixed text using first { ... } block.
    """
    text = _clean_text(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Safety Bot: No JSON object found.")
    block = match.group(0)
    return json.loads(block)


# -------------------------------------------------------------------
# SAFETY LABELER BOT CORE
# -------------------------------------------------------------------
def label_safety_llm(patient_record: dict) -> dict:
    """
    Safety Labeler Bot:
    Flags:
      - Medication risks
      - Drug–drug interactions
      - Abnormal labs
      - Abnormal vitals
      - Radiology/Path red flags
      - Procedure complications
      - Diagnosis severity
      - Acute timeline events

    Returns:
      { "safety_labels": { ... } }
    """

    record_str = json.dumps(patient_record, ensure_ascii=False)[:15000]

    prompt = f"""
You are a senior clinical risk auditor reviewing a synthetic EMR.

Your task is to assign **high-risk labels** anywhere risk appears:
- Dangerous medications
- Drug interactions
- Abnormal labs (critical high/low)
- Abnormal vitals (critical ranges)
- Radiology red-flags (acute findings)
- Pathology red-flags (aggressive or malignant processes)
- Procedures with complications
- Severe diagnosis indicators
- Timeline showing acute deterioration

Return ONLY valid JSON in EXACTLY this structure:

{{
 "safety_labels": {{
    "medication_risks": [],
    "interaction_risks": [],
    "lab_risks": [],
    "vital_risks": [],
    "pathology_risks": [],
    "radiology_risks": [],
    "procedure_risks": [],
    "global_warnings": []
 }}
}}
    
PATIENT RECORD (truncate for context):
{record_str}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2000
    )

    raw = response.output_text or ""
    return _safe_json_extract(raw)
