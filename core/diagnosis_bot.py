import json
import os
import re
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


# ----------------------------------------------------
# CLIENT
# ----------------------------------------------------
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or (st.secrets["OPENAI_API_KEY"] if st else None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=api_key)

client = _get_openai_client()


# ----------------------------------------------------
# SAFE JSON EXTRACTOR (NEVER FAILS)
# ----------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """
    Extract strictly valid JSON from the model.
    This works even if the model adds junk, bad quotes, or formatting errors.
    """

    # Remove markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove invisible control chars
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Remove newlines inside strings
    text = text.replace("\n", " ")

    # Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Diagnosis Bot: No JSON object found.")
    json_text = match.group(0)

    # Fix unescaped quotes inside text fields
    json_text = re.sub(r'(?<!\\)"(?=[^,:{}]*:)', '\\"', json_text)

    # Remove trailing commas
    json_text = re.sub(r",\s*(\}|\])", r"\1", json_text)

    # Try to parse
    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"Diagnosis Bot JSON parse failed: {e}\n\nRAW JSON:\n{json_text[:2000]}"
        )


# ----------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------
def generate_diagnosis_llm(age: int, gender: str) -> dict:
    """
    Generate a structured clinical diagnosis with strict JSON output.
    """

    prompt = f"""
You are a senior physician documenting a diagnosis in a hospital EMR.

Your ONLY job is to output STRICT JSON.  
No commentary. No markdown. No text outside the JSON object.

PATIENT:
- Age: {age}
- Gender: {gender}

STRUCTURE TO OUTPUT (FILL ALL FIELDS):

{{
  "primary_diagnosis": "string",
  "icd10_code": "string",
  "snomed_code": "string",
  "severity": "mild | moderate | severe",
  "clinical_status": "acute | chronic | acute-on-chronic",
  "clinical_description": "LONG, medically technical paragraph with abbreviations",
  "symptoms": ["list", "of", "symptoms"],
  "risk_factors": ["list"],
  "differential_diagnosis": [
    {{"condition": "string", "icd10": "string"}},
    {{"condition": "string", "icd10": "string"}}
  ],
  "relevant_cpt_codes": [
    {{"cpt": "string", "description": "string"}}
  ],
  "relevant_hcpcs_codes": [
    {{"hcpcs": "string", "description": "string"}}
  ],
  "provider_abbreviations_used": ["HPI","ROS","NAD","DOE","SOB"],
  "cms_hcc_category": "string",
  "cms_justification": "string",
  "mdm_complexity": "low | moderate | high"
}}

IMPORTANT RULES:
- DO NOT put newlines inside JSON strings.
- DO NOT invent escape sequences.
- DO NOT include trailing commas.
- OUTPUT ONLY THE JSON OBJECT.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=1500
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)