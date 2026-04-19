import json
import os
import re
from openai import OpenAI

try:
    import streamlit as st
except:
    st = None


# ----------------------------------------------------
# OPENAI CLIENT
# ----------------------------------------------------
def _get_client():
    api_key = os.getenv("OPENAI_API_KEY") or (st.secrets["OPENAI_API_KEY"] if st else None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=api_key)

client = _get_client()


# ----------------------------------------------------
# SUPER SAFE JSON EXTRACTION
# ----------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """
    Extract JSON from an LLM response safely:
    - remove code fences
    - remove control chars
    - extract first { ... } block
    - remove trailing commas
    - fallback to empty structure instead of crashing pipeline
    """
    if not text:
        return {
            "consistency_report": {
                "errors": ["Empty response from LLM"],
                "warnings": [],
                "suggested_fixes": []
            }
        }

    # Remove markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Kill invisible characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "consistency_report": {
                "errors": ["No JSON object found in LLM output"],
                "warnings": [],
                "suggested_fixes": []
            }
        }

    json_text = match.group(0)

    # Remove trailing commas
    json_text = re.sub(r",\s*(\])", r"\1", json_text)
    json_text = re.sub(r",\s*(\})", r"\1", json_text)

    try:
        return json.loads(json_text)
    except Exception:
        # fail-safe fallback
        return {
            "consistency_report": {
                "errors": ["JSON parsing failed"],
                "warnings": [],
                "suggested_fixes": []
            }
        }


# ----------------------------------------------------
# MAIN BOT
# ----------------------------------------------------
def check_consistency_llm(patient_record: dict) -> dict:
    """
    Uses GPT to detect contradictions across the full patient record.
    This version NEVER breaks the pipeline.
    """

    # Limit to avoid runaway token cost
    record_str = json.dumps(patient_record, ensure_ascii=False)[:15000]

    prompt = f"""
You are a senior clinical auditor reviewing a synthetic EMR for internal consistency.
Identify contradictions across:

- diagnosis vs pathology
- labs vs vitals
- timeline vs radiology vs procedures
- medications vs renal/hepatic labs
- age vs reported comorbidities
- procedure dates vs diagnosis dates
- imaging findings vs clinical notes

Return ONLY valid JSON:

{{
  "consistency_report": {{
    "errors": ["major contradictions"],
    "warnings": ["minor inconsistencies"],
    "suggested_fixes": ["how to resolve issues"]
  }}
}}

PATIENT RECORD BELOW:
{record_str}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=1500
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
