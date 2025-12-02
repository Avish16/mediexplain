import json
import os
import re
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and st is not None:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment or Streamlit secrets.")
    return OpenAI(api_key=api_key)


client = _get_openai_client()


def _safe_extract_json(text: str) -> dict:
    """Extracts and sanitizes JSON from messy LLM output."""

    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output.")
    json_text = match.group(0)

    # Fix invalid quotes
    json_text = re.sub(r'(?<!\\)"(?=[^,:{}]*:)', '\\"', json_text)
    json_text = re.sub(r'"\s*\n\s*"', ' ', json_text)
    json_text = json_text.replace('\\""', '"').replace('"\\"', '"')

    try:
        return json.loads(json_text)
    except:
        pass

    # Self repair
    repair_prompt = f"""
    Fix the following JSON. 
    Do NOT change values — only repair formatting, quotes, escapes.
    Return ONLY valid JSON.

    {json_text}
    """

    repaired = client.responses.create(
        model="gpt-4.1",
        input=repair_prompt,
        max_output_tokens=1200
    ).output_text

    repaired = repaired.replace("```json", "").replace("```", "").strip()
    repaired = re.sub(r"[\x00-\x1f\x7f]", " ", repaired)

    match2 = re.search(r"\{.*\}", repaired, re.DOTALL)
    if not match2:
        raise ValueError("No JSON found after repair.")
    return json.loads(match2.group(0))



def generate_timeline_llm(age: int, gender: str, diagnosis: dict) -> dict:
    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    prompt = f"""
    You are a senior clinician creating a highly detailed, multi-year, medically realistic
    clinical timeline for a fictional patient.

    Patient:
    - Age: {age}
    - Gender: {gender}
    - Primary Diagnosis: {dx}
    - ICD-10: {icd}
    - SNOMED: {snomed}

    REQUIREMENTS:
    - Output ONLY valid JSON.
    - Must include 12–30 timeline events.
    - Timeline must span 1–5 years.
    - Events must use medical terminology, acronyms, CMS language.
    - Include: ED visits, imaging, specialty consults, complications, readmissions,
      therapy failures, procedures with CPT codes, HCPCS injectables, labs, follow-up care.

    EVENT FIELDS:
    - date (YYYY-MM-DD)
    - location
    - event_type
    - description (3–5 sentence medical summary)
    - actions_taken (tests, meds, interventions)
    - outcome

    ALSO INCLUDE:
    "timeline_summary": a long multi-paragraph narrative describing disease evolution,
    complications, imaging findings, medication adjustments, treatment response,
    and overall clinical course.

    JSON FORMAT:

    {{
      "timeline_summary": "long narrative",
      "timeline_table": [
        {{
          "date": "YYYY-MM-DD",
          "location": "string",
          "event_type": "string",
          "description": "string",
          "actions_taken": "string",
          "outcome": "string"
        }}
      ]
    }}

    Generate ONLY valid JSON.
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=3000
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
