import json
import os
from datetime import datetime, timedelta
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


def generate_lab_report_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> dict:
    """
    Generates a very complex, diagnosis-aware, multi-section laboratory report.
    Values depend on age, gender, diagnosis, and timeline start date.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # Determine earliest timeline date to keep labs consistent
    timeline_events = timeline.get("timeline_table", [])
    if timeline_events:
        first_date = timeline_events[0]["date"]
    else:
        first_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    You are a senior clinical pathologist generating a highly complex, synthetic laboratory report
    for a fictional patient. The report must look EXACTLY like a real hospital EMR lab module
    (EPIC/Cerner/Meditech).

    Patient:
    - Age: {age}
    - Gender: {gender}
    - Primary Diagnosis: {dx}
    - ICD-10: {icd}
    - SNOMED: {snomed}

    TIMELINE ALIGNMENT RULE:
    - Lab tests MUST be dated on or AFTER the first timeline event date: {first_date}.
    - DO NOT generate lab tests before this date.

    CRITICAL OUTPUT RULES:
    - Output ONLY valid JSON.
    - NO commentary, NO code fences, NO notes.
    - JSON must start with '{{' and end with '}}'.
    - Use REAL medical language, abbreviations, and advanced technical terms.
    - The report MUST be hard to understand for a layperson.

    REQUIRED LAB SECTIONS:
    1. CBC with differential
    2. CMP
    3. Lipid Panel
    4. Coagulation Panel (PT/INR, PTT, fibrinogen)
    5. Cardiac Markers (troponin, BNP/NT-proBNP, CK-MB)
    6. Endocrine Labs (A1c, TSH, T3/T4, cortisol)
    7. Renal Labs (GFR, urine protein, microalbumin)
    8. Infection Markers (CRP, ESR, procalcitonin, lactate)
    9. Microbiology (cultures, gram stain, interpretation)
    10. Toxicology (UDS panel or serum tox)
    11. Diagnosis-specific panels (choose based on {dx})
    12. CPT codes for each section
    13. Reference ranges
    14. Abnormal flags: H, L, C (Critical)
    15. Collection metadata (date, time, specimen type, order ID)
    16. A long detailed provider interpretation

    JSON FORMAT EXACTLY LIKE THIS:

    {{
      "collection_metadata": {{
        "collection_date": "YYYY-MM-DD",
        "collection_time": "HH:MM",
        "specimen_type": "string",
        "order_id": "string",
        "performing_lab": "string"
      }},
      "cbc": {{
         "panel_cpt": "85025",
         "tests": [
           {{
             "name": "WBC",
             "value": "numeric",
             "unit": "K/uL",
             "reference_range": "4.0–11.0",
             "flag": "H|L|C|"
           }},
           ...
         ]
      }},
      "cmp": {{
         "panel_cpt": "80053",
         "tests": [...]
      }},
      "lipid_panel": {{
         "panel_cpt": "80061",
         "tests": [...]
      }},
      "coagulation_panel": {{
         "panel_cpt": "85610/85730",
         "tests": [...]
      }},
      "cardiac_markers": {{
         "panel_cpt": "84484/83880",
         "tests": [...]
      }},
      "endocrine_labs": {{
         "panel_cpt": "83036/84443",
         "tests": [...]
      }},
      "renal_panel": {{
         "panel_cpt": "82565/82043",
         "tests": [...]
      }},
      "infection_markers": {{
         "panel_cpt": "86140/83605",
         "tests": [...]
      }},
      "microbiology": {{
         "panel_cpt": "87040/87070",
         "culture_results": "string",
         "organism_identified": "string or null",
         "sensitivity_pattern": "string",
         "gram_stain": "string"
      }},
      "toxicology": {{
         "panel_cpt": "80307",
         "tests": [...]
      }},
      "diagnosis_specific_labs": {{
         "panel_description": "string",
         "panel_cpt": "varies",
         "tests": [...]
      }},
      "interpretation_summary": "LONG, highly technical provider analysis of abnormalities, clinical implications, trends, and differential considerations."
    }}

    Make values medically consistent with:
    - Age: {age}
    - Gender: {gender}
    - Diagnosis: {dx}

    Make abnormalities reflect the primary condition.
    Use complex medical terminology.
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=4000
    )

    raw = response.output_text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
