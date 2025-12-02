import json
import os
from datetime import datetime
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


def generate_vitals_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> dict:
    """
    Generate a complete 15-vital-sign report with values, thresholds,
    flags, units, metadata, and interpretation.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")

    # timeline alignment
    timeline_events = timeline.get("timeline_table", [])
    if timeline_events:
        first_date = timeline_events[0]["date"]
    else:
        first_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    You are generating a highly realistic Vitals Report for a fictional patient.

    Patient:
    - Age: {age}
    - Gender: {gender}
    - Primary Diagnosis: {dx} ({icd})

    ALL vitals MUST be generated regardless of relevance to diagnosis.
    ALL 15 vitals MUST appear.

    Date alignment rule:
    - Vitals must be collected ON or AFTER the timeline start date: {first_date}

    CRITICAL RULES:
    - Output ONLY valid JSON.
    - No code fences, no commentary.
    - JSON must start with '{{' and end with '}}'.
    - Use medical language and clinician terminology.
    - Include abnormal, borderline, and critical flags where appropriate.

    REQUIRED VITALS (all 15):
      1. Heart Rate (HR)
      2. Blood Pressure (Systolic/Diastolic)
      3. Respiratory Rate (RR)
      4. Temperature
      5. Oxygen Saturation (SpO₂)
      6. Height
      7. Weight
      8. Body Mass Index (BMI)
      9. Pain Score (0–10)
      10. Blood Glucose (POC)
      11. Peak Expiratory Flow (PEF)
      12. FiO₂ (Fraction of Inspired Oxygen)
      13. Mean Arterial Pressure (MAP)
      14. Waist Circumference
      15. Level of Consciousness (AVPU or GCS)

    REQUIRED OUTPUT STRUCTURE:

    {{
      "collection_metadata": {{
         "collection_date": "YYYY-MM-DD",
         "collection_time": "HH:MM",
         "device": "string",
         "location": "string"
      }},
      "vitals": [
         {{
            "name": "string",
            "value": "string or number",
            "unit": "string",
            "reference_range": "string",
            "thresholds": {{
                "low_critical": "value or null",
                "low": "value or null",
                "high": "value or null",
                "high_critical": "value or null"
            }},
            "flag": "H | L | C | ''",
            "interpretation": "clinically technical interpretation"
         }}
      ],
      "clinical_summary": "LONG multi-paragraph provider interpretation using complex medical terminology, referencing abnormal vitals, linking findings to diagnosis, and describing clinical implications."
    }}

    REQUIREMENTS:
    - Use the patient’s age + gender to bias values.
    - Use the diagnosis to bias abnormalities (if applicable).
    - Produce realistic units, ranges, and interpretations.
    - Interpretation MUST be highly medical and difficult for non-clinicians to understand.
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2500
    )

    raw = response.output_text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
