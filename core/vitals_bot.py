import json
import os
import re
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


# ============================================================
# SUPER-ROBUST JSON EXTRACTOR (same strength as Lab Bot)
# ============================================================

def _safe_extract_json(text: str) -> dict:
    """
    Extremely robust JSON extractor for Vitals Bot.
    Fixes:
    - control chars
    - unescaped quotes
    - newlines inside strings
    - trailing commas
    - markdown fences
    - partial blocks
    """

    # Remove markdown leftovers
    text = text.replace("```json", "").replace("```", "")
    text = text.strip()

    # Remove all control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Replace raw newlines inside text
    text = text.replace("\n", " ")

    # Find ANY JSON block { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Vitals Bot: No JSON block found.")

    json_text = match.group(0)

    # Escape unescaped quotes inside strings
    json_text = re.sub(r'(?<!\\)"(?=[^:{},\]\[])', '\\"', json_text)

    # Remove trailing commas
    json_text = re.sub(r",\s*(\}|\])", r"\1", json_text)

    # FINAL ATTEMPT
    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"\n❌ Vitals Bot JSON Clean Failed: {e}\n"
            f"------- RAW START -------\n"
            f"{json_text[:3500]}\n"
            f"------- RAW END ---------"
        )



# ============================================================
# MAIN BOT
# ============================================================

def generate_vitals_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> dict:
    """
    Generate a full vitals set — 15 vitals, metadata, flags, interpretation.
    Fully JSON-safe after sanitizer.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")

    # timeline alignment
    events = timeline.get("timeline_table", [])
    if events:
        first_date = events[0]["date"]
    else:
        first_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    You are generating a highly realistic Vitals Report.

    Patient:
    - Age: {age}
    - Gender: {gender}
    - Primary Diagnosis: {dx} ({icd})

    RULES:
    - Output ONLY valid JSON.
    - DO NOT use newlines inside strings.
    - JSON must begin with '{{' and end with '}}'.

    REQUIRED 15 VITALS:
      1. Heart Rate
      2. Blood Pressure
      3. Respiratory Rate
      4. Temperature
      5. SpO₂
      6. Height
      7. Weight
      8. BMI
      9. Pain Score
      10. Blood Glucose
      11. PEF
      12. FiO₂
      13. MAP
      14. Waist Circumference
      15. Level of Consciousness (AVPU/GCS)

    JSON FORMAT:

    {{
      "collection_metadata": {{
         "collection_date": "{first_date}",
         "collection_time": "08:32",
         "device": "Automated vitals monitor",
         "location": "Inpatient room"
      }},
      "vitals": [
         {{
            "name": "string",
            "value": "number/string",
            "unit": "string",
            "reference_range": "string",
            "thresholds": {{
                "low_critical": "value or null",
                "low": "value or null",
                "high": "value or null",
                "high_critical": "value or null"
            }},
            "flag": "H | L | C | ''",
            "interpretation": "short clinician-style interpretation (no newlines)"
         }}
      ],
      "clinical_summary": "LONG multi-paragraph summary using \\n for line breaks."
    }}

    Produce ONLY JSON. No extra commentary.
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2500,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
