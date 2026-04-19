import json
import os
import re
from datetime import datetime
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


# ----------------------------------------------------
# OPENAI CLIENT
# ----------------------------------------------------
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and st is not None:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment or Streamlit secrets.")
    return OpenAI(api_key=api_key)


client = _get_openai_client()


# ----------------------------------------------------
# ROBUST JSON EXTRACTOR (Medication Bot)
# ----------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """
    Extremely defensive JSON extractor for Medication Bot.
    Handles:
    - code fences
    - control characters
    - illegal escapes
    - trailing commas
    - newline contamination
    - partial double braces
    """

    if not text:
        raise ValueError("Medication Bot: Empty model output.")

    # Remove markdown
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove control chars
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Replace newlines inside JSON strings
    text = text.replace("\n", " ")

    # Fix illegal escapes like \q
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)

    # Extract the actual JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Medication Bot: Could not find a JSON object.\n"
            f"RAW START:\n{text[:1200]}\n..."
        )

    json_text = match.group(0)

    # Remove trailing commas
    json_text = re.sub(r",\s*(\})", r"\1", json_text)
    json_text = re.sub(r",\s*(\])", r"\1", json_text)

    # Remove double-double braces {{ }}
    json_text = json_text.replace("{{", "{").replace("}}", "}")

    # Collapse spaces
    json_text = re.sub(r"\s+", " ", json_text)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"\n❌ Medication Bot JSON failed: {e}\n"
            f"--------- RAW START ---------\n{json_text[:2500]}\n"
            f"--------- RAW END -----------"
        )


# ----------------------------------------------------
# MAIN FUNCTION (With Retry)
# ----------------------------------------------------
def generate_medication_plan_llm(
    age: int,
    gender: str,
    diagnosis: dict,
    timeline: dict,
    labs: dict,
    vitals: dict
) -> dict:

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # Controlled snippets
    def _j(x, limit=2000):
        try:
            return json.dumps(x, ensure_ascii=False)[:limit]
        except:
            return "{}"

    timeline_str = _j(timeline)
    labs_str = _j(labs)
    vitals_str = _j(vitals)

    today = datetime.now().strftime("%Y-%m-%d")

    # ----------------------------------------------------
    # PROMPT
    # ----------------------------------------------------
    prompt = f"""
You are a clinical pharmacologist generating a **synthetic but realistic medication plan**.

STRICT RULES:
- Output ONLY valid JSON.
- No markdown.
- No explanation.
- No commentary outside JSON.
- No trailing commas.
- All interaction references MUST match meds in current_medications.

PATIENT:
Age: {age}
Gender: {gender}
Diagnosis: {dx}
ICD-10: {icd}
SNOMED: {snomed}

RELEVANT SNIPPETS:
Timeline: {timeline_str}
Labs: {labs_str}
Vitals: {vitals_str}

OUTPUT FORMAT (MUST MATCH EXACTLY):

{{
  "medication_summary": {{
    "polypharmacy_level": "low | moderate | high",
    "overall_risk_commentary": "string"
  }},
  "current_medications": [
    {{
      "name": "string",
      "generic_name": "string",
      "drug_class": "string",
      "route": "PO | IV | SQ | IM | transdermal | inhaled",
      "dose": "string",
      "frequency": "string",
      "indication": "string",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD or null",
      "is_prn": true,
      "common_side_effects": "string",
      "serious_risks": "string",
      "monitoring_requirements": "string",
      "high_risk_for_elderly": false,
      "black_box_warning": "string or null",
      "interaction_flags": [
        {{
          "other_med_name": "string",
          "interaction_type": "string",
          "interaction_severity": "mild | moderate | major",
          "interaction_rationale": "string"
        }}
      ]
    }}
  ],
  "historical_medications": [
    {{
      "name": "string",
      "generic_name": "string",
      "drug_class": "string",
      "route": "string",
      "dose": "string",
      "frequency": "string",
      "indication": "string",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "reason_stopped": "string",
      "notable_side_effects_observed": "string",
      "interaction_related_stop": false
    }}
  ]
}}
"""

    # ----------------------------------------------------
    # Retry logic (3 attempts)
    # ----------------------------------------------------
    last_error = None

    for attempt in range(3):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=3500,
            )
            raw = (response.output_text or "").strip()
            return _safe_extract_json(raw)

        except Exception as e:
            print(f"[Medication Bot] Attempt {attempt+1} failed:", e)
            last_error = e

    raise ValueError(f"Medication Bot failed after 3 attempts: {last_error}")
