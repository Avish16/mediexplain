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


def _safe_extract_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Prescription Bot: No JSON object found in output.")
    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Prescription Bot JSON parse error: {e}\nRaw: {json_text[:400]}...")


def generate_prescriptions_llm(
    age: int,
    gender: str,
    diagnosis: dict,
    medication_plan: dict,
    vitals: dict,
    labs: dict
) -> dict:
    """
    Generates the doctor’s actual handwritten-style prescriptions
    (converted to structured JSON).
    These are the medications the patient is SENT HOME WITH.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    today = datetime.now().strftime("%Y-%m-%d")

    # Helper to keep snippets small but informative
    def _j(x, limit=2500):
        try:
            s = json.dumps(x)
            return s[:limit]
        except:
            return "{}"

    meds_snippet = _j(medication_plan)
    labs_str = _j(labs)
    vitals_str = _j(vitals)

    prompt = f"""
You are a prescribing physician writing **final discharge prescriptions**
for a synthetic patient.

The patient has the following:

Diagnosis:
{json.dumps(diagnosis)}

Medication History (snippet):
{meds_snippet}

Vitals:
{vitals_str}

Labs:
{labs_str}

GOAL:
Produce a complete **prescription order set**, including:
- Chronic meds the patient must continue
- New meds started this visit
- Short-course prescriptions (e.g., antibiotics, steroids)
- Pain meds (if safe)
- PRN meds
- Any meds discontinued with substitute prescribed
- Include common drug–drug interactions and explicit warnings

VERY IMPORTANT:
This prescription list must ALWAYS contain:
- Several meds with **known side effects**
- At least 2 drug pairs with **non-trivial interactions**, like:
    - SSRIs + triptans (serotonin syndrome)
    - Warfarin + NSAIDs (bleeding)
    - ACE inhibitor + spironolactone (hyperkalemia)
    - Macrolides + QT-prolonging agents
- Prescriptions MUST be safe *except for the interactions section*
  (because your project’s RAG bot needs to detect these).

OUTPUT JSON EXACTLY IN THIS STRUCTURE:

{{
  "prescription_metadata": {{
    "prescriber_name": "Dr. First Last",
    "prescriber_role": "Attending Physician",
    "prescriber_id": "fake DEA/NPI-style ID",
    "prescription_date": "{today}",
    "facility": "synthetic medical center name"
  }},
  "prescriptions": [
    {{
      "drug_name": "string",
      "generic_name": "string",
      "strength": "e.g., 20 mg",
      "route": "PO | IV | SQ | inhaled | topical | etc.",
      "form": "tablet | capsule | inhaler | vial | suspension",
      "sig_instructions": "full patient instructions (Sig)",
      "quantity": "e.g., #30",
      "refills": "0 | 1 | 2 | PRN",
      "indication": "why this med is prescribed",
      "common_side_effects": "string list",
      "serious_warnings": "string list",
      "interaction_warnings": [
        {{
          "with_drug": "another drug from this list",
          "interaction_type": "bleeding | hyperkalemia | serotonin syndrome | QT prolongation | etc.",
          "severity": "mild | moderate | major",
          "mechanism": "short rationale"
        }}
      ],
      "substitution_allowed": true
    }}
  ],
  "discontinued_medications": [
    {{
      "name": "string",
      "reason_stopped": "side effects | interaction | no longer indicated",
      "replacement_medication": "string or null"
    }}
  ],
  "patient_counseling": "long narrative of counseling given to patient, including side effects, warnings, follow-up."
}}

RULES:
- Use realistic prescribing language.
- Use drugs that MATCH the patient’s diagnosis.
- MUST include multiple meds with significant interactions.
- MUST be long, detailed, and medically technical.
- Write ONLY valid JSON.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=3500,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
