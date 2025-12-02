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
    """Extract and sanitize JSON from LLM output for billing bot."""
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Billing Bot: No JSON object found.")
    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Billing Bot JSON parse error: {e}\nRaw: {json_text[:400]}...")


def generate_billing_summary_llm(
    age: int,
    gender: str,
    demographics: dict,
    diagnosis: dict,
    procedures: dict,
    labs: dict,
    radiology: dict,
    medications: dict,
    length_of_stay_days: int = 5
) -> dict:
    """
    Generate a synthetic but realistic billing/coding summary:
    - ICD-10 / secondary diagnoses
    - CPT / HCPCS for procedures and diagnostics
    - DRG / HCC risk
    - Line-item charges with payer vs patient responsibility
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    primary_icd = diagnosis.get("icd10_code", "")

    def _j(x, limit=2500):
        try:
            return json.dumps(x, ensure_ascii=False)[:limit]
        except Exception:
            return "{}"

    demo_str = _j(demographics)
    proc_str = _j(procedures)
    lab_str = _j(labs)
    rad_str = _j(radiology)
    med_str = _j(medications)

    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are a hospital billing and coding specialist creating a synthetic but realistic
billing summary for a hospitalized patient.

PATIENT SNAPSHOT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}
- Primary ICD-10: {primary_icd}
- Approximate length of stay: {length_of_stay_days} days

DEMOGRAPHICS (snippet):
{demo_str}

PROCEDURES (snippet):
{proc_str}

LABS (snippet):
{lab_str}

RADIOLOGY (snippet):
{rad_str}

MEDICATIONS (snippet):
{med_str}

GOAL:
Create a coding + billing summary that would look genuine in a hospital record.
It should include:
- ICD-10 diagnosis codes (primary + a few secondary codes, consistent with the case)
- CPT codes for procedures and key diagnostic tests
- HCPCS codes for drugs/devices where appropriate
- DRG grouping (with description)
- HCC risk category or RAF-style comment (if appropriate)
- Line-item charges (professional + facility)
- Payer vs patient split (insurance vs copay/deductible)
- High-level narrative comments about coding/billing rationale.

Return ONLY valid JSON in this exact structure:

{{
  "billing_metadata": {{
    "statement_date": "{today}",
    "facility_name": "synthetic hospital name",
    "billing_account_number": "fake account number",
    "payer_name": "synthetic insurance plan",
    "coverage_type": "Commercial | Medicare | Medicaid | Self-pay | etc.",
    "length_of_stay_days": {length_of_stay_days}
  }},
  "diagnosis_codes": [
    {{
      "icd10_code": "string",
      "description": "string",
      "is_primary": true
    }}
  ],
  "procedure_codes": [
    {{
      "cpt_code": "string",
      "description": "string",
      "related_service": "short description (e.g., central line, CT chest, bronchoscopy)",
      "date_of_service": "YYYY-MM-DD",
      "place_of_service": "Inpatient hospital | ICU | ED | Outpatient",
      "hcpcs_codes": ["string HCPCS codes if applicable"]
    }}
  ],
  "drg_grouping": {{
    "drg_code": "e.g., 291",
    "drg_description": "string",
    "ms_drg_weight": 1.0,
    "notes": "technical note about why this DRG applies"
  }},
  "hcc_risk": {{
    "hcc_categories": [
      {{
        "hcc_code": "e.g., HCC85",
        "description": "string",
        "driving_diagnoses": ["ICD-10 codes feeding this HCC"]
      }}
    ],
    "risk_commentary": "narrative on overall risk capture and documentation quality"
  }},
  "line_items": [
    {{
      "service_date": "YYYY-MM-DD",
      "code_type": "ICD-10 | CPT | HCPCS | DRG",
      "code": "string",
      "description": "string",
      "quantity": 1,
      "charge_amount": 0.0,
      "allowed_amount": 0.0,
      "payer_paid": 0.0,
      "patient_responsibility": 0.0,
      "billing_notes": "short comment about this line item (e.g., bundled into DRG, separate professional fee, etc.)"
    }}
  ],
  "totals": {{
    "total_charges": 0.0,
    "total_allowed": 0.0,
    "total_payer_paid": 0.0,
    "total_patient_responsibility": 0.0
  }},
  "billing_commentary": "long technical narrative explaining coding decisions, DRG choice, use of secondary diagnoses, and any potential under-documentation or compliance issues."
}}

RULES:
- All codes must be **plausible** for the described case (they do not need to be perfect).
- ICD-10 and CPT/HCPCS should be consistent with diagnosis, procedures, labs, imaging, and meds.
- DRG and HCC discussion should sound like real coder/biller language.
- Monetary values should be realistic at US hospital scale (e.g., thousands to tens of thousands total).
- Output ONLY the JSON object, no extra commentary.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=3500,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
