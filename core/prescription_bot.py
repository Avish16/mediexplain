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
# ROBUST JSON EXTRACTOR (Prescription Bot)
# ----------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """
    Defensive JSON extractor for Prescription Bot:
    - strips markdown fences
    - removes control chars
    - cleans illegal backslash escapes
    - trims trailing commas
    - returns parsed JSON dict
    """
    if not text:
        raise ValueError("Prescription Bot: Empty model output.")

    # Remove code fences if model ever adds them
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove invisible control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Flatten newlines
    text = text.replace("\n", " ")

    # Remove illegal escapes like \q, \Z, etc. (keep only legal JSON escapes)
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(
            "Prescription Bot: No JSON object found in output.\n"
            f"RAW START:\n{text[:1500]}\n..."
        )

    json_text = match.group(0)

    # Fix double braces if they slip in
    json_text = json_text.replace("{{", "{").replace("}}", "}")

    # Remove trailing commas before } or ]
    json_text = re.sub(r",\s*(\})", r"\1", json_text)
    json_text = re.sub(r",\s*(\])", r"\1", json_text)

    # Collapse whitespace
    json_text = re.sub(r"\s+", " ", json_text)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"\n❌ Prescription Bot JSON parse error: {e}\n"
            f"--------- RAW JSON START ---------\n{json_text[:4000]}\n"
            f"--------- RAW JSON END -----------"
        )


# ----------------------------------------------------
# MAIN LLM CALL (Prescription Bot)
# ----------------------------------------------------
def generate_prescriptions_llm(
    age: int,
    gender: str,
    diagnosis: dict,
    medication_plan: dict,
    vitals: dict,
    labs: dict
) -> dict:
    """
    Generates the doctor’s discharge prescriptions as structured JSON:
    - Per-drug fields for name, strength, route, frequency, duration, etc.
    - Approving provider + department + timestamp
    - Side effects, serious warnings, interaction warnings
    - Discontinued meds + reasons
    - Long patient counseling narrative
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    today = datetime.now().strftime("%Y-%m-%d")

    # Helper to keep snippets small but informative
    def _j(x, limit=2500):
        try:
            s = json.dumps(x, ensure_ascii=False)
            return s[:limit]
        except Exception:
            return "{}"

    meds_snippet = _j(medication_plan)
    labs_str = _j(labs)
    vitals_str = _j(vitals)

    prompt = f"""
You are a prescribing physician writing FINAL DISCHARGE PRESCRIPTIONS
for a synthetic hospital patient.

CLINICAL CONTEXT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}
- ICD-10: {icd}
- SNOMED: {snomed}

Diagnosis (JSON):
{json.dumps(diagnosis, ensure_ascii=False)[:2000]}

Medication History (snippet, JSON):
{meds_snippet}

Vitals (snippet, JSON):
{vitals_str}

Labs (snippet, JSON):
{labs_str}

GOAL:
Produce a complete DISCHARGE PRESCRIPTION ORDER SET, including:
- Chronic meds to continue
- New meds started this admission
- Short-course meds (e.g., antibiotics, steroids)
- Pain meds (if safe)
- PRN meds
- Meds discontinued and replaced (if appropriate)

IMPORTANT:
This list is used by a downstream RAG safety checker, so you MUST:
1. Include several meds with well-known side effects.
2. Include at least 2–3 drug pairs with non-trivial interactions:
   - e.g., SSRIs + triptans (serotonin syndrome)
   - e.g., warfarin + NSAIDs (bleeding risk)
   - e.g., ACE inhibitor + spironolactone (hyperkalemia)
   - e.g., macrolide + QT-prolonging agent (torsades risk)
3. Clearly annotate:
   - common side effects
   - serious warnings
   - interaction warnings (with other meds in THIS list)
4. Include structured data that makes it easy to render a PDF table
   with: drug name, strength, route, frequency, duration, dose schedule,
   prescribing doctor, and warnings.

OUTPUT FORMAT:
Return ONLY valid JSON with EXACTLY this structure (you may add fields, but DO NOT remove any):

{{
  "prescription_metadata": {{
    "prescriber_name": "Dr. First Last",
    "prescriber_role": "Attending Physician",
    "prescriber_id": "fake DEA/NPI-style ID",
    "prescribing_department": "e.g., Cardiology, Hospital Medicine",
    "prescription_date": "{today}",
    "facility": "Synthetic Medical Center name",
    "approved_by": "Dr. Approver Name (can match prescriber or supervising physician)",
    "approved_datetime": "{today} 14:32",
    "dispensing_pharmacy": "Synthetic Outpatient Pharmacy name"
  }},
  "prescriptions": [
    {{
      "drug_name": "string (brand or generic)",
      "generic_name": "string",
      "drug_class": "e.g., ACE inhibitor, SSRI, NSAID",
      "strength": "e.g., 20 mg",
      "route": "PO | IV | SQ | inhaled | topical | etc.",
      "form": "tablet | capsule | inhaler | vial | suspension | patch",
      "dose_per_administration": "e.g., 20 mg per dose",
      "frequency": "e.g., once daily, BID, q8h, PRN q6h",
      "administration_instructions": "patient-facing instructions, e.g., take with food, avoid lying down, etc.",
      "indication": "why this med is prescribed for THIS patient/diagnosis",
      "start_date": "YYYY-MM-DD",
      "stop_date": "YYYY-MM-DD or null if ongoing",
      "duration_days": 7,
      "quantity": "e.g., #30",
      "refills": "0 | 1 | 2 | PRN | none",
      "prescribing_provider": "Dr. Name (can match prescriber_name)",
      "prescribing_department": "e.g., Cardiology, Hospital Medicine",
      "monitoring_plan": "what labs/vitals/symptoms to monitor and how often",
      "high_risk_medication": true,
      "black_box_warning": "string or null",
      "common_side_effects": "comma-separated or narrative list of common side effects",
      "serious_warnings": "narrative description of serious adverse events to watch for",
      "interaction_warnings": [
        {{
          "with_drug": "another drug FROM THIS prescriptions list (by name)",
          "interaction_type": "bleeding | hyperkalemia | serotonin syndrome | QT prolongation | CNS depression | etc.",
          "severity": "mild | moderate | major",
          "mechanism": "short rationale (e.g., both increase serotonin, both prolong QT, both affect INR)",
          "clinical_management": "how to manage (e.g., avoid combo, close INR monitoring, ECG monitoring, dose reduction)"
        }}
      ],
      "documented_past_reactions": "any prior ADRs this patient had to this drug or class, or 'none known'",
      "substitution_allowed": true,
      "dispense_as_written": false
    }}
  ],
  "discontinued_medications": [
    {{
      "name": "string",
      "generic_name": "string",
      "reason_stopped": "side effects | interaction | no longer indicated | duplicate therapy | formulary change",
      "replacement_medication": "string or null",
      "discontinuation_date": "YYYY-MM-DD",
      "adverse_reactions_observed": "narrative of what happened clinically, or 'none documented'"
    }}
  ],
  "patient_counseling": "long narrative of counseling given to the patient and/or caregiver, including: how and when to take each med, key side effects to watch for, red-flag symptoms requiring urgent evaluation, lab/clinic follow-up, what to avoid (OTC meds, alcohol, certain foods), and adherence tips.",
  "caregiver_instructions": "if applicable, guidance directed specifically to caregiver about organizing meds, monitoring for side effects, and when to contact the care team."
}}

RULES:
- Use realistic prescribing language for a US-style EMR discharge prescription.
- All meds MUST be consistent with the diagnosis, age, labs, and vitals.
- Make the prescriptions list fairly long (polypharmacy is OK if clinically plausible).
- Ensure interaction_warnings reference actual drug_name entries from the prescriptions array.
- Ensure text fields (side effects, warnings, monitoring_plan, counseling) are NON-EMPTY and medically meaningful.
- Output ONLY the JSON object. NO markdown, NO commentary.
"""

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
            print(f"[Prescription Bot] Attempt {attempt + 1} failed:", e)
            last_error = e

    raise ValueError(f"Prescription Bot failed after 3 attempts: {last_error}")
