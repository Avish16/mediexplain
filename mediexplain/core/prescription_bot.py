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
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to .streamlit/secrets.toml or set the env var.")
    return OpenAI(api_key=api_key)


# Lazy singleton — resolved on first page load, not at import time
_client = None


def _client_instance():
    global _client
    if _client is None:
        _client = _get_openai_client()
    return _client


# ----------------------------------------------------
# ROBUST JSON EXTRACTOR (Prescription Bot)
# ----------------------------------------------------
def _close_truncated_json(text: str) -> str:
    """Close unclosed { [ brackets left by a truncated response."""
    open_stack = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                open_stack.append("}")
            elif c == "[":
                open_stack.append("]")
            elif c in "}]":
                if open_stack and open_stack[-1] == c:
                    open_stack.pop()
        i += 1
    return text.rstrip().rstrip(",").rstrip() + "".join(reversed(open_stack))


def _close_truncated_json(text: str) -> str:
    """
    Close any unclosed { [ brackets left by a truncated response.
    Walks the string tracking string/escape state so brackets inside
    string values are ignored.
    """
    open_stack = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2          # skip escaped character
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                open_stack.append("}")
            elif c == "[":
                open_stack.append("]")
            elif c in "}]":
                if open_stack and open_stack[-1] == c:
                    open_stack.pop()
        i += 1
    # Strip trailing comma/whitespace then close open structures
    repaired = text.rstrip().rstrip(",").rstrip()
    repaired += "".join(reversed(open_stack))
    return repaired


def _safe_extract_json(text: str) -> dict:
    """
    Defensive JSON extractor that handles truncated responses, markdown
    fences, control chars, illegal escapes, and trailing commas.
    On unrecoverable failure returns a partial-result dict instead of
    raising, so the pipeline can continue with degraded data.
    """
    if not text:
        return {"_parse_error": "Prescription Bot: empty model output"}

    # Strip markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove invisible control characters (keep \n for now)
    text = re.sub(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Remove illegal JSON escape sequences like \q \s \3
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)

    # Find the first JSON object boundary
    start = text.find("{")
    if start == -1:
        return {"_parse_error": "Prescription Bot: no JSON object found",
                "_raw_preview": text[:500]}

    json_text = text[start:]

    # Fix double braces and trailing commas
    json_text = json_text.replace("{{", "{").replace("}}", "}")
    json_text = re.sub(r",\s*(\})", r"\1", json_text)
    json_text = re.sub(r",\s*(\])", r"\1", json_text)

    # Attempt 1: parse as-is
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair truncation then parse
    try:
        repaired = _close_truncated_json(json_text)
        result = json.loads(repaired)
        result["_truncated"] = True     # flag so callers know data is partial
        return result
    except json.JSONDecodeError:
        pass

    # Attempt 3: extract whatever top-level keys parsed before the break
    partial: dict = {}
    # Pull out any "key": "string value" pairs we can find
    for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]{0,2000})"', json_text):
        partial[m.group(1)] = m.group(2)

    if partial:
        partial["_truncated"] = True
        partial["_parse_error"] = "JSON was truncated; only string fields recovered"
        return partial

    # Final fallback: return a safe shell so the pipeline does not crash
    print("[Prescription Bot] [FAIL] All JSON recovery attempts failed. Returning empty shell.")
    return {
        "_truncated": True,
        "_parse_error": "Prescription Bot: JSON unrecoverable",
        "_raw_preview": json_text[:500],
    }


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
            response = _client_instance().responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=8000,
            )
            raw = (response.output_text or "").strip()
            return _safe_extract_json(raw)
        except Exception as e:
            print(f"[Prescription Bot] Attempt {attempt + 1} failed:", e)
            last_error = e
            continue

    # FINAL FALLBACK:
    print("[Prescription Bot] [WARN] All attempts exhausted:", last_error)
    return _safe_extract_json(raw if "raw" in dir() else "")