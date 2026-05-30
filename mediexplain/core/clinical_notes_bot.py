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
# ROBUST JSON EXTRACTOR (Clinical Notes Bot)
# ----------------------------------------------------
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
        return {"_parse_error": "Clinical Notes Bot: empty model output"}

    # Strip markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove invisible control characters (keep \n for now)
    text = re.sub(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Remove illegal JSON escape sequences like \q \s \3
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)

    # Find the first JSON object boundary
    start = text.find("{")
    if start == -1:
        return {"_parse_error": "Clinical Notes Bot: no JSON object found",
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
    print("[Clinical Notes Bot] [FAIL] All JSON recovery attempts failed. Returning empty shell.")
    return {
        "_truncated": True,
        "_parse_error": "Clinical Notes Bot: JSON unrecoverable",
        "_raw_preview": json_text[:500],
    }


# ----------------------------------------------------
# MAIN LLM CALL (with retry)
# ----------------------------------------------------
def generate_clinical_notes_llm(
    age: int,
    gender: str,
    demographics: dict,
    diagnosis: dict,
    timeline: dict,
    labs: dict,
    vitals: dict,
    radiology: dict
) -> dict:
    """
    Generate a comprehensive set of clinical notes (SOAP, H&P, ED note,
    progress notes, consults, procedure snippets, discharge summary),
    heavily using medical terminology and aligned with the synthetic patient.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # Serialize supporting data (truncated if extremely long)
    def _j(x, limit=4000):
        try:
            s = json.dumps(x, ensure_ascii=False)
            return s[:limit]
        except Exception:
            return "{}"

    demo_str = _j(demographics)
    timeline_str = _j(timeline)
    labs_str = _j(labs)
    vitals_str = _j(vitals)
    rads_str = _j(radiology)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt = f"""
You are an experienced attending physician documenting a full clinical record
for a single fictional patient in a hospital EMR.

The system has already generated structured data for this patient.

PATIENT DEMOGRAPHICS (JSON SNIPPET):
{demo_str}

PRIMARY DIAGNOSIS (JSON SNIPPET):
{json.dumps(diagnosis, ensure_ascii=False)[:4000]}

TIMELINE (JSON SNIPPET):
{timeline_str}

LABS (JSON SNIPPET):
{labs_str}

VITALS (JSON SNIPPET):
{vitals_str}

RADIOLOGY (JSON SNIPPET):
{rads_str}

PATIENT CONTEXT SUMMARY:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}
- ICD-10: {icd}
- SNOMED: {snomed}
- Current documentation datetime (for note headers): {now_str}

GOAL:
Generate a highly detailed set of clinical notes that together would occupy
AT LEAST 8+ pages when rendered in a typical PDF (single spacing,
normal margins, 11–12 pt font). The language should be dense, technical,
and difficult for laypersons to understand.

You must output ONLY JSON with the structure below. Use realistic provider
voice, CMS/HCC terminology, medical abbreviations (HPI, ROS, NAD, DOE, SOB,
MDM, etc.), and proper section headers inside the text (but keep them as plain text).

JSON OUTPUT FORMAT (EXACT KEYS):

{{
  "note_metadata": {{
    "facility_name": "string",
    "department": "string",
    "encounter_location": "string",
    "note_datetime": "YYYY-MM-DD HH:MM",
    "author_name": "string",
    "author_role": "string",
    "author_id": "string"
  }},
  "chief_complaint": "string",
  "soap_note": {{
    "subjective": {{
      "hpi": "long, multi-paragraph HPI",
      "ros": "multi-system review of systems",
      "pmh": "past medical history narrative",
      "psh": "past surgical history",
      "medications": "med list as narrative or bullet-like lines",
      "allergies": "allergy summary",
      "family_history": "family history",
      "social_history": "social history"
    }},
    "objective": {{
      "vitals_section": "summary of vitals with interpretation",
      "physical_exam": "very detailed multi-system physical exam",
      "labs_section": "summary of key labs and trends",
      "imaging_section": "summary of radiology findings and impressions",
      "other_data": "other relevant objective data"
    }},
    "assessment": "dense assessment with DDx, staging, ICD-10 references",
    "plan": "detailed plan: meds, labs, imaging, consults, procedures, follow-up"
  }},
  "hp_note": {{
    "chief_complaint": "string",
    "history_of_present_illness": "long narrative",
    "past_history_overview": "integrated PMH/PSH/FH/SH",
    "physical_exam": "H&P physical exam",
    "initial_ddx": "initial differential diagnosis",
    "admission_plan": "orders at admission",
    "risk_stratification": "discussion of risk scores / severity",
    "condition_severity": "summary line"
  }},
  "ed_note": {{
    "included": true,
    "triage_assessment": "ED triage description",
    "ed_hpi": "focused ED HPI",
    "ed_ros": "ED-focused ROS",
    "stabilization": "ABCs and emergent interventions",
    "ed_orders": "labs, imaging, meds ordered in ED",
    "disposition": "admit vs discharge vs transfer with rationale"
  }},
  "progress_notes": [
    {{
      "date": "YYYY-MM-DD",
      "interval_history": "what changed since prior day/visit",
      "events": "overnight events, new symptoms",
      "exam_changes": "changes in exam or vitals",
      "mdm_summary": "technical MDM summary",
      "plan_updates": "adjustments to plan"
    }}
  ],
  "consult_notes": [
    {{
      "service": "Cardiology | Pulmonology | Neurology | etc.",
      "reason_for_consult": "why the team was consulted",
      "consult_assessment": "specialty-specific assessment",
      "consult_recommendations": "detailed recommendations"
    }}
  ],
  "procedure_notes": [
    {{
      "procedure_name": "e.g., central line, thoracentesis",
      "indication": "why performed",
      "technique": "brief technique description",
      "findings": "key findings",
      "complications": "none or describe"
    }}
  ],
  "discharge_summary": {{
    "hospital_course": "long narrative of entire course",
    "key_diagnostics": "summary of key labs/imaging",
    "medications_at_discharge": "details of discharge meds with doses",
    "follow_up_recommendations": "PCP/specialist follow-up and timeframe",
    "pending_results": "any pending tests",
    "prognosis": "clinical prognosis statement",
    "pcp_instructions": "communication to PCP/outpatient team"
  }}
}}

RULES:
- Use dense, technical medical language and abbreviations.
- All content must be consistent with the demographics, diagnosis, labs, vitals, radiology, and timeline.
- Combined narratives should approximate 8+ pages of text.
- Output ONLY the JSON object. No markdown, no commentary.
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
            print(f"[Clinical Notes Bot] Attempt {attempt+1} [FAIL]:", e)
            last_error = e
            continue

    # FINAL FALLBACK: _safe_extract_json already returns a dict, never raises
    print("[Clinical Notes Bot] [WARN] All attempts exhausted:", last_error)
    return _safe_extract_json(raw if "raw" in dir() else "")