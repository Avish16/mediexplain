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
# ROBUST JSON EXTRACTOR (Clinical Notes Bot)
# ----------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """
    Extremely defensive JSON extractor for Clinical Notes Bot.
    Handles:
    - markdown fences
    - control chars
    - illegal backslash escapes
    - trailing commas
    - extra whitespace / noise
    """

    if not text:
        raise ValueError("Clinical Notes Bot: Empty model output.")

    # Strip markdown fences if any
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove invisible control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Flatten newlines
    text = text.replace("\n", " ")

    # Remove illegal escapes like \q, \s, \3, etc.
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)

    # Find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(
            "Clinical Notes Bot: No JSON object found.\n"
            f"RAW START:\n{text[:1500]}\n..."
        )

    json_text = match.group(0)

    # Fix double braces
    json_text = json_text.replace("{{", "{").replace("}}", "}")

    # Remove trailing commas before } or ]
    json_text = re.sub(r",\s*(\})", r"\1", json_text)
    json_text = re.sub(r",\s*(\])", r"\1", json_text)

    # Collapse extra whitespace
    json_text = re.sub(r"\s+", " ", json_text)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"\n❌ Clinical Notes Bot: JSON parse failed: {e}\n"
            f"--------- RAW JSON START ---------\n{json_text[:4000]}\n"
            f"--------- RAW JSON END -----------"
        )


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
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=2000,
            )
            raw = (response.output_text or "").strip()
            return _safe_extract_json(raw)
        except Exception as e:
            print(f"[Clinical Notes Bot] Attempt {attempt+1} failed:", e)
            last_error = e
            continue

    # FINAL FALLBACK:
    print("[Clinical Bot WARNING] JSON parse failed, returning raw output:", last_error)
    return raw