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
    """Extract and sanitize JSON from LLM output for nursing notes."""
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Nursing Notes Bot: No JSON object found.")
    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Nursing Notes Bot JSON parse error: {e}\nRaw: {json_text[:300]}...")


def generate_nursing_notes_llm(
    age: int,
    gender: str,
    demographics: dict,
    diagnosis: dict,
    vitals: dict,
    labs: dict,
    timeline: dict
) -> dict:
    """Generate an extremely detailed nursing shift note aligned with the patient case."""
    
    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # small helper to truncate huge JSON before sending to model
    def _j(x):
        try:
            s = json.dumps(x)
            return s[:3000]
        except:
            return "{}"

    demo_str = _j(demographics)
    vitals_str = _j(vitals)
    labs_str = _j(labs)
    timeline_str = _j(timeline)

    prompt = f"""
You are an experienced inpatient RN writing a full nursing shift note
for a fictional patient. The note must be long, clinically detailed,
and align with the patient's diagnosis and hospital course.

PATIENT SNAPSHOT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}
- ICD-10: {icd}

DEMOGRAPHICS (snippet):
{demo_str}

VITALS (snippet):
{vitals_str}

LABS (snippet):
{labs_str}

TIMELINE (snippet):
{timeline_str}

GOAL:
Produce a comprehensive nursing shift note that includes:
- Head-to-toe assessment
- Pain assessment
- Mobility assessment
- Neuro/respiratory/cardiac/GI/GU/skin exam
- Lines/drains/airways
- Nursing interventions performed
- Med administrations
- Patient response to interventions
- Safety precautions
- Fall risk assessment
- Intake/output summary
- Shift events
- RN communication with MD
- End-of-shift handoff narrative

FORMAT:
Output ONLY valid JSON with this exact structure:

{{
  "nursing_note_metadata": {{
    "note_datetime": "{now_str}",
    "author_name": "string (fake RN)",
    "author_role": "Registered Nurse",
    "unit": "e.g., Med-Surg, Telemetry, ICU",
    "room_number": "fake room number"
  }},
  "shift_assessment": {{
    "neuro": "long nursing neuro exam",
    "respiratory": "detailed resp exam + device settings if any",
    "cardiac": "cardiac assessment, edema, rhythms",
    "gastrointestinal": "GI assessment",
    "genitourinary": "GU assessment",
    "skin": "skin integrity, wounds, breakdown, dressings",
    "musculoskeletal": "mobility status, gait, assist level",
    "pain_assessment": "pain score + description + response to meds",
    "lines_tubes_drains": "IVs, Foley, wound vac, JP drains, etc.",
    "safety_measures": "fall risk, bed alarms, sitter, precautions"
  }},
  "interventions": {{
    "med_admin": "all meds given this shift with response",
    "wound_care": "wound care performed",
    "resp_therapy": "breathing treatments, IS usage, suctioning",
    "mobility_care": "turning, ambulation, PT/OT interactions",
    "patient_education": "education provided"
  }},
  "shift_events": {{
    "new_symptoms": "new issues during shift",
    "provider_notifications": "MD notified events",
    "diagnostics_performed": "labs, scans done this shift",
    "consults": "any consults seen",
    "transport_events": "patient left unit for imaging/procedures"
  }},
  "intake_output": {{
    "intake": "summary of PO, IV, enteral intake",
    "output": "urine, stool, drains, emesis"
  }},
  "end_of_shift_handoff": "long narrative nurse-to-nurse handoff summarizing stability, concerns, and plan for next shift."
}}

RULES:
- Use REAL nursing terminology and abbreviations (A&O x3, RR even/unlabored, SR/ST on monitor, fecal smear, stage II pressure injury, etc.).
- Must sound like REAL RN documentation.
- Must align with vitals, labs, radiology, and diagnosis context.
- No text outside the JSON.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=3500,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
