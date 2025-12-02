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
    """Extract and sanitize JSON from LLM output for clinical notes."""
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Clinical Notes Bot: No JSON object found in LLM output.")
    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Clinical Notes Bot: JSON parse failed: {e}\nRaw: {json_text[:500]}...")


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
    def _j(x):
        try:
            s = json.dumps(x, ensure_ascii=False)
            # hard cap to avoid giant prompts
            return s[:4000]
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
{json.dumps(diagnosis, ensure_ascii=False)}

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
AT LEAST 8+ pages when rendered in a typical PDF (assume single spacing,
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
    "author_name": "string (fake physician)",
    "author_role": "string (e.g., Attending Physician, Hospitalist)",
    "author_id": "string (fake NPI-style ID)"
  }},
  "chief_complaint": "short CC line",
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
      "labs_section": "summary of key labs and trends, referencing specific abnormalities",
      "imaging_section": "summary of radiology findings and impressions",
      "other_data": "any relevant nursing notes, telemetry, monitoring data, etc."
    }},
    "assessment": "dense assessment, including differential diagnosis (DDx), disease severity, staging, ICD-10 references, and comparison to prior visits/timeline.",
    "plan": "detailed plan covering meds, labs, imaging, consults, procedures, escalation/de-escalation of care, and follow-up."
  }},
  "hp_note": {{
    "chief_complaint": "string",
    "history_of_present_illness": "long H&P-style narrative (can reference HPI above but expand further)",
    "past_history_overview": "integrated PMH/PSH/FH/SH",
    "physical_exam": "H&P physical exam (can overlap with SOAP but more formal)",
    "initial_ddx": "initial differential diagnosis discussion",
    "admission_plan": "what was ordered at admission (labs, imaging, monitoring)",
    "risk_stratification": "discussion of risk scores / severity",
    "condition_severity": "summary line (e.g., acute-on-chronic, moderate-severe, etc.)"
  }},
  "ed_note": {{
    "included": true,
    "triage_assessment": "ED triage description with rapid vitals and chief concern",
    "ed_hpi": "focused ED HPI",
    "ed_ros": "ED-focused ROS",
    "stabilization": "airway/breathing/circulation, emergent interventions if any",
    "ed_orders": "labs, imaging, meds ordered in ED",
    "disposition": "admit vs discharge vs transfer rationale"
  }},
  "progress_notes": [
    {{
      "date": "YYYY-MM-DD",
      "interval_history": "what changed since prior day/visit",
      "events": "overnight events, new symptoms",
      "exam_changes": "changes in exam or vitals",
      "mdm_summary": "short but technical MDM summary",
      "plan_updates": "what was adjusted (meds, tests, consults)"
    }}
  ],
  "consult_notes": [
    {{
      "service": "Cardiology | Pulmonology | Neurology | etc.",
      "reason_for_consult": "why the team was consulted",
      "consult_assessment": "specialty-specific assessment",
      "consult_recommendations": "detailed recs"
    }}
  ],
  "procedure_notes": [
    {{
      "procedure_name": "e.g., central line, thoracentesis, etc.",
      "indication": "why performed",
      "technique": "short description of technique",
      "findings": "key findings, if any",
      "complications": "none or describe"
    }}
  ],
  "discharge_summary": {{
    "hospital_course": "long narrative of entire course, incorporating timeline, labs, imaging, and response to treatment.",
    "key_diagnostics": "summary of most important labs/imaging",
    "medications_at_discharge": "details of discharge meds with doses",
    "follow_up_recommendations": "PCP/specialist follow-up and timeframe",
    "pending_results": "any labs/imaging still pending",
    "prognosis": "clinical prognosis statement",
    "pcp_instructions": "communication to PCP or outpatient team"
  }}
}}

REQUIREMENTS:
- Use dense, technical medical language and abbreviations; make it hard for non-clinicians to understand fully.
- Ensure all notes are consistent with:
    - The given demographics
    - The primary diagnosis and its severity
    - The timeline of events
    - The labs, vitals, and radiology findings
- The combined length of all narrative text fields should be roughly equivalent to 8+ pages of typed content.
- DO NOT include any explanatory text outside the JSON; output only the JSON object.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=6000,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
