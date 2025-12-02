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
    """Extract and sanitize JSON from LLM output for procedures."""
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Procedure Bot: No JSON object found in output.")
    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Procedure Bot JSON parse error: {e}\nRaw: {json_text[:400]}...")


def generate_procedures_llm(
    age: int,
    gender: str,
    diagnosis: dict,
    timeline: dict,
    labs: dict,
    radiology: dict
) -> dict:
    """
    Generate a detailed list of procedures performed on this synthetic patient
    across their clinical course, with CPT/HCPCS codes, indications, findings,
    complications, and follow-up.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    def _j(x, limit=2500):
        try:
            s = json.dumps(x, ensure_ascii=False)
            return s[:limit]
        except Exception:
            return "{}"

    timeline_str = _j(timeline)
    labs_str = _j(labs)
    rads_str = _j(radiology)

    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are an attending physician documenting all **invasive and semi-invasive procedures**
performed on a synthetic patient during their disease course.

PATIENT CONTEXT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}
- ICD-10: {icd}
- SNOMED: {snomed}
- Do NOT invent a completely unrelated condition – procedures must match this context.

TIMELINE (snippet, for sequence and dates):
{timeline_str}

LABS (snippet, for coagulopathy/renal risk etc.):
{labs_str}

RADIOLOGY (snippet, for indications and guidance):
{rads_str}

GOAL:
Create a realistic **procedure history** including:
- Diagnostic procedures (e.g., colonoscopy, bronchoscopy, cardiac cath)
- Therapeutic procedures (e.g., thoracentesis, central line, chest tube, PCI)
- Minor bedside procedures (e.g., paracentesis, wound debridement)
- At least one procedure that had **minor or moderate complications**
  (e.g., hypotension, small bleed, transient arrhythmia) – not catastrophic.

OUTPUT FORMAT:
Return ONLY valid JSON, exactly in this structure:

{{
  "procedure_summary": {{
    "total_procedures": 0,
    "overall_risk_profile": "short technical summary of procedural risk across course",
    "complications_overview": "narrative overview of any complications and how they were managed"
  }},
  "procedures": [
    {{
      "procedure_date": "YYYY-MM-DD",
      "procedure_name": "e.g., Ultrasound-guided thoracentesis",
      "cpt_code": "e.g., 32555",
      "hcpcs_codes": ["e.g., J3010", "E1390"],
      "location": "ICU | Med-Surg | ED | Cath lab | OR | Interventional Radiology",
      "performing_service": "Hospitalist | Interventional Radiology | Cardiology | Pulmonology | Surgery | etc.",
      "operator_name": "fake physician name",
      "operator_role": "Attending / Fellow / Resident",
      "indication": "why the procedure was done, in clinical language",
      "pre_procedure_status": "brief status including vitals/coags relevant to risk",
      "anesthesia_sedation": "local only | moderate sedation | general anesthesia, with agents if appropriate",
      "technique": "short description of technique, landmarks/imaging guidance, and key steps",
      "devices_implanted": "stents, drains, catheters, ports, hardware, etc. or 'none'",
      "findings": "detailed intra-procedure findings (e.g., fluid characteristics, lesions, stenosis, etc.)",
      "specimens_sent": "if biopsies/fluids sent to lab, describe",
      "immediate_complications": "none | description of hypotension, bleed, arrhythmia, etc.",
      "complication_severity": "none | minor | moderate | severe",
      "post_procedure_care": "monitoring, orders, imaging confirmation, activity restrictions",
      "follow_up_plan": "next steps related to this procedure (e.g., repeat imaging, clinic follow-up, device removal)",
      "billing_comments": "short note referencing appropriateness of CPT/HCPCS coding in technical language"
    }}
  ]
}}

RULES:
- Procedures MUST be consistent with the diagnosis and the timeline of disease progression.
- Use realistic CPT and HCPCS style codes (they do not need to be perfectly accurate,
  but they must look like real code formats).
- At least 3–6 procedures total; more if clinically reasonable.
- At least 1–2 procedures should have minor or moderate complications
  (no catastrophic or fatal events).
- Use dense, technical language appropriate for clinician-to-clinician documentation.
- Output ONLY the JSON object, no commentary.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=3000,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
