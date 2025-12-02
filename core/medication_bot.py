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
    """Extract and sanitize JSON from LLM output for medication bot."""
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Medication Bot: No JSON object found.")
    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Medication Bot JSON parse error: {e}\nRaw: {json_text[:400]}...")


def generate_medication_plan_llm(
    age: int,
    gender: str,
    diagnosis: dict,
    timeline: dict,
    labs: dict,
    vitals: dict
) -> dict:
    """
    Generate a full synthetic medication profile:
    - chronic + acute meds
    - doses, routes, frequencies
    - start/stop dates
    - indication
    - common side effects
    - serious risks
    - drug–drug interaction flags (pairs that are problematic together)
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # Helper to keep prompt size under control
    def _j(x, limit=2500):
        try:
            s = json.dumps(x, ensure_ascii=False)
            return s[:limit]
        except Exception:
            return "{}"

    timeline_str = _j(timeline)
    labs_str = _j(labs)
    vitals_str = _j(vitals)

    today_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are a clinical pharmacologist generating a **synthetic but realistic**
medication profile for a fictional patient.

PATIENT CONTEXT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}
- ICD-10: {icd}
- SNOMED: {snomed}

TIMELINE (snippet):
{timeline_str}

LABS (snippet):
{labs_str}

VITALS (snippet):
{vitals_str}

GOAL:
Create a rich, diagnosis-aware list of medications that this patient has been
on over the course of their illness, including:
- Chronic meds
- Acute/short-course meds
- PRN meds
- At least a few combinations with **non-trivial interactions** and **meaningful side effects**.

IMPORTANT:
This synthetic list is for testing a RAG-based medication safety checker.
So you MUST:

1. Include **several medications with well-known side effects**.
2. Include **at least 2–3 drug pairs** that are known to interact or increase risk
   when used together (e.g., bleeding risk, hyperkalemia, QT prolongation,
   serotonin syndrome, renal toxicity, etc.).
3. Clearly annotate:
   - side effects
   - serious risks
   - interaction flags (linking to other meds in the list by name).

OUTPUT FORMAT:
Return ONLY valid JSON with this exact structure:

{{
  "medication_summary": {{
    "polypharmacy_level": "low | moderate | high",
    "overall_risk_commentary": "technical narrative on med burden, risks, and monitoring needs."
  }},
  "current_medications": [
    {{
      "name": "brand or generic string",
      "generic_name": "string",
      "drug_class": "e.g., ACE inhibitor, SSRI, NSAID",
      "route": "PO | IV | SQ | IM | transdermal | inhaled | etc.",
      "dose": "e.g., 20 mg",
      "frequency": "e.g., once daily, BID, PRN q6h",
      "indication": "why the med is used for THIS patient (diagnosis-linked)",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD or null if ongoing",
      "is_prn": true,
      "common_side_effects": "comma-separated or narrative list of common side effects",
      "serious_risks": "narrative of serious adverse events (e.g., GI bleed, torsades, AKI)",
      "monitoring_requirements": "which labs/vitals need monitoring and why",
      "high_risk_for_elderly": true,
      "black_box_warning": "string or null",
      "interaction_flags": [
        {{
          "other_med_name": "name of interacting med FROM THIS LIST",
          "interaction_type": "e.g., increased bleeding risk, hyperkalemia, QT prolongation",
          "interaction_severity": "mild | moderate | major",
          "interaction_rationale": "short mechanistic explanation"
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
      "reason_stopped": "e.g., side effects, lack of efficacy, completed course, interaction concern",
      "notable_side_effects_observed": "what happened clinically (synthetic)",
      "interaction_related_stop": true
    }}
  ]
}}

RULES:
- All meds must be **consistent** with the patient's diagnosis, age, and overall picture.
- Always include **some** of the usual suspects for side effects / interactions, such as
  (only examples, choose what fits the case):
  - anticoagulants or antiplatelets
  - NSAIDs
  - ACE inhibitors / ARBs / spironolactone
  - SSRIs / SNRIs
  - opioids
  - QT-prolonging agents
- side_effects and serious_risks fields must be **non-empty** and clinically meaningful.
- interaction_flags MUST reference other meds actually present in current_medications.
- Dates should be realistic relative to today: no future dates beyond {today_str}.
- Output ONLY the JSON object, no explanations or extra text.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=3500,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
