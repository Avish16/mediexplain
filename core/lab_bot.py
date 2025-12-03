import os
from datetime import datetime
from openai import OpenAI

try:
    import streamlit as st
except:
    st = None


# ============================================================
# OPENAI CLIENT
# ============================================================
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or (st.secrets["OPENAI_API_KEY"] if st else None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=api_key)


client = _get_openai_client()

# ============================================================
# JSON CLEANER
# ============================================================
def _safe_extract_json(text: str) -> dict:
    """
    Extracts the FIRST valid JSON object from any LLM output.
    Cleans:
    - Markdown fences
    - Control characters
    - Multiple spaces
    - Trailing commas
    - Braces mismatches
    """
    import json, re

    if not text:
        raise ValueError("Lab Bot: Empty LLM output")

    # Remove markdown noise
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove invisible ctrl chars
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Lab Bot: No JSON object found in output")

    json_text = match.group(0)

    # Remove trailing commas
    json_text = re.sub(r",\s*(\}|\])", r"\1", json_text)

    # Collapse multiple spaces
    json_text = re.sub(r"\s+", " ", json_text)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"Lab Bot JSON parse error: {e}\n"
            f"----- RAW JSON START -----\n"
            f"{json_text[:2500]}\n"
            f"----- RAW JSON END -----"
        )

# ============================================================
#  MAIN LAB BOT (PLAIN TEXT)
# ============================================================
def generate_lab_report_llm(age: int, gender: str, diagnosis, timeline: dict) -> dict:
    """
    Generate a LARGE, highly structured, multi-panel lab report.
    Handles diagnosis being either dict or string.
    """

    # ------------------------------------------------------------
    # FIX: Normalize diagnosis (HANDLE STR CASE)
    # ------------------------------------------------------------
    if isinstance(diagnosis, str):
        diagnosis = {
            "primary_diagnosis": diagnosis,
            "icd10_code": "",
            "snomed_code": ""
        }

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # ------------------------------------------------------------
    # Timeline date handling
    # ------------------------------------------------------------
    timeline_events = timeline.get("timeline_table", [])
    if timeline_events:
        first_date = timeline_events[0].get("date", "")
    else:
        first_date = datetime.now().strftime("%Y-%m-%d")

    # ------------------------------------------------------------
    # PROMPT — unchanged from your version
    # ------------------------------------------------------------
    prompt = f"""
You are a senior clinical pathologist generating a detailed synthetic LABORATORY REPORT.

PATIENT:
- Age: {age}
- Gender: {gender}
- Diagnosis: {dx} (ICD {icd}, SNOMED {snomed})

HARD RULES:
- Output ONLY **valid JSON**.
- JSON must start with '{{' and end with '}}'.
- NO markdown, NO backticks, NO commentary outside the JSON.
- NO newlines inside strings; use spaces instead.
- All numeric values must be realistic for an adult {gender} with {dx}.
- collection_date must be >= {first_date}.

STRUCTURE (DO NOT CHANGE KEYS):

{{
  "collection_metadata": {{
    "collection_date": "YYYY-MM-DD",
    "collection_time": "HH:MM",
    "specimen_type": "Blood",
    "order_id": "LAB-XXXXXX",
    "performing_lab": "string"
  }},
  "cbc": {{
    "panel_cpt": "85025",
    "panel_name": "Complete Blood Count",
    "tests": []
  }},
  "cmp": {{
    "panel_cpt": "80053",
    "panel_name": "Comprehensive Metabolic Panel",
    "tests": []
  }},
  "lipid_panel": {{
    "panel_cpt": "80061",
    "panel_name": "Lipid Panel",
    "tests": []
  }},
  "coagulation_panel": {{
    "panel_cpt": "85610/85730",
    "panel_name": "Coagulation Panel",
    "tests": []
  }},
  "cardiac_markers": {{
    "panel_cpt": "84484/83880",
    "panel_name": "Cardiac Marker Panel",
    "tests": []
  }},
  "endocrine_labs": {{
    "panel_cpt": "83036/84443",
    "panel_name": "Endocrine Panel",
    "tests": []
  }},
  "renal_panel": {{
    "panel_cpt": "82565/82043",
    "panel_name": "Renal Panel",
    "tests": []
  }},
  "infection_markers": {{
    "panel_cpt": "86140/83605",
    "panel_name": "Infection / Inflammation Markers",
    "tests": []
  }},
  "microbiology": {{
    "panel_cpt": "87040/87070",
    "panel_name": "Microbiology",
    "culture_results": "string",
    "organism_identified": "string or null",
    "sensitivity_pattern": "string",
    "gram_stain": "string"
  }},
  "toxicology": {{
    "panel_cpt": "80307",
    "panel_name": "Toxicology Screen",
    "tests": []
  }},
  "diagnosis_specific_labs": {{
    "panel_description": "Diagnosis-specific tests",
    "panel_cpt": "varies",
    "tests": []
  }},
  "interpretation_summary": "LONG, technical paragraph connecting all abnormal labs back to the diagnosis."
}}

POPULATION REQUIREMENTS:
(CBC, CMP, Lipid panel, etc… all identical to your original.)

RETURN ONLY THE JSON OBJECT.
"""

    # ------------------------------------------------------------
    # LLM CALL
    # ------------------------------------------------------------
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=4500,
    )

    raw = (response.output_text or "").strip()
    raw = raw.replace("```json", "").replace("```", "")

    # ------------------------------------------------------------
    # JSON PARSE (uses your existing extractor)
    # ------------------------------------------------------------
    return _safe_extract_json(raw)


def render_lab_section(labs: dict) -> str:
    lines = []

    meta = labs.get("collection_metadata", {})
    lines.append("LAB REPORT")
    lines.append(f"Date: {meta.get('collection_date', 'N/A')}  "
                 f"Time: {meta.get('collection_time', 'N/A')}  "
                 f"Specimen: {meta.get('specimen_type', 'N/A')}")
    lines.append("")

    for panel_key in ["cbc", "cmp", "lipid_panel", "coagulation_panel",
                      "cardiac_markers", "endocrine_labs",
                      "renal_panel", "infection_markers",
                      "toxicology", "diagnosis_specific_labs"]:
        panel = labs.get(panel_key)
        if not panel:
            continue

        panel_name = panel.get("panel_name", panel_key.upper())
        cpt = panel.get("panel_cpt", "")
        lines.append(f"{panel_name} (CPT {cpt})")

        for t in panel.get("tests", []):
            name = t.get("name", "Test")
            value = t.get("value", "")
            unit = t.get("unit", "")
            ref = t.get("reference_range", "")
            flag = t.get("flag", "N")
            interp = t.get("interpretation", "")
            lines.append(
                f"  {name}: {value} {unit} ({flag}) [ref: {ref}] – {interp}"
            )

        lines.append("")

    lines.append("INTERPRETATION:")
    lines.append(labs.get("interpretation_summary", ""))
    lines.append("")
    return "\n".join(lines)

