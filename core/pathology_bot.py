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
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Pathology Bot: No JSON in output.")
    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Pathology Bot JSON parse error: {e}\nRaw: {json_text[:300]}...")


def generate_pathology_report_llm(
    age: int,
    gender: str,
    diagnosis: dict,
    procedures: dict,
    radiology: dict,
    labs: dict
) -> dict:
    """
    Generate a realistic pathology report based on biopsy/surgical specimens.
    Includes gross, microscopic, IHC stains, grading/staging, molecular markers,
    accession numbers, and pathologist interpretation.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    def _j(x, limit=2000):
        try:
            return json.dumps(x, ensure_ascii=False)[:limit]
        except:
            return "{}"

    proc_snip = _j(procedures)
    rads_snip = _j(radiology)
    labs_snip = _j(labs)

    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are a senior board-certified surgical pathologist generating a 
FULL pathology report for a fictional patient.

PATIENT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}
- ICD-10: {icd}
- SNOMED: {snomed}

RELEVANT PROCEDURES (snippet):
{proc_snip}

RELEVANT RADIOLOGY (snippet):
{rads_snip}

RELEVANT LABS (snippet):
{labs_snip}

GOAL:
Generate a **synthetic but highly realistic** pathology report that could come from
a hospital pathology department after biopsy or surgical resection.

MUST INCLUDE:
- Specimen metadata (source, type, container count)
- Accession number (fake but realistic)
- Gross description
- Microscopic description
- Histologic type
- Grade (if neoplasm)
- Depth, margin, invasion details (if cancer)
- IHC panel (immunohistochemistry markers)
- Molecular markers (EGFR, KRAS, HER2, ALK, PD-L1 etc. ONLY if relevant)
- Special stains (H&E mandatory, PAS, trichrome, GMS, etc. as needed)
- Pathologist interpretation (clinical correlation)
- Final diagnosis (dense technical language)
- Signature block (fake pathologist name, credentials, date/time)
- CPT codes for pathology services

FORMAT:
Return ONLY valid JSON with this structure:

{{
  "pathology_metadata": {{
    "accession_number": "string",
    "report_date": "{today}",
    "specimen_source": "e.g., right lower lobe lung biopsy",
    "specimen_type": "core biopsy | excisional biopsy | FNA | surgical specimen",
    "number_of_containers": 1,
    "ordering_provider": "fake physician name",
    "clinical_history": "brief clinical history from diagnosis/timeline"
  }},
  "gross_description": "very detailed gross exam description",
  "microscopic_description": "long multi-paragraph microscopic findings using pathology terminology",
  "special_stains": [
    {{
      "stain": "H&E | PAS | GMS | trichrome | etc.",
      "findings": "what the stain revealed"
    }}
  ],
  "immunohistochemistry": [
    {{
      "marker": "e.g., TTF-1, CK7, p40, ER, PR, HER2",
      "result": "positive | negative | equivocal",
      "intensity": "weak | moderate | strong",
      "distribution": "focal | diffuse",
      "interpretation": "pathologist interpretation"
    }}
  ],
  "molecular_studies": [
    {{
      "test": "EGFR | KRAS | ALK | BRAF | PD-L1 | MSI",
      "result": "positive | negative | mutation type",
      "interpretation": "clinical meaning"
    }}
  ],
  "final_diagnosis": "concise but technical pathology diagnosis with grading/staging if relevant",
  "margin_status": {{
    "involved": false,
    "details": "if margins involved, describe which and how"
  }},
  "cpt_codes": ["88305", "88342", "88341"],
  "pathologist_signature": {{
    "name": "Dr. First Last",
    "credentials": "MD, FCAP",
    "signature_date": "{today}",
    "lab_location": "Synthetic Pathology Laboratory"
  }}
}}

RULES:
- MUST align with diagnosis and procedures (e.g., lung cancer → lung biopsy pathology).
- MUST integrate hints from radiology and labs if relevant.
- MUST NOT produce cancer pathology for a non-cancer case.
- MUST produce complex microscopic and gross descriptions.
- MUST output ONLY the JSON object.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=4500,
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
