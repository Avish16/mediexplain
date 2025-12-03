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
# ROBUST JSON EXTRACTOR (Pathology Bot)
# ----------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """
    Ultra-safe JSON extraction for Pathology Bot.
    Handles:
    - code fences
    - control characters
    - illegal escape sequences
    - trailing commas
    - excessive whitespace
    """

    if not text:
        raise ValueError("Pathology Bot: Empty model output.")

    # Remove markdown fencing
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Remove illegal escapes like \q, \%
    text = re.sub(r'\\(?!["\\/bfnrtu])', "", text)

    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(
            "Pathology Bot: Did not find a JSON object.\n"
            f"RAW OUTPUT START:\n{text[:1200]}\n..."
        )

    json_text = match.group(0)

    # Remove trailing commas
    json_text = re.sub(r",\s*(\})", r"\1", json_text)
    json_text = re.sub(r",\s*(\])", r"\1", json_text)

    # Compact whitespace
    json_text = re.sub(r"\s+", " ", json_text)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"\n❌ Pathology Bot JSON parse failed: {e}\n"
            f"------- RAW START -------\n{json_text[:2500]}\n"
            f"------- RAW END -------"
        )


# ----------------------------------------------------
# MAIN PATHOLOGY BOT (with retry safety net)
# ----------------------------------------------------
def generate_pathology_report_llm(
    age: int,
    gender: str,
    diagnosis: dict,
    procedures: dict,
    radiology: dict,
    labs: dict
) -> dict:

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # Safe snippet for context
    def _j(x, limit=2500):
        try:
            return json.dumps(x, ensure_ascii=False)[:limit]
        except:
            return "{}"

    proc_snip = _j(procedures)
    rads_snip = _j(radiology)
    labs_snip = _j(labs)

    today = datetime.now().strftime("%Y-%m-%d")

    # ----------------------------------------------------
    # PROMPT
    # ----------------------------------------------------
    prompt = f"""
You are a board-certified surgical pathologist generating a FULL pathology report.

STRICT RULES:
- OUTPUT ONLY VALID JSON.
- NO markdown.
- NO code fences.
- NO commentary outside the JSON.
- NO extra quotes, no trailing commas.
- MUST align with diagnosis and procedures.

PATIENT:
- Age: {age}
- Gender: {gender}
- Diagnosis: {dx}
- ICD: {icd}
- SNOMED: {snomed}

PROCEDURES (snippet):
{proc_snip}

RADIOLOGY (snippet):
{rads_snip}

LABS (snippet):
{labs_snip}

OUTPUT FORMAT:

{{
  "pathology_metadata": {{
    "accession_number": "string",
    "report_date": "{today}",
    "specimen_source": "string",
    "specimen_type": "string",
    "number_of_containers": 1,
    "ordering_provider": "string",
    "clinical_history": "string"
  }},
  "gross_description": "string",
  "microscopic_description": "string",
  "special_stains": [
    {{
      "stain": "string",
      "findings": "string"
    }}
  ],
  "immunohistochemistry": [
    {{
      "marker": "string",
      "result": "positive | negative | equivocal",
      "intensity": "weak | moderate | strong",
      "distribution": "focal | diffuse",
      "interpretation": "string"
    }}
  ],
  "molecular_studies": [
    {{
      "test": "string",
      "result": "string",
      "interpretation": "string"
    }}
  ],
  "final_diagnosis": "string",
  "margin_status": {{
    "involved": false,
    "details": "string"
  }},
  "cpt_codes": ["88305", "88342", "88341"],
  "pathologist_signature": {{
    "name": "Dr. First Last",
    "credentials": "MD, FCAP",
    "signature_date": "{today}",
    "lab_location": "Synthetic Pathology Laboratory"
  }}
}}
"""

    # ----------------------------------------------------
    # 3-ATTEMPT SAFETY NET
    # ----------------------------------------------------
    last_error = None

    for attempt in range(3):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=4500,
            )
            raw = (response.output_text or "").strip()
            return _safe_extract_json(raw)

        except Exception as e:
            print(f"[Pathology Bot] Attempt {attempt + 1} failed:", e)
            last_error = e

    raise ValueError(f"Pathology Bot failed after 3 attempts: {last_error}")