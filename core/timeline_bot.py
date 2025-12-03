import json
import os
import re
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
    """
    Extract JSON from <JSON>...</JSON>. If missing, fallback to first {...} block.
    Cleans control characters and ensures valid JSON formatting.
    """

    # 1) Try <JSON>...</JSON>
    match = re.search(r"<JSON>(.*?)</JSON>", text, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
    else:
        # 2) Fallback: find first {...} block
        match2 = re.search(r"\{.*\}", text, re.DOTALL)
        if not match2:
            raise ValueError("Timeline Bot: <JSON> wrapper missing AND no {} JSON block found.")
        json_text = match2.group(0)

    # Cleanup
    json_text = re.sub(r"[\x00-\x1f\x7f]", " ", json_text)
    json_text = json_text.replace("\n", " ")
    json_text = re.sub(r",\s*(\}|\])", r"\1", json_text)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"Timeline Bot JSON parse failed: {e}\n\n--- RAW JSON START ---\n"
            f"{json_text[:3000]}\n--- RAW JSON END ---"
        )


def generate_timeline_llm(age: int, gender: str, diagnosis: dict) -> dict:
    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    prompt = f"""
You are a senior clinician generating a multi-year medical timeline.

### ABSOLUTE FORMAT RULES
- Your ENTIRE output MUST be inside these tags ONLY:

<JSON>
{{
  "timeline_summary": "...",
  "timeline_table": [...]
}}
</JSON>

- NOTHING is allowed before <JSON> or after </JSON>.
- Do NOT include explanation, markdown, or comments.
- All quotes inside strings must be escaped.
- Strings must NOT contain raw newlines. Use "\\n" for paragraph breaks.

### PATIENT:
Age: {age}
Gender: {gender}
Primary Diagnosis: {dx}
ICD-10: {icd}
SNOMED: {snomed}

### CONTENT REQUIREMENTS
- Timeline must span 1–5 years.
- Include 12–30 medically realistic events.
- Include ED visits, consults, therapies, complications, imaging, labs,
  procedures (with CPT codes), medication adjustments, readmissions.
- Summary must be long, technical, and professional.
- Use abbreviations (CTA, NAD, DOE, SOB, BNP, A1c, LE edema, HPI, etc).

### REQUIRED JSON FORMAT INSIDE <JSON> TAGS
{{
  "timeline_summary": "long narrative with \\n paragraphs",
  "timeline_table": [
    {{
      "date": "YYYY-MM-DD",
      "location": "string",
      "event_type": "string",
      "description": "3-5 sentence clinical summary",
      "actions_taken": "CPT/HCPCS, tests, meds",
      "outcome": "clinical outcome"
    }}
  ]
}}

Now produce ONLY this JSON, wrapped in <JSON> ... </JSON>.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=3000
    )

    raw = response.output_text or ""
    return _safe_extract_json(raw)
