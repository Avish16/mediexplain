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
#  MAIN LAB BOT (PLAIN TEXT)
# ============================================================
def generate_lab_report_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> str:
    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    timeline_events = timeline.get("timeline_table", [])
    if timeline_events:
        first_date = timeline_events[0].get("date", "")
    else:
        first_date = datetime.now().strftime("%Y-%m-%d")

    # ------------------------------------------------------------
    # SUPER CLEAN PLAIN-TEXT PROMPT
    # ------------------------------------------------------------
    prompt = f"""
You are generating a LARGE, REALISTIC, MULTI-PANEL LAB REPORT in plain text.

STRICT RULES:
- Output ONLY plain text.
- NO JSON.
- NO markdown.
- NO brackets of any kind (no [], {}, ()).
- NO bullet symbols (•, -, *, >).
- ONLY use:
   LAB SUMMARY:
   LAB TABLE:
   and numbered items 1., 2., 3., etc.
- This output will be directly inserted into a PDF.

OUTPUT FORMAT (MUST FOLLOW EXACTLY):

LAB SUMMARY:
<one long realistic clinical summary describing abnormalities, patterns, and diagnosis relevance>

LAB TABLE:
1. <Panel Name>
   CPT: <CPT code>
   Values:
      Test Name: Value Unit (Flag)
      Test Name: Value Unit (Flag)
   Interpretation: <1–3 sentences>

2. <Next Panel>
   CPT: <CPT>
   Values:
      ...
   Interpretation: ...

(Continue until 8–14 panels: CBC, CMP, Lipids, Coagulation, Cardiac markers, Endocrine, Renal, Infection markers, Microbiology, Toxicology, Diagnosis-related)

PANELS REQUIRED (ALL MUST APPEAR):
- CBC Panel
- CMP Panel
- Lipid Panel
- Coagulation Panel
- Cardiac Marker Panel
- Endocrine Panel
- Renal Panel
- Infection Marker Panel
- Microbiology Panel
- Toxicology Panel
- Diagnosis-Specific Panel

CONTENT REQUIREMENTS:
- 40–120 lab values across panels
- Use real clinical abbreviations as needed
- Include Flags (H, L, C)
- All dates must be >= {first_date}
- Make the interpretation medically deep + realistic

PATIENT:
Age {age}
Gender {gender}
Diagnosis: {dx}
ICD-10: {icd}
SNOMED: {snomed}

Return ONLY the LAB SUMMARY and LAB TABLE sections.
"""

    last_error = None

    # ------------------------------------------------------------
    # RETRY SAFETY (3 attempts)
    # ------------------------------------------------------------
    for attempt in range(3):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=4500
            )

            text = (response.output_text or "").strip()

            # IMPORTANT: Remove forbidden characters
            banned = ["[", "]", "{", "}", "<", ">", "*", "•"]
            for b in banned:
                text = text.replace(b, "")

            # Must contain headers
            if "LAB SUMMARY:" not in text:
                raise ValueError("Missing LAB SUMMARY section")

            if "LAB TABLE:" not in text:
                raise ValueError("Missing LAB TABLE section")

            # Must contain event numbering
            if "1." not in text:
                raise ValueError("No panel numbering found")

            return text

        except Exception as e:
            print(f"[Lab Bot] Attempt {attempt + 1} failed:", e)
            last_error = e

    raise ValueError(f"Lab Bot failed after 3 attempts: {last_error}")
