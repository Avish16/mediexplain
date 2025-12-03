import os
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
#  MAIN TIMELINE BOT (PLAIN TEXT)
# ============================================================
def generate_timeline_llm(age: int, gender: str, diagnosis: dict) -> str:
    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # ------------------------------------------------------------
    # NOTE: NO INDENTATION inside prompt — avoids LLM formatting drift.
    # ------------------------------------------------------------
    prompt = f"""
You are generating a multi-year CLINICAL TIMELINE for a synthetic patient.

RULES (IMPORTANT):
- Output ONLY plain text.
- NO JSON.
- NO markdown.
- NO brackets of any kind: [], {{}}, ().
- NO bullet symbols like • or - or *.
- Only use numbered events: 1., 2., 3., etc.
- MUST include the EXACT TWO HEADERS:

TIMELINE SUMMARY:
<one dense paragraph summary>

TIMELINE TABLE:
1. <event date> – <location>
   Event Type: <type>
   Description: <3–5 sentences>
   Actions Taken: <text>
   Outcome: <text>

2. <next event> …
(Continue numbering up to 12–20 events)

CONTENTS REQUIRED:
- 12–20 chronological events spanning 1–5 years.
- Include ED visits, imaging, labs, procedures, follow-ups.
- Use clinician abbreviations: SOB, DOE, CTA, EF%, BNP, CRT, A&O×3.
- Description paragraphs MUST NOT contain line breaks in the middle.
- All content must be medically realistic.

PATIENT DATA:
Age: {age}
Gender: {gender}
Diagnosis: {dx}
ICD-10: {icd}
SNOMED-CT: {snomed}

Return ONLY the final plain-text timeline.
"""

    last_error = None

    # ------------------------------------------------------------
    # RETRY SAFETY NET — try 3 times
    # ------------------------------------------------------------
    for attempt in range(3):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=3500
            )

            text = (response.output_text or "").strip()

            # --------------------------------------------------
            # HARD CLEANUP — remove any forbidden characters
            # --------------------------------------------------
            banned = ["[", "]", "{", "}", "<", ">", "*", "•"]
            for b in banned:
                text = text.replace(b, "")

            # --------------------------------------------------
            # Ensure mandatory headers exist
            # --------------------------------------------------
            if "TIMELINE SUMMARY:" not in text:
                raise ValueError("Missing 'TIMELINE SUMMARY:'")

            if "TIMELINE TABLE:" not in text:
                raise ValueError("Missing 'TIMELINE TABLE:'")

            # --------------------------------------------------
            # Ensure numbering is present
            # --------------------------------------------------
            if "1." not in text:
                raise ValueError("Timeline does not contain numbered events.")

            return text

        except Exception as e:
            print(f"[Timeline Bot] Attempt {attempt + 1} failed:", e)
            last_error = e

    raise ValueError(f"Timeline Bot failed after 3 attempts: {last_error}")
