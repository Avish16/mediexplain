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
#  PLAIN TEXT VITALS BOT (NO JSON ANYWHERE)
# ============================================================
def generate_vitals_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> str:
    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # pick a plausible date
    timeline_events = timeline.get("timeline_table", [])
    if timeline_events and "date" in timeline_events[0]:
        day = timeline_events[0]["date"]
    else:
        day = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are generating a clean, realistic, synthetic VITAL SIGNS report in plain text.

STRICT OUTPUT RULES:
- Output ONLY plain text.
- NO JSON.
- NO brackets of any kind.
- NO markdown.
- NO bullet symbols like dashes or asterisks.
- ONLY use:
    VITALS SUMMARY:
    VITALS TABLE:
    and numbered items 1., 2., 3., etc.
- This output will go directly into a PDF generator.

FORMAT YOU MUST FOLLOW EXACTLY:

VITALS SUMMARY:
<one paragraph summarizing overall stability or abnormalities, referencing trends, risk factors, and relation to diagnosis>

VITALS TABLE:
1. Heart Rate
   Value: <number> bpm
   Reference Range: <range>
   Flag: <H, L, or blank>
   Interpretation: <1 to 2 sentences>

2. Blood Pressure
   Value: <number/number mmHg>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

3. Respiratory Rate
   Value: <number breaths per minute>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

4. Temperature
   Value: <number C>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

5. Oxygen Saturation
   Value: <number percent>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

6. Weight
   Value: <number kg>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

7. Height
   Value: <number cm>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

8. Body Mass Index
   Value: <number>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

9. Pain Score
   Value: <number out of ten>
   Reference Range: <range>
   Flag: <H or L or blank>
   Interpretation: <1 to 2 sentences>

CONTENT RULES:
- All values must be clinically realistic.
- Flags should match the interpretation.
- Pain Score must be 0 to 10.
- Oxygen Saturation must be 85 to 100.
- Temperature must be 35.5 to 40.0.
- Heart Rate must be 40 to 160.
- Blood Pressure must be realistic ranges such as 90 slash 60 to 180 slash 120.
- No bracket symbols allowed.
- No code formatting allowed.
- EVERYTHING must be plain text.

PATIENT:
Age: {age}
Gender: {gender}
Diagnosis: {dx}
ICD: {icd}
SNOMED: {snomed}
Date: {day}

Return ONLY the VITALS SUMMARY and VITALS TABLE sections.
"""

    last_error = None

    # ============================================================
    # RETRY LOOP
    # ============================================================
    for attempt in range(3):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=2000
            )

            text = (response.output_text or "").strip()

            # Remove any forbidden characters
            banned = ["[", "]", "{", "}", "<", ">", "*", "•"]
            for b in banned:
                text = text.replace(b, "")

            # Required headers
            if "VITALS SUMMARY:" not in text:
                raise ValueError("Vitals missing summary header")

            if "VITALS TABLE:" not in text:
                raise ValueError("Vitals missing table header")

            # Ensure numbering exists
            if "1." not in text:
                raise ValueError("Vitals missing numbered items")

            return text

        except Exception as e:
            print(f"[Vitals Bot] Attempt {attempt + 1} failed:", e)
            last_error = e

    raise ValueError(f"Vitals Bot failed after 3 attempts: {last_error}")
