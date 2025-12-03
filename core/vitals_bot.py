import os
from openai import OpenAI
import re

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


def _safe_extract_json(text: str) -> dict:
    """
    Safest extractor possible:
    - Detects escaped JSON
    - Unescapes it
    - Removes illegal characters
    - Removes trailing commas
    - Attempts 2-level JSON decode
    """

    # ---------- 1) Strip markdown ----------
    text = text.replace("```json", "").replace("```", "").strip()

    # ---------- 2) If output starts with \" instead of " it is ESCAPED JSON ----------
    if text.startswith("{\\") or text.startswith("\\{") or "\\\"" in text[:50]:
        # Unescape once
        text = text.encode("utf-8").decode("unicode_escape")

    # ---------- 3) Kill invisible chars ----------
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # ---------- 4) Remove newlines inside strings ----------
    text = text.replace("\n", " ")

    # ---------- 5) Extract JSON block ----------
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Vitals Bot: No JSON found at all.")

    json_text = match.group(0)

    # ---------- 6) Remove stray backslashes ----------
    json_text = re.sub(r"\\(?![\"\\/bfnrt])", "", json_text)

    # ---------- 7) Remove trailing commas ----------
    json_text = re.sub(r",\s*(\]|\})", r"\1", json_text)

    # ---------- 8) Now try parse ----------
    try:
        return json.loads(json_text)
    except:
        # Try second-pass decode (in case it's double-escaped)
        try:
            cleaned = json_text.encode("utf-8").decode("unicode_escape")
            return json.loads(cleaned)
        except Exception as e2:
            raise ValueError(
                f"\n❌ Vitals Bot JSON decode failed: {e2}\n"
                f"------- RAW JSON START -------\n{json_text[:2500]}\n"
                f"------- RAW JSON END ---------"
            )



# ============================================================
# SIMPLE TEXT CLEANER (NOT JSON)
# ============================================================
def _clean_text(text: str) -> str:
    """
    Cleans output but DOES NOT attempt JSON parsing.
    """
    if not text:
        return ""

    # Remove markdown fences
    text = text.replace("```", "").replace("```text", "").replace("```plaintext", "")

    # Remove invisible chars
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# MAIN VITALS BOT — PLAIN TEXT VERSION (UNBREAKABLE)
# ============================================================
def generate_vitals_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> str:
    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")

    prompt = f"""
You are generating a FULL hospital-grade VITALS REPORT in **plain text only**.

RULES:
- NO JSON
- NO braces
- NO code blocks
- NO markdown
- Write like a real EPIC/Cerner hospital vitals section
- Produce 2–3 pages of detailed vitals
- Include: HR, BP, MAP, RR, Temp, SpO2, ETCO2, Height, Weight, BMI, Pain Score, I/O Summary
- Add trend commentary, clinical interpretation, and risk context
- Use headers exactly like this:

VITALS REPORT — COLLECTION METADATA
VITALS – COMPLETE SET
ADDITIONAL OBSERVATIONS
CLINICAL INTERPRETATION SUMMARY

PATIENT:
Age: {age}
Gender: {gender}
Primary Diagnosis: {dx}

Return ONLY plain text.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=5000
    )

    return (response.output_text or "").strip()
