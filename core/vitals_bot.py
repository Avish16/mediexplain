import os
from datetime import datetime
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
    if not text:
        raise ValueError("Vitals Bot: empty model output.")

    # Strip accidental fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Vitals Bot: no JSON braces found.\nRAW: {text[:1000]}")

    json_text = match.group(0)

    # Fix common escape issue: \" appears literally
    json_text = json_text.replace('\\"', '"')

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"❌ Vitals Bot JSON Clean Failed: {e}\n"
            f"------- RAW START -------\n{json_text[:2500]}\n------- RAW END -------"
        )


def render_vitals_section(vitals: dict) -> str:
    lines = []
    meta = vitals.get("collection_metadata", {})
    lines.append("VITALS")
    lines.append(f"Date: {meta.get('collection_date', 'N/A')}  "
                 f"Location: {meta.get('location', 'N/A')}")
    lines.append("")

    for series in vitals.get("vital_series", []):
        time = series.get("time", "HH:MM")
        ctx = series.get("context", "")
        lines.append(f"Time: {time}  Context: {ctx}")
        for m in series.get("measurements", []):
            name = m.get("name", "Measurement")
            val = m.get("value", "")
            unit = m.get("unit", "")
            ref = m.get("reference_range", "")
            flag = m.get("flag", "N")
            interp = m.get("interpretation", "")
            lines.append(
                f"  {name}: {val} {unit} ({flag}) [ref: {ref}] – {interp}"
            )
        lines.append("")

    lines.append("VITALS SUMMARY:")
    lines.append(vitals.get("overall_interpretation", ""))
    lines.append("")
    return "\n".join(lines)

# ============================================================
#  PLAIN TEXT VITALS BOT (NO JSON ANYWHERE)
# ============================================================
def generate_vitals_llm(age: int, gender: str, diagnosis, timeline) -> str:
    """
    Generate a plain-text VITALS REPORT.
    No JSON parsing, no schema enforcement.
    Whatever the LLM outputs is returned as raw text.
    """

    # --- Normalize diagnosis (string OR dict) ---
    if isinstance(diagnosis, dict):
        dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    else:
        dx = str(diagnosis)[:200]  # take first sentence or so

    prompt = f"""
You are generating a detailed VITALS REPORT for a hospitalized adult patient.

PATIENT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}

Return a clear, human-readable vitals section with:
- At least 6–10 time points over 24 hours
- Heart rate, blood pressure, temperature, respiratory rate, SpO2, pain score
- Occasional weight, BMI, and composite score (NEWS, MEWS)
- A final short summary of overall interpretation

VERY IMPORTANT:
- Output MUST be plain text.
- DO NOT use JSON.
- DO NOT wrap anything in ``` or code fences.
Just write a realistic vitals narrative like a hospital chart.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2500,
    )

    # Return raw output exactly as provided
    return (response.output_text or "").strip()
