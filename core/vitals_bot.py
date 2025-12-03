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
def generate_vitals_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> dict:
    """
    Generate a large, structured vitals report as JSON.
    Multiple timepoints + detailed measurements, but easy to render as key: value.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")

    prompt = f"""
You are generating a detailed VITALS REPORT for a hospitalized adult patient.

PATIENT:
- Age: {age}
- Gender: {gender}
- Primary Diagnosis: {dx}

HARD RULES:
- Output ONLY valid JSON.
- JSON must start with '{{' and end with '}}'.
- NO markdown, NO backticks, NO commentary outside JSON.
- NO raw newlines inside strings.

SCHEMA (DO NOT CHANGE KEYS):

{{
  "collection_metadata": {{
    "collection_date": "YYYY-MM-DD",
    "device": "Automated vitals monitor",
    "location": "Inpatient room",
    "encounter_id": "VIT-XXXXXX"
  }},
  "vital_series": [
    {{
      "time": "HH:MM",
      "context": "Resting | Post-ambulation | Post-medication | Nighttime | Early-morning",
      "measurements": [
        {{
          "name": "Heart Rate",
          "value": 0,
          "unit": "bpm",
          "reference_range": "60-100",
          "flag": "H|L|N",
          "interpretation": "short phrase"
        }},
        {{
          "name": "Blood Pressure",
          "value": "120/80",
          "unit": "mmHg",
          "reference_range": "90/60-130/85",
          "flag": "H|L|N",
          "interpretation": "short phrase"
        }}
      ]
    }}
  ],
  "overall_interpretation": "LONG paragraph summarizing hemodynamic stability, trends, and risk."
}}

POPULATION REQUIREMENTS:
- vital_series: at least 8 different time points spanning 24 hours.
- EACH timepoint's "measurements" must include at minimum:
  - Heart Rate
  - Blood Pressure
  - Respiratory Rate
  - Temperature
  - SpO2
  - Pain Score
- Additionally, scatter in:
  - Weight (once or twice),
  - BMI,
  - MEWS or NEWS-like composite score (as a 'name'),
  - any other relevant bedside scoring.
- For all vital values, keep them physiologically realistic for {age}-year-old {gender} with {dx}.
- Flags: "H" (high), "L" (low), "N" (normal).

RETURN ONLY THE JSON OBJECT. DO NOT wrap in ```json or add explanations.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2500,
    )

    raw = (response.output_text or "").replace("```json", "").replace("```", "").strip()
    return _safe_extract_json(raw)
