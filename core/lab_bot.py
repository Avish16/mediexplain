import json
import os
import re
from datetime import datetime
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


# ---------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and st is not None:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=api_key)


client = _get_openai_client()


# ---------------------------------------------------------
# SUPER-ROBUST JSON CLEANER FOR LAB BOT
# ---------------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """
    Extremely robust JSON extractor for LAB BOT.
    Handles:
    - double braces {{ }}
    - unterminated strings
    - stray newlines
    - stray commas
    - control characters
    - unescaped quotes inside values
    - GPT hallucinations in giant lab sections
    """

    # Remove markdown noise
    text = text.replace("```json", "").replace("```", "").strip()

    # Remove invisible ctrl chars
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Fix {{ }}
    text = text.replace("{{", "{").replace("}}", "}")

    # Remove line breaks inside JSON strings
    text = text.replace("\n", " ")

    # Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Lab Bot: No JSON found at all.")
    json_text = match.group(0)

    # Fix unescaped quotes inside values
    # (Safest broad rule: escape quotes that are not structural)
    def fix_quotes(s):
        out = ""
        in_string = False
        prev = ""
        for ch in s:
            if ch == '"' and prev != "\\":
                in_string = not in_string
                out += ch
            elif in_string and ch == '"' and prev != "\\":
                out += '\\"'
            else:
                out += ch
            prev = ch
        return out

    json_text = fix_quotes(json_text)

    # Remove trailing commas before } or ]
    json_text = re.sub(r",\s*(\})", r"\1", json_text)
    json_text = re.sub(r",\s*(\])", r"\1", json_text)

    # Remove multiple spaces
    json_text = re.sub(r"\s+", " ", json_text)

    # Attempt parsing
    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"\n❌ LAB BOT JSON FAILED: {e}\n"
            f"---- RAW CLEANED START ----\n"
            f"{json_text[:4000]}\n"
            f"---- RAW CLEANED END ----\n"
        )


# ---------------------------------------------------------
# MAIN LLM CALL
# ---------------------------------------------------------
def generate_lab_report_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> dict:
    """
    Generate extremely large 14-section lab report.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    timeline_events = timeline.get("timeline_table", [])
    if timeline_events:
        first_date = timeline_events[0]["date"]
    else:
        first_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
      You are a senior clinical pathologist generating a highly complex, synthetic laboratory report.
      STRICT RULES:
      - Output JSON ONLY (no comments, no text).
      - JSON must start with '{{' and end with '}}'.
      - No raw newlines inside strings.
      - Use medical terminology.
      - Date must be >= {first_date}.

      REQUIRED LAB SECTIONS:
      CBC, CMP, Lipid, Coagulation, Cardiac, Endocrine, Renal, Infection markers,
      Microbiology, Toxicology, Diagnosis-specific tests, CPT codes, reference ranges,
      abnormal flags, specimen metadata, long interpretation summary.

      FORMAT:
      {{
      "collection_metadata": {{
         "collection_date": "YYYY-MM-DD",
         "collection_time": "HH:MM",
         "specimen_type": "string",
         "order_id": "string",
         "performing_lab": "string"
      }},
      "cbc": {{
         "panel_cpt": "85025",
         "tests": [...]
      }},
      "cmp": {{
         "panel_cpt": "80053",
         "tests": [...]
      }},
      "lipid_panel": {{
         "panel_cpt": "80061",
         "tests": [...]
      }},
      "coagulation_panel": {{
         "panel_cpt": "85610/85730",
         "tests": [...]
      }},
      "cardiac_markers": {{
         "panel_cpt": "84484/83880",
         "tests": [...]
      }},
      "endocrine_labs": {{
         "panel_cpt": "83036/84443",
         "tests": [...]
      }},
      "renal_panel": {{
         "panel_cpt": "82565/82043",
         "tests": [...]
      }},
      "infection_markers": {{
         "panel_cpt": "86140/83605",
         "tests": [...]
      }},
      "microbiology": {{
         "panel_cpt": "87040/87070",
         "culture_results": "string",
         "organism_identified": "string or null",
         "sensitivity_pattern": "string",
         "gram_stain": "string"
      }},
      "toxicology": {{
         "panel_cpt": "80307",
         "tests": [...]
      }},
      "diagnosis_specific_labs": {{
         "panel_description": "string",
         "panel_cpt": "varies",
         "tests": [...]
      }},
      "interpretation_summary": "LONG technical interpretation text"
      }}
   """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=4500
    )

    raw = response.output_text or ""
    print("\n\n========== RAW LAB BOT OUTPUT ==========\n")
    print(raw)
    print("\n========== END RAW LAB BOT OUTPUT ==========\n")

    raw_cleaned = raw.replace("```json", "").replace("```", "").strip()

    # TEMP: show what we are sending to the parser
    print("\n\n========== RAW_CLEANED BEFORE PARSING ==========\n")
    print(raw_cleaned[:4000])  # first 4k chars
    print("\n========== END RAW_CLEANED ==========\n")

    return _safe_extract_json(raw_cleaned)

