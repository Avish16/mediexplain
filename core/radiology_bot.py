import json
import os
import re
import base64
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
# JSON EXTRACTOR
# ----------------------------------------------------
def _safe_extract_json(text: str) -> dict:
    """Extracts and sanitizes JSON from LLM output for radiology metadata."""
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Radiology Bot: No JSON object found in LLM output.")
    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Radiology Bot: JSON parse failed: {e}\nRaw: {json_text[:500]}...")


# ----------------------------------------------------
# MAIN RADIology BOT
# ----------------------------------------------------
def generate_radiology_studies_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> dict:
    """
    Generate radiology study metadata + image prompts + dense findings/impression,
    then call the Image API to create grayscale radiology-like images and
    save them as PNGs under core/assets/.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # Use timeline dates for old vs recent imaging if possible
    timeline_events = timeline.get("timeline_table", [])
    if timeline_events:
        try:
            first_date = timeline_events[0].get("date", "")
            last_date = timeline_events[-1].get("date", "")
        except Exception:
            today = datetime.now().strftime("%Y-%m-%d")
            first_date = today
            last_date = today
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        first_date = today
        last_date = today

    prompt = f"""
    You are a board-certified radiologist generating synthetic but realistic radiology reports
    and image prompts for a fictional patient.

    Patient:
    - Age: {age}
    - Gender: {gender}
    - Primary Diagnosis: {dx}
    - ICD-10: {icd}
    - SNOMED: {snomed}

    Timeline:
    - Earlier imaging date should be around: {first_date}
    - Most recent imaging date should be around: {last_date}

    GOAL:
    Create TWO radiology studies for this patient:
      1) An OLDER baseline study
      2) A MORE RECENT follow-up study

    Each study should:
      - Choose the most appropriate modality based on the diagnosis:
        (e.g., chest X-ray for COPD/HF, CT brain for stroke, MRI spine, CT abdomen, etc.).
      - Use body-region-specific anatomy and pathology.
      - Use realistic radiology phrasing (e.g., "no acute osseous abnormality",
        "consolidation in the right lower lobe", "diffuse interstitial infiltrates",
        "marrow edema", "subchondral sclerosis", etc.).
      - Include disease progression or improvement between old vs recent where appropriate.

    IMAGE STYLE REQUIREMENTS:
      - Black-and-white, grayscale radiology style.
      - No color.
      - No text overlays or labels.
      - High contrast, medical imaging aesthetic.
      - Should visually represent the described pathology and evolution.

    OUTPUT FORMAT (JSON):

    {{
      "studies": [
        {{
          "role": "old",
          "study_date": "YYYY-MM-DD",
          "modality": "X-ray | CT | MRI | Ultrasound | etc.",
          "body_region": "e.g., chest, abdomen, brain, spine, knee",
          "clinical_indication": "why the study was ordered, in clinical language",
          "findings": "dense radiology-style findings paragraph",
          "impression": "concise radiologist impression, technical",
          "image_prompt": "a detailed prompt describing exactly what the radiology image should show in grayscale"
        }},
        {{
          "role": "recent",
          "study_date": "YYYY-MM-DD",
          "modality": "same or different modality as appropriate",
          "body_region": "string",
          "clinical_indication": "string",
          "findings": "string",
          "impression": "string",
          "image_prompt": "detailed grayscale radiology image description reflecting evolution (worse, better, or stable)"
        }}
      ],
      "radiology_summary": "long narrative comparing baseline vs follow-up, using dense radiology + clinical terms."
    }}

    RULES:
      - Output ONLY valid JSON, nothing else.
      - Use complex radiology jargon that is hard for laypersons to understand.
      - Ensure the image_prompt for each study clearly specifies: modality, body region,
        grayscale/black-and-white, patient positioning, and visible pathology.
    """

    # 1) Use Responses API to get structured metadata + image prompts
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=1800,
    )

    raw = response.output_text or ""
    meta = _safe_extract_json(raw)

    # 2) Prepare assets directory: core/assets/
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 3) For each study, generate an actual image and SAVE as PNG
    studies = meta.get("studies", [])
    for idx, study in enumerate(studies):
        prompt_text = study.get("image_prompt", "")
        if not prompt_text:
            continue

        full_image_prompt = (
            prompt_text
            + " Radiology-style grayscale medical image, no color, no text, high contrast, clinical X-ray/CT/MRI aesthetic."
        )

        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=full_image_prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )

        b64_data = img_resp.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)

        # Build a safe filename like: radiology_old_chest_2025-12-03.png
        role = study.get("role", f"study{idx}")
        body_region = study.get("body_region", "region")
        study_date = study.get("study_date", "unknown-date")

        raw_name = f"radiology_{role}_{body_region}_{study_date}".lower()
        safe_name = re.sub(r"[^a-z0-9_-]+", "_", raw_name)
        filename = f"{safe_name}.png"

        file_path = os.path.join(assets_dir, filename)
        with open(file_path, "wb") as f:
            f.write(img_bytes)

        # Save both local path and a simple relative path for downstream use
        study["image_path"] = file_path
        # If you still want to keep a URL-like field for compatibility:
        study["image_url"] = file_path

    return meta
