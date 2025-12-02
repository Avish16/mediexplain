import json
import os
import re
from datetime import datetime
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
    """Extracts and sanitizes JSON from LLM output for radiology metadata."""

    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Radiology Bot: No JSON object found in LLM output.")
    json_text = match.group(0)

    # Minimal cleanup; radiology payload is smaller than timeline
    try:
        return json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Radiology Bot: JSON parse failed: {e}\nRaw: {json_text[:500]}...")


def generate_radiology_studies_llm(age: int, gender: str, diagnosis: dict, timeline: dict) -> dict:
    """
    Generate radiology study metadata + image prompts + dense findings/impression,
    then call the Image API to create grayscale radiology-like images.

    Returns a dict with:
      - studies: list of old + recent studies
      - each study has metadata, findings, impression, and generated image URL.
    """

    dx = diagnosis.get("primary_diagnosis", "Unknown Condition")
    icd = diagnosis.get("icd10_code", "")
    snomed = diagnosis.get("snomed_code", "")

    # Use timeline dates for old vs recent imaging if possible
    timeline_events = timeline.get("timeline_table", [])
    if timeline_events:
        try:
            first_date = timeline_events[0]["date"]
            last_date = timeline_events[-1]["date"]
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
          "role": "old",  // either "old" or "recent"
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

    # 2) For each study, generate an actual image using the Images API
    studies = meta.get("studies", [])
    for study in studies:
        prompt_text = study.get("image_prompt", "")
        if not prompt_text:
            continue

        # Enforce grayscale radiology style at image level too
        full_image_prompt = (
            prompt_text
            + " Radiology-style grayscale medical image, no color, no text, high contrast, clinical X-ray/CT/MRI aesthetic."
        )

        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=full_image_prompt,
            size="1024x1024",
            n=1
        )

        # You can choose url or base64; here we use URL for simplicity
        image_url = img_resp.data[0].url
        study["image_url"] = image_url

    return meta
