import json
import os

try:
    import streamlit as st
except ImportError:
    st = None  # allows use outside Streamlit if needed

from openai import OpenAI

def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and st is not None:
        # Try loading from Streamlit secrets if env var not set
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or Streamlit secrets.")
    return OpenAI(api_key=api_key)

client = _get_openai_client()

def generate_demographics_llm(age: int, gender: str) -> dict:
    """
    Generate synthetic but realistic medical demographics using an LLM.
    Only age and gender come from the user; everything else is LLM-generated.
    """

    prompt = f"""
    You are generating synthetic but realistic medical demographics for a fictional patient.

    CRITICAL RULES:
    - Output must be ONLY valid JSON.
    - Do NOT wrap the JSON in code fences.
    - Do NOT include explanations, comments, or any text outside the JSON object.
    - The JSON must start with '{{' and end with '}}'.

    User-provided fields:
    - Age: {age}
    - Gender: {gender}

    You must choose plausible values for all other fields yourself, including:
    - Ethnicity
    - Full US-style address (fake house number + street, city, state, ZIP)
    - Phone number (fake but valid US format)
    - Email (realistic-looking but fake)
    - Insurance details
    - Emergency contact
    - Social background

    Required JSON structure:

    {{
      "name": "string",
      "age": integer,
      "gender": "string",
      "ethnicity": "string",
      "mrn": "string",
      "address": "string",
      "phone": "string",
      "email": "string",
      "insurance": {{
        "provider": "string",
        "insurance_id": "string"
      }},
      "emergency_contact": {{
        "name": "string",
        "relationship": "string",
        "phone": "string"
      }},
      "social_history": {{
        "occupation": "string",
        "living_situation": "string"
      }}
    }}

    Make all values realistic but completely synthetic.
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=600,
    )

    raw_output = response.output_text.strip()

    # Safety: remove accidental code fences if the model disobeys
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    demographics = json.loads(raw_output)
    return demographics
