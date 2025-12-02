import json
import os
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


def generate_diagnosis_llm(age: int, gender: str) -> dict:
    """
    Generate a fully detailed, medically realistic diagnosis influenced
    by age + gender, including ICD-10, SNOMED-CT, CPT, HCPCS,
    severity, provider shorthand, and CMS terminology.
    """

    prompt = f"""
    You are an experienced clinician documenting diagnoses in an EMR.
    Produce a realistic diagnosis section for a fictional patient.

    CRITICAL RULES:
    - Output ONLY valid JSON.
    - No code fences, no commentary.
    - JSON must start with '{{' and end with '}}'.
    - Use professional medical terminology.
    - Include acronyms and provider shorthand used in real clinical documentation.

    Patient:
    - Age: {age}
    - Gender: {gender}

    YOU MUST INCLUDE ALL OF THE FOLLOWING FIELDS:

    {{
      "primary_diagnosis": "string (clinician style with acronyms)",
      "icd10_code": "ICD-10 code",
      "snomed_code": "SNOMED-CT code",
      "severity": "mild | moderate | severe",
      "clinical_status": "acute | chronic | acute-on-chronic",
      "clinical_description": "2–3 paragraphs using medical terminology, abbreviations, CMS phrasing, and pathophysiology based on age + gender.",
      "symptoms": ["list", "of", "key symptoms"],
      "risk_factors": ["list of risk factors, including age/gender-specific"],
      "differential_diagnosis": [
        {{"condition": "string", "icd10": "string"}},
        {{"condition": "string", "icd10": "string"}}
      ],
      "relevant_cpt_codes": [
        {{
          "cpt": "5-digit CPT code",
          "description": "procedure description (e.g., E/M, labs, imaging, pulmonary function test, etc.)"
        }}
      ],
      "relevant_hcpcs_codes": [
        {{
          "hcpcs": "HCPCS Level II code",
          "description": "supply/service description"
        }}
      ],
      "provider_abbreviations_used": [
        "list common provider acronyms used in this diagnosis (e.g., HPI, ROS, NAD, DOE, SOB, PERRLA, WNL, RLL, CXR, CT w/ contrast)"
      ],
      "cms_hcc_category": "HCC category name if applicable (e.g., HCC 18)",
      "cms_justification": "provider-style justification referencing MEAT documentation standards",
      "mdm_complexity": "low | moderate | high"
    }}

    DOCUMENTATION RULES:
    - Choose a diagnosis influenced by age + gender.
    - Diagnosis may be common or rare.
    - Use provider shorthand: SOB, DOE, NAD, A&Ox3, R/O, DDx, F/U.
    - Use CMS/HCC wording: MEAT, complexity of MDM, medical necessity.
    - Include CPT codes (99213, 93000, 71046, etc.).
    - Include HCPCS codes (J-codes for medications, E-codes for DME).
    - Include SNOMED-CT code for accuracy.
    - Write the clinical description as if entering into EPIC/Cerner.

    Make everything medically realistic but fully synthetic.
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=1200
    )

    raw = response.output_text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    return json.loads(raw)