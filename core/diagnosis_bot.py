import os
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
        api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=api_key)


client = _get_openai_client()


# ----------------------------------------------------
# PLAIN TEXT DIAGNOSIS GENERATOR (NO JSON REQUIRED)
# ----------------------------------------------------
def generate_diagnosis_llm(age: int, gender: str) -> str:
    """
    Generates a fully detailed clinical diagnosis section
    in plain text with clear headers.
    """

    prompt = f"""
You are a senior physician completing an EMR diagnostic assessment.

Return ONLY plain text. 
No JSON. No code fences. No lists. 
Write a highly structured diagnosis section with clear headers.

PATIENT:
- Age: {age}
- Gender: {gender}

FORMAT EXACTLY LIKE THIS (plain text):

==========================
PRIMARY DIAGNOSIS
==========================
<detailed diagnosis name with clinician shorthand>

ICD-10: <code>
SNOMED-CT: <code>
Severity: <mild/moderate/severe>
Clinical Status: <acute/chronic/acute-on-chronic>

==========================
CLINICAL DESCRIPTION
==========================
<2–3 dense paragraphs of medical reasoning, pathophysiology,
abbreviations (NAD, DOE, SOB, EF%, A&O×3), and clinician terminology>

==========================
SYMPTOMS
==========================
<comma-separated symptoms>

==========================
RISK FACTORS
==========================
<comma-separated risk factors based on age & gender>

==========================
DIFFERENTIAL DIAGNOSIS
==========================
<condition 1> (ICD: <code>)
<condition 2> (ICD: <code>)

==========================
CPT / HCPCS CODES
==========================
CPT: <code> – <description>
HCPCS: <code> – <description>

==========================
PROVIDER ABBREVIATIONS USED
==========================
HPI, ROS, NAD, DOE, SOB, RLL, CT w/ contrast, A&O×3

==========================
CMS HCC CATEGORY
==========================
<category name>

==========================
CMS JUSTIFICATION
==========================
<provider-style MEAT documentation>

==========================
MDM COMPLEXITY
==========================
<low / moderate / high>

Return only the final formatted text.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2000
    )

    return (response.output_text or "").strip()
