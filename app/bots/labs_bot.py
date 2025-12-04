import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


_client = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and st is not None:
            api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def _persona_block(mode: str) -> str:
    mode = (mode or "").lower()
    if mode == "caregiver":
        return (
            "Explain the lab results to a medically experienced caregiver.\n"
            "- Use standard lab terminology (CBC, CMP, troponin, BNP, etc.).\n"
            "- Mention typical reference ranges and what 'high' or 'low' implies\n"
            "  clinically (e.g., anemia, kidney dysfunction, myocardial injury).\n"
            "- Connect abnormal values to likely pathophysiology and ICD-10 style\n"
            "  diagnostic categories (e.g., I21.x for acute MI, N17.x for AKI).\n"
            "- Organize output by section: Hematology, Chemistry, Cardiac markers,\n"
            "  Inflammation, Other.\n"
        )
    return (
        "Explain these lab results to a non-medical patient.\n"
        "- Avoid numbers overload; focus on which values are normal vs. not.\n"
        "- For each important value, say:\n"
        "    * what the test measures\n"
        "    * whether it is in a safe range\n"
        "    * why the doctor might care about it\n"
        "- Use short, friendly sentences and avoid scary wording when possible.\n"
    )


_DISCLAIMER = (
    "This lab explanation is for understanding only. It does not replace real "
    "medical care, and treatment decisions must always be made by a licensed clinician."
)


def explain_labs(
    mode: str,
    labs_section_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1000,
) -> str:
    """
    Labs explainer bot.

    Parameters
    ----------
    mode : 'patient' or 'caregiver'
    labs_section_text : text or table describing the lab results

    Returns
    -------
    Markdown explanation of labs.
    """
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – an AI assistant that explains laboratory test results safely.

{persona}

Make clear which values look reassuring and which might be concerning,
and suggest example follow-up questions the person could ask their clinician.

Always end with a brief reminder similar to:

{_DISCLAIMER}
"""

    user_content = (
        "Here are the lab results to explain (may include units, ranges and flags):\n"
        "--------------------\n"
        f"{labs_section_text}\n"
        "--------------------\n"
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_tokens,
    )

    return (response.output_text or "").strip()


def run_labs(user_input: str, mode: str, pdf_text: str, memory_snippets):
    """
    Wrapper callable by the main MediExplain router.
    Extracts the 'labs section' from the PDF text and sends it
    to explain_labs().
    """
    labs_section = pdf_text  # SIMPLE: use whole text, or later refine extraction
    return explain_labs(mode, labs_section)
