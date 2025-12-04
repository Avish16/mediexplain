import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


_client = None



def _get_openai_client() -> OpenAI:
    """Return a singleton OpenAI client using env or Streamlit secrets."""
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
            "You are explaining this report to a medically experienced caregiver.\n"
            "- Use precise clinical language and documentation style.\n"
            "- Where possible, reference ICD-10 descriptions, typical clinical course,\n"
            "  risk factors, and red-flag symptoms.\n"
            "- You may use abbreviations like MI, CHF, COPD, etc., but define them once.\n"
            "- Structure your explanation into clear sections (Diagnosis, Findings,\n"
            "  Clinical reasoning, Monitoring, When to escalate care).\n"
        )
    # default: patient
    return (
        "You are explaining this report directly to a non-medical patient.\n"
        "- Use calm, reassuring, plain language (around 6th–8th grade level).\n"
        "- Avoid medical jargon. If you must use a term, explain it in simple words.\n"
        "- Focus on what is happening, why it matters, and what they might discuss\n"
        "  with their doctor.\n"
        "- Use short paragraphs and bullet points.\n"
    )


_DISCLAIMER = (
    "Important: This explanation is for education only and is **not** a diagnosis "
    "or medical advice. The patient must always confirm everything with their "
    "licensed healthcare team."
)


def generate_overall_explanation(
    mode: str,
    report_text: str,
    user_question: str | None = None,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1200,
) -> str:
    """
    High-level case explainer bot.

    Parameters
    ----------
    mode : 'patient' or 'caregiver'
    report_text : full clinical text / EMR / synthetic case
    user_question : optional specific question from the user
    model : OpenAI model name
    max_tokens : max output tokens

    Returns
    -------
    Markdown string explanation.
    """
    client = _get_openai_client()

    persona = _persona_block(mode)

    question_part = (
        f"\nThe person has this specific question:\n\"{user_question}\"\n"
        if user_question
        else "\nAddress the most important parts a person is likely worried about.\n"
    )

    system_prompt = f"""
You are MediExplain – an AI assistant that explains medical reports clearly and safely.

{persona}

Always include at the end a short section called
'Important Reminder' containing this (paraphrased in your own words):

{_DISCLAIMER}
"""

    user_content = (
        "Here is the medical report that needs to be explained:\n\n"
        "--------------------\n"
        f"{report_text}\n"
        "--------------------\n"
        f"{question_part}"
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


def run_explainer(mode: str, report_text: str, user_question: str | None = None):
    """
    Wrapper so other modules can call the explainer using a simple function name.
    Keeps backward compatibility.
    """
    return generate_overall_explanation(
        mode=mode,
        report_text=report_text,
        user_question=user_question
    )
