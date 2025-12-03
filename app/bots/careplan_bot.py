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
            "You are outlining a high-level care plan for a medically experienced caregiver.\n"
            "- Use problem-oriented structure (Problem, Goals, Monitoring, Contingency).\n"
            "- Reference common guideline concepts (e.g., GDMT, step-wise escalation)\n"
            "  without giving specific prescriptions.\n"
            "- Mention which specialties might be involved (cardiology, pulmonology, etc.).\n"
        )
    return (
        "You are outlining a simple, supportive care plan summary for a patient.\n"
        "- Focus on themes: medicines, appointments, lifestyle, warning signs.\n"
        "- Avoid telling them exactly what to do; instead suggest topics to confirm\n"
        "  with their doctor.\n"
        "- Use clear headings and bullet points.\n"
    )


_DISCLAIMER = (
    "This care-plan summary is only a discussion guide. It is not a treatment plan "
    "and does not replace the plan made by the patient's healthcare team."
)


def generate_care_plan(
    mode: str,
    clinical_summary_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1200,
) -> str:
    """
    Care-plan outline bot.

    Parameters
    ----------
    mode : 'patient' or 'caregiver'
    clinical_summary_text : short summary / key diagnoses, labs, meds, etc.

    Returns
    -------
    Structured markdown care-plan style explanation.
    """
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – you help people understand the *shape* of a care plan
based on a report, without giving direct medical orders.

{persona}

You MUST:
- Avoid creating new diagnoses.
- Avoid giving exact medication names, doses, or frequencies beyond what is
  already in the report.
- Focus on 'what to monitor', 'who might be involved', and 'which questions to ask'.

End with a section called 'Talk to Your Healthcare Team About' plus this idea:

{_DISCLAIMER}
"""

    user_content = (
        "Here is the summarized clinical information to base the care-plan outline on:\n"
        "--------------------\n"
        f"{clinical_summary_text}\n"
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

