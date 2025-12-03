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
            "Explain these medications to an experienced caregiver.\n"
            "- Include mechanism of action, key indications, common and serious\n"
            "  adverse effects, and important drug–drug interactions.\n"
            "- When relevant, mention guideline-based roles (e.g., GDMT in HFrEF)\n"
            "  and typical ICD-10 diagnostic clusters.\n"
            "- Keep it concise but technically detailed.\n"
        )
    return (
        "Explain these medications directly to a patient.\n"
        "- Use simple language: what each medicine is for, when it is usually taken,\n"
        "  and very common side effects in plain words.\n"
        "- Emphasize adherence, not scaring the patient.\n"
        "- Suggest a few questions they could ask their doctor or pharmacist.\n"
    )


_DISCLAIMER = (
    "Do not change, start, or stop any medicines based on this explanation. "
    "Always confirm with the prescribing clinician or pharmacist."
)


def explain_medications(
    mode: str,
    meds_section_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1100,
) -> str:
    """
    Medication explainer bot.

    Parameters
    ----------
    mode : 'patient' or 'caregiver'
    meds_section_text : list/table/text of current meds, doses, frequencies

    Returns
    -------
    Markdown explanation of meds and key points.
    """
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – an AI assistant that explains medication lists.

{persona}

You MUST:
- Respect that the report reflects a clinician's choices; do not contradict them.
- Never give direct dosing instructions or titration schedules.
- Highlight red-flag side effects where urgent care would be important.

End with a short 'Safety Reminder' that paraphrases:

{_DISCLAIMER}
"""

    user_content = (
        "Here is the medication list or section from the report:\n"
        "--------------------\n"
        f"{meds_section_text}\n"
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
