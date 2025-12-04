# app/bots/prescription_bot.py

import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


_client = None


def _get_openai_client() -> OpenAI:
    """
    Shared client with lazy init, same pattern as other bots.
    """
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
    """
    Patient vs Caregiver tone.
    """
    mode = (mode or "").lower()
    if "caregiver" in mode:
        return (
            "Explain these discharge prescriptions to a medically experienced caregiver.\n"
            "- Assume they understand basic pharmacology and chronic disease management.\n"
            "- Organize content by medication with sub-headings:\n"
            "  * Purpose / indication\n"
            "  * How it is usually taken (but do NOT override written instructions)\n"
            "  * Common adverse effects and serious red-flag reactions\n"
            "  * Important drug–drug interactions (especially with home meds)\n"
            "  * What to monitor at home (BP, sugars, weight, symptoms, etc.).\n"
            "- You may mention guideline or ICD-10 style labels briefly in parentheses.\n"
        )

    # default: patient mode
    return (
        "Explain these discharge prescriptions directly to a patient.\n"
        "- Use very clear, simple language.\n"
        "- For each medicine, cover:\n"
        "    • what it is for in everyday words\n"
        "    • roughly when and how often they are usually told to take it\n"
        "    • the most common mild side effects in plain language\n"
        "    • a few serious warning signs when they should call a doctor or go to ER.\n"
        "- Avoid changing any instructions from the written prescription.\n"
        "- Encourage the patient to bring this list to their next visit and ask questions.\n"
    )


_DISCLAIMER = (
    "These explanations do NOT change your prescription. "
    "Always follow the written instructions on the prescription label and the advice "
    "of the prescribing clinician or pharmacist. Do not start, stop, or change doses "
    "based on this summary."
)


def explain_prescriptions(
    mode: str,
    prescriptions_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1300,
) -> str:
    """
    Prescription explainer bot.

    Parameters
    ----------
    mode : 'patient' or 'caregiver'
        Controls tone + depth of explanation.
    prescriptions_text : str
        Raw text of the prescriptions section (drug name, strength, sig,
        quantity, refills, notes, etc.) pasted from the report / EMR.

    Returns
    -------
    Markdown explanation of the prescriptions, ready to show in the chat UI.
    """
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – an AI assistant that helps people understand their
DISCHARGE PRESCRIPTIONS and outpatient medication orders.

{persona}

You MUST:
- Treat the provided text as the source of truth for names, doses, and timing.
- Never invent new medicines or change the dose or frequency.
- Never tell the user to stop a medicine, skip doses, double doses, or share meds.
- Emphasize that all dose decisions belong to their clinical team.
- Call out potential interaction clusters in general terms (e.g., 'blood thinner'
  plus 'anti-inflammatory pain medicine' can raise bleeding risk), but always tell
  them to confirm details with their clinician or pharmacist.

Finish with a short 'Safety Reminder' that paraphrases:

{_DISCLAIMER}
"""

    user_content = (
        "Here is the prescriptions section or medication list to explain.\n"
        "You may see drug names, strengths, SIG instructions, quantity, and refills.\n"
        "--------------------\n"
        f"{prescriptions_text}\n"
        "--------------------\n"
        "Focus ONLY on explaining and contextualizing what is already written here.\n"
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

def run_prescriptions(user_input: str, mode: str, pdf_text: str, memory_snippets):
    """
    Wrapper so the main router can call the prescription explainer
    with the same signature as other bots.
    """
    return explain_prescriptions(mode, pdf_text)
