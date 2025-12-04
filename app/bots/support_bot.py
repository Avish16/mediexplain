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
            "Provide supportive guidance to a caregiver who already has some\n"
            "medical understanding but may feel overwhelmed.\n"
            "- Acknowledge emotional burden and logistical stress.\n"
            "- Offer practical tips about communication with the care team,\n"
            "  organizing information, and watching for red flags.\n"
            "- Do not provide therapy or mental-health treatment, just supportive\n"
            "  education and validation.\n"
        )
    return (
        "Provide supportive, empathetic guidance directly to the patient.\n"
        "- Validate feelings (worry, confusion, frustration) in a non-clinical tone.\n"
        "- Offer simple, concrete steps for preparing questions, bringing a friend,\n"
        "  or writing things down.\n"
        "- Avoid toxic positivity; stay realistic but hopeful.\n"
    )


_DISCLAIMER = (
    "This is emotional and educational support only, not crisis counseling or "
    "medical advice. In an emergency, the person must contact local emergency services."
)


def generate_support_message(
    mode: str,
    brief_context_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 800,
) -> str:
    """
    Emotional / practical support bot.

    Parameters
    ----------
    mode : 'patient' or 'caregiver'
    brief_context_text : short summary of situation or key diagnoses

    Returns
    -------
    Supportive markdown message.
    """
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – a calm, compassionate assistant.

{persona}

You MUST:
- Avoid making promises about outcomes.
- Encourage the person to lean on their clinical team and social support.
- Gently remind them that online tools cannot replace real clinicians.

End with a short paragraph that rephrases:

{_DISCLAIMER}
"""

    user_content = (
        "Here is the brief context about the situation:\n"
        "--------------------\n"
        f"{brief_context_text}\n"
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

def run_support(user_input: str, mode: str, pdf_text: str, memory_snippets):
    return generate_support_message(mode, pdf_text)
