import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None

_client = None


# ----------------------------------------------------------
# OPENAI CLIENT
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# PERSONA
# ----------------------------------------------------------
def _persona_block(mode: str) -> str:
    mode = (mode or "").lower()
    
    if "caregiver" in mode:
        return (
            "You are explaining this medical case to a medically experienced caregiver.\n"
            "- Use clinical terminology but remain readable.\n"
            "- Provide structured sections (Diagnosis, Findings, Monitoring, Red Flags).\n"
            "- Reference guideline logic when helpful.\n"
        )
    
    # PATIENT FRIENDLY MODE
    return (
        "You are explaining this medical report to a patient in simple, calm language.\n"
        "- Avoid jargon.\n"
        "- Use short sentences and analogies.\n"
        "- Focus on reassurance, clarity, and understanding.\n"
    )


# ----------------------------------------------------------
# DISCLAIMER
# ----------------------------------------------------------
_DISCLAIMER = (
    "This explanation is for understanding only — it is not a diagnosis or medical advice. "
    "The patient must confirm everything with their licensed healthcare team."
)


# ----------------------------------------------------------
# MAIN GENERATION FUNCTION (CONVERSATIONAL)
# ----------------------------------------------------------
def generate_overall_explanation(
    mode: str,
    report_text: str,
    user_question: str | None = None,
    conversation_history: str = "",
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1200,
) -> str:
    
    client = _get_openai_client()
    persona = _persona_block(mode)

    # Add question block only if present
    if user_question:
        question_part = f"\nThe user is asking specifically:\n\"{user_question}\"\n"
    else:
        question_part = "\nPlease summarize the most important concerns.\n"

    # SYSTEM PROMPT with conversation history
    system_prompt = f"""
You are MediExplain – an AI assistant that explains medical reports clearly.

{persona}

### Conversation History
{conversation_history}

You MUST:
- Stay consistent with the medical report.
- Avoid adding new diagnoses or medication changes.
- Answer in the persona style.
- Use calm, structured, and safe medical explanations.

End with a section titled **Important Reminder** paraphrasing:

{_DISCLAIMER}
"""

    user_content = f"""
Here is the medical report that needs to be explained:

--------------------
{report_text}
--------------------

{question_part}
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_tokens,
    )

    return (response.output_text or "").strip()


# ----------------------------------------------------------
# EXTERNAL ENTRYPOINT (USED BY ORCHESTRATOR)
# ----------------------------------------------------------
def run_explainer(
    mode: str,
    report_text: str,
    user_question: str | None = None,
    conversation_history: str = "",
):
    return generate_overall_explanation(
        mode=mode,
        report_text=report_text,
        user_question=user_question,
        conversation_history=conversation_history
    )
