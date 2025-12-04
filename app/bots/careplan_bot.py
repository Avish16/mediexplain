import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


# ----------------------------------------------------------
# OPENAI CLIENT (cached)
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# PERSONA BLOCK
# ----------------------------------------------------------
def _persona_block(mode: str) -> str:
    mode = (mode or "").lower()
    
    if "caregiver" in mode:
        return (
            "You are outlining a high-level care plan for a medically experienced caregiver.\n"
            "- Use a problem-oriented structure (Problem, Goals, Monitoring, Contingency).\n"
            "- You may reference guideline concepts (e.g., GDMT, clinical escalation) "
            "but avoid prescribing medications.\n"
            "- Mention which specialties might be involved (cardiology, pulmonology, neurology).\n"
        )

    # Patient-friendly mode
    return (
        "You are outlining a simple, supportive care plan for a patient.\n"
        "- Use reassuring, easy-to-understand language.\n"
        "- Focus on themes: medicines, follow-up, lifestyle, warning signs.\n"
        "- Never give specific medication instructions.\n"
        "- Use clear headings and bullet points.\n"
    )


# ----------------------------------------------------------
# DISCLAIMER
# ----------------------------------------------------------
_DISCLAIMER = (
    "This care-plan summary is only a discussion guide. It does not replace "
    "the treatment plan made by the patient’s healthcare team."
)


# ----------------------------------------------------------
# BOT ENTRYPOINT (THIS IS WHAT ORCHESTRATOR CALLS)
# ----------------------------------------------------------
def run_careplan(
    user_input: str,
    mode: str,
    pdf_text: str,
    memory_snippets,
    conversation_history: str = ""
):
    """
    Wrapper called by the orchestrator.  
    It now correctly forwards conversation history into generate_care_plan().
    """
    return generate_care_plan(
        mode=mode,
        clinical_summary_text=pdf_text,
        conversation_history=conversation_history
    )


# ----------------------------------------------------------
# MAIN GENERATION FUNCTION (CONVERSATIONAL VERSION)
# ----------------------------------------------------------
def generate_care_plan(
    mode: str,
    clinical_summary_text: str,
    conversation_history: str = "",
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1200,
) -> str:
    """
    Produces a care plan outline using both clinical summary and conversation history.
    """
    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain’s care-plan assistant.

Your job:
- Help the user understand the *shape* of a care plan using the medical report.
- You must *never* prescribe medications or change doses.
- You must *not* create new diagnoses.
- Adjust your tone based on persona:
{persona}

### Conversation So Far
{conversation_history}

You MUST:
- Stay aligned with information found in the user's medical report.
- Focus on: monitoring, follow-up, lifestyle, red flags, and topics to confirm with doctor.
- Avoid giving medical orders or exact medication changes.

End with a section titled **'Talk to Your Healthcare Team About:'**

Include this statement at the end:

{_DISCLAIMER}
"""

    user_content = (
        "Here is the summarized clinical information to base the care-plan on:\n"
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
