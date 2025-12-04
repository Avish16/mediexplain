import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


# =========================================================
# OPENAI CLIENT
# =========================================================
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


# =========================================================
# PERSONA
# =========================================================
def _persona_block(mode: str) -> str:
    mode = (mode or "").lower()

    if "caregiver" in mode:
        return (
            "You are explaining laboratory findings to a medically experienced caregiver.\n"
            "- Use standard lab terms (CBC, CMP, BMP, LFTs, troponin, BNP, CRP).\n"
            "- Mention reference ranges when helpful.\n"
            "- Connect abnormalities to clinical implications.\n"
            "- Organize by system: Hematology, Chemistry, Cardiac markers, Inflammation.\n"
        )

    # PATIENT MODE
    return (
        "You are explaining lab results to a patient in simple, friendly language.\n"
        "- Avoid unnecessary numbers.\n"
        "- Explain what each test checks for.\n"
        "- Tell whether it looks normal or not.\n"
        "- Keep it calm and clear.\n"
    )


# =========================================================
# DISCLAIMER
# =========================================================
_DISCLAIMER = (
    "This explanation is only for understanding your lab results. "
    "It does NOT replace medical advice or treatment from your clinician."
)


# =========================================================
# LABS EXPLAINER (CORE FUNCTION)
# =========================================================
def explain_labs(
    mode: str,
    labs_text: str,
    conversation_history: str = "",
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1100
) -> str:

    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – an AI assistant that explains lab results clearly and safely.

{persona}

### Conversation History
{conversation_history}

Rules:
- If labs appear normal, reassure and explain why doctors check them.
- If abnormal, discuss possible concerns WITHOUT diagnosing or prescribing.
- Never invent values that aren't in the text.
- Never give medical orders.

End with a short **Safety Reminder** paraphrasing:
{_DISCLAIMER}
"""

    user_content = f"""
Here are the lab results to explain:

--------------------
{labs_text}
--------------------
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


# =========================================================
# MAIN ENTRYPOINT FOR ORCHESTRATOR
# =========================================================
def run_labs(
    user_input: str,
    mode: str,
    pdf_text: str,
    memory_snippets,
    conversation_history: str = ""
):
    """
    Wrapper used by the orchestrator.
    Currently uses all PDF text as lab context.
    Later you can extract exact lab section by regex.
    """

    labs_section = pdf_text or "No lab data found in the document."

    return explain_labs(
        mode=mode,
        labs_text=labs_section,
        conversation_history=conversation_history
    )
