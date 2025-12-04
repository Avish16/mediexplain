import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None

# RAG search helper
from app.bots.meds_rag_search import search_meds_knowledge

# ✅ Your meds vector store ID
MEDS_VECTOR_STORE_ID = "vs_6930ffbfc0188191997f62a2ebe5daf5"

_client = None


# =========================================================
# OPENAI CLIENT
# =========================================================
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
            "You are explaining these medications to a medically experienced caregiver.\n"
            "- Include mechanism of action, typical indications, and major side effects.\n"
            "- Mention common interaction concerns in general terms.\n"
            "- Do NOT give exact prescribing instructions or change the regimen.\n"
        )

    # PATIENT MODE (default)
    return (
        "You are explaining these medications to a patient in simple, reassuring language.\n"
        "- Focus on what each medicine is for and why it matters.\n"
        "- Avoid heavy jargon; if you must use a medical word, explain it.\n"
        "- Highlight key side effects and safety warnings without causing panic.\n"
    )


_DISCLAIMER = (
    "Do not start, stop, or change any medication based on this explanation. "
    "Always confirm with the prescribing clinician or pharmacist."
)


# =========================================================
# CORE MEDICATION EXPLAINER
# =========================================================
def explain_medications(
    mode: str,
    meds_context_text: str,
    conversation_history: str = "",
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1200,
) -> str:

    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – an AI assistant that explains medication information safely.

{persona}

### Conversation so far
{conversation_history}

You MUST:
- Use ONLY the medication names and facts present in the provided text.
- Never invent new medicines, doses, or instructions.
- Never advise dose changes, skipping doses, or stopping treatment.
- Emphasize that final decisions belong to the clinician.

End with a short **Safety Reminder** paraphrasing:
{_DISCLAIMER}
"""

    user_content = (
        "Here is the medication-related information to explain. It may include:\n"
        "- Text from the patient's medical report\n"
        "- Retrieved literature about side effects and safety\n"
        "--------------------\n"
        f"{meds_context_text}\n"
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


# =========================================================
# ENTRYPOINT FOR ORCHESTRATOR
# =========================================================
def run_meds(
    user_input: str,
    mode: str,
    pdf_text: str,
    memory_snippets,
    conversation_history: str = "",
):
    """
    Medication bot entrypoint used by chat_app.

    Flow:
    1. Use the user's question to query the medication RAG index.
    2. Combine report text (pdf_text) + RAG evidence.
    3. Send merged context into explain_medications().
    """

    # 1) RAG retrieval from your meds vector store
    rag_context = search_meds_knowledge(
        query=user_input,
        top_k=6,
        vector_store_id=MEDS_VECTOR_STORE_ID,
    )

    # 2) Build combined context
    combined_context = (
        "=== MEDICATION-RELATED TEXT FROM REPORT ===\n"
        f"{(pdf_text or '').strip()}\n\n"
        "=== EVIDENCE FROM MEDICATION SAFETY LITERATURE (RAG) ===\n"
        f"{rag_context.strip()}\n"
    )

    # 3) Normalize mode string for persona
    persona_mode = "caregiver" if "caregiver" in (mode or "").lower() else "patient"

    # 4) Call the explainer
    return explain_medications(
        mode=persona_mode,
        meds_context_text=combined_context,
        conversation_history=conversation_history,
    )
