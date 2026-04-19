import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None

from app.bots.meds_rag_search import search_meds_knowledge

# ✅ Your prescription/meds vector store ID
PRESCRIPTION_VECTOR_STORE_ID = "vs_6930ffbfc0188191997f62a2ebe5daf5"

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
            "You are explaining DISCHARGE PRESCRIPTIONS to a medically experienced caregiver.\n"
            "- Provide clinical context but do NOT override clinician instructions.\n"
            "- Cover indications, major warnings, interactions, and monitoring needs.\n"
            "- You may mention common guideline concepts at a high level.\n"
        )

    # PATIENT MODE (default)
    return (
        "You are explaining DISCHARGE PRESCRIPTIONS directly to a patient.\n"
        "- Stick closely to the written prescription text.\n"
        "- Do NOT change dose, timing, or instructions.\n"
        "- Use clear, non-technical language and gentle safety reminders.\n"
    )


_DISCLAIMER = (
    "These explanations do NOT change your prescription. "
    "Always follow the written label and your prescribing clinician’s instructions."
)


# =========================================================
# CORE PRESCRIPTION EXPLAINER
# =========================================================
def explain_prescriptions(
    mode: str,
    prescriptions_context_text: str,
    conversation_history: str = "",
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1300,
) -> str:

    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – you explain DISCHARGE PRESCRIPTIONS safely and clearly.

{persona}

### Conversation so far
{conversation_history}

You MUST:
- Treat the provided prescription text as the source of truth.
- Never invent new medicines or instructions.
- Never tell users to change dose, skip doses, or stop medicines.
- Highlight general interaction patterns and red-flag symptoms in broad terms only.
- Encourage users to confirm details with their clinician or pharmacist.

End with a short **Safety Reminder** that paraphrases:
{_DISCLAIMER}
"""

    user_content = (
        "Here is the discharge prescription text and any retrieved medication evidence:\n"
        "--------------------\n"
        f"{prescriptions_context_text}\n"
        "--------------------\n"
        "Explain ONLY what is already present in this text.\n"
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
def run_prescriptions(
    user_input: str,
    mode: str,
    pdf_text: str,
    memory_snippets,
    conversation_history: str = "",
):
    """
    Prescription bot entrypoint used by chat_app.

    Flow:
    1. Use the user's question to query the medication RAG index.
    2. Combine discharge prescription text from the report + external literature.
    3. Send merged context into explain_prescriptions().
    """

    # 1) Retrieve external evidence from your RAG index
    rag_evidence = search_meds_knowledge(
        query=user_input,
        top_k=6,
        vector_store_id=PRESCRIPTION_VECTOR_STORE_ID,
    ) or ""

    # 2) Merge report text and RAG evidence
    combined_context = (
        "=== DISCHARGE PRESCRIPTION TEXT FROM REPORT ===\n"
        f"{(pdf_text or '').strip()}\n\n"
        "=== EVIDENCE FROM MEDICATION SAFETY LITERATURE (RAG) ===\n"
        f"{rag_evidence.strip()}\n"
    )

    # 3) Normalize persona mode
    persona_mode = "caregiver" if "caregiver" in (mode or "").lower() else "patient"

    # 4) Call the prescription explainer LLM
    return explain_prescriptions(
        mode=persona_mode,
        prescriptions_context_text=combined_context,
        conversation_history=conversation_history,
    )
