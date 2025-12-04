import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None

# RAG search helper
from app.bots.meds_rag_search import search_meds_knowledge

# Set your vector store ID here
PRESCRIPTION_VECTOR_STORE_ID = 'vs_6930ffbfc0188191997f62a2ebe5daf5'

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
    if "caregiver" in mode:
        return (
            "Explain these discharge prescriptions to a medically experienced caregiver.\n"
            "- Provide clinical context but do not override clinician instructions.\n"
            "- Cover indications, warnings, interactions, and monitoring.\n"
        )
    return (
        "Explain discharge prescriptions in simple words to a patient.\n"
        "- Stick closely to the written prescription.\n"
        "- Do not change dose, timing, or directions.\n"
        "- Provide gentle safety reminders and red flags.\n"
    )


_DISCLAIMER = (
    "These explanations do NOT change your prescription. "
    "Always follow the written label and the prescribing clinician."
)


def explain_prescriptions(
    mode: str,
    prescriptions_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1300,
):

    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – you explain DISCHARGE PRESCRIPTIONS safely.

{persona}

You MUST:
- Treat the provided prescription text as truth.
- Never invent new medicines or instructions.
- Never tell users to change their medicines.
- Highlight general interaction patterns and red-flag symptoms.

End with a 'Safety Reminder' paraphrasing:

{_DISCLAIMER}
"""

    user_content = (
        "Prescription section + retrieved evidence:\n"
        "--------------------\n"
        f"{prescriptions_text}\n"
        "--------------------\n"
        "Explain ONLY what is already in the text.\n"
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
    1. RAG retrieval from medication knowledge PDFs
    2. Fuse PDF context + RAG context
    3. Send to prescription explainer
    """

    # ---- Retrieve drug knowledge ----
    rag_context = search_meds_knowledge(
        query=user_input,
        vector_store_id=PRESCRIPTION_VECTOR_STORE_ID,
        k=6,
    )

    # ---- Fuse PDF + RAG ----
    combined = pdf_text + "\n\n" + rag_context

    return explain_prescriptions(mode, combined)
