import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None

# RAG search helper
from app.bots.meds_rag_search import search_meds_knowledge

# Set your vector store ID here
MEDS_VECTOR_STORE_ID = "vs_6930ffbfc0188191997f62a2ebe5daf5"

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
            "- Include mechanism, indications, key side effects, and interactions.\n"
            "- Reference guideline logic (e.g., GDMT) when relevant.\n"
        )
    return (
        "Explain these medications to a patient in simple, reassuring language.\n"
        "- Focus on what each medicine does and why it matters.\n"
        "- Avoid medical jargon unless explained.\n"
    )


_DISCLAIMER = (
    "Do not change, start, or stop medicines based on this explanation. "
    "Always confirm with the prescribing clinician."
)


def explain_medications(
    mode: str,
    meds_section_text: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 1100,
) -> str:

    client = _get_openai_client()
    persona = _persona_block(mode)

    system_prompt = f"""
You are MediExplain – an AI assistant that explains medication lists safely.

{persona}

You MUST:
- Never override written prescription instructions.
- Never invent new drug names or change doses.
- Highlight warning signs carefully without scaring the patient.

End with a 'Safety Reminder' paraphrasing:

{_DISCLAIMER}
"""

    user_content = (
        "Medication list + retrieved knowledge:\n"
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


def run_meds(user_input: str, mode: str, pdf_text: str, memory_snippets):
    """
    1. RAG search from medication knowledge PDFs
    2. Fuse PDF context + RAG context
    3. Send to explainer
    """

    # ---- RAG retrieval ----
    rag_context = search_meds_knowledge(
        query=user_input,
        vector_store_id=MEDS_VECTOR_STORE_ID,
        k=6,
    )

    # ---- Combine PDF + RAG ----
    combined_context = pdf_text + "\n\n" + rag_context

    return explain_medications(mode, combined_context)
