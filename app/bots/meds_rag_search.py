# app/bots/meds_rag_search.py
import os
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


def _get_client():
    api_key = os.getenv("OPENAI_API_KEY") or (
        st.secrets["OPENAI_API_KEY"] if st is not None else None
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def search_meds_knowledge(
    query: str,
    vector_store_id: str,
    k: int = 6,
) -> str:
    """
    Search the RAG vector store containing medical research PDFs
    (side-effects, pharmacology, guidelines).

    Returns: fused text summary of retrieved chunks.
    """

    if not vector_store_id:
        return ""

    client = _get_client()

    rag_prompt = f"""
Retrieve medical evidence relevant to this medication question:

"{query}"

Summarize ONLY what is in the retrieved documents.
2–4 paragraphs. Do not invent new facts.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=rag_prompt,
        tools=[
            {
                "type": "file_search",
                "file_search": {
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": k,
                },
            }
        ],
        max_output_tokens=700,
    )

    return response.output_text or ""
