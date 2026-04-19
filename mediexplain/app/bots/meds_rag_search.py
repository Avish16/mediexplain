# app/bots/meds_rag_search.py

import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None

_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """Lazy-init OpenAI client using env or Streamlit secrets."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and st is not None:
            api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set it in env, .env, or Streamlit secrets."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def search_meds_knowledge(
    query: str,
    top_k: int = 5,
    vector_store_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a RAG search over your medication PDF vector store.

    Returns a dict:
    {
      "answer": str,          # fused explanation for the user
      "chunks": [             # list of retrieved evidence chunks
         {
           "rank": int,
           "score": float,
           "source": str,     # filename or paper title if present
           "doc_id": str,     # some identifier (we use filename + rank)
           "snippet": str
         },
         ...
      ]
    }
    """
    if not vector_store_id:
        raise ValueError("vector_store_id is required for medication RAG search")

    client = get_openai_client()

    system_prompt = """
You are MediExplain's medication RAG engine.

You have access to a file_search tool over medical research PDFs
about medication side effects, safety, and monitoring.

Your job:
1. Use file_search to retrieve the most relevant passages for the question.
2. Synthesize them into a clear educational answer.
3. ALSO return a JSON structure listing the top evidence chunks.

You MUST respond ONLY with valid JSON of this form:

{
  "answer": "final explanation for the user, grounded in the PDFs",
  "chunks": [
    {
      "rank": 1,
      "score": 0.99,
      "source": "paper_file_name_or_title.pdf",
      "doc_id": "some-doc-id-or-name",
      "snippet": "short quoted or paraphrased passage"
    }
  ]
}

- 'score' can be an approximate relevance score between 0 and 1 (you may estimate).
- 'source' should reflect the file name or paper title you see.
- 'snippet' should be a short passage (1–3 sentences) that supports the answer.
"""

    user_prompt = f"""
User question about medications:

\"\"\"{query}\"\"\"

Retrieve up to {top_k} of the most relevant passages.
Ground the answer STRICTLY in the retrieved content.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[
            {
                "type": "file_search",
                # 🔥 IMPORTANT: vector_store_ids is TOP-LEVEL on the tool
                "vector_store_ids": [vector_store_id],
            }
        ],
        max_output_tokens=900,
    )

    text = (response.output_text or "").strip()

    # Try to parse the JSON the model returns
    try:
        data = json.loads(text)
    except Exception:
        # Fallback: treat the whole text as the answer, no structured chunks
        data = {
            "answer": text,
            "chunks": [],
        }

    # Ensure keys exist with the right types
    if "answer" not in data or not isinstance(data["answer"], str):
        data["answer"] = text

    if "chunks" not in data or not isinstance(data["chunks"], list):
        data["chunks"] = []

    # Normalize chunks to expected fields
    norm_chunks: List[Dict[str, Any]] = []
    for i, c in enumerate(data["chunks"]):
        if not isinstance(c, dict):
            continue
        norm_chunks.append(
            {
                "rank": int(c.get("rank", i + 1)),
                "score": float(c.get("score", 1.0 - 0.05 * i)),
                "source": str(c.get("source", "unknown_source")),
                "doc_id": str(c.get("doc_id", f"doc_{i+1}")),
                "snippet": str(c.get("snippet", "")),
            }
        )

    data["chunks"] = norm_chunks
    return data
