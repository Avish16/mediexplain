# app/rag/rag_engine.py

from openai import OpenAI
from app.rag.retriever import retrieve
from app.rag.config import RAG_LLM_MODEL

client = OpenAI()


def build_prompt(query, docs):
    context = "\n\n---\n\n".join(docs)
    return f"""
You are a medical assistant that ONLY uses the context provided.

CONTEXT:
{context}

QUESTION:
{query}

Give a concise answer with medically accurate information.
If the context does not contain the answer, say "The provided data does not contain that information."
Always include a short disclaimer: "This is not medical advice."
"""


def rag_answer(query: str):
    docs, metas = retrieve(query, k=5)
    prompt = build_prompt(query, docs)

    completion = client.chat.completions.create(
        model=RAG_LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    answer = completion.choices[0].message.content
    return answer, metas
