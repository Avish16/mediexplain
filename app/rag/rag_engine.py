from openai import OpenAI
from app.bots.rag.retriever import retrieve
from app.bots.rag.config import RAG_LLM_MODEL

client = OpenAI()

def build_prompt(query, docs):
    context = "\n\n---\n\n".join(docs)
    return f"""
You are a medical assistant. Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}

Give a structured, concise medical explanation.
"""

def rag_answer(query: str):
    docs, metas = retrieve(query, k=5)
    prompt = build_prompt(query, docs)

    resp = client.chat.completions.create(
        model=RAG_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return resp.choices[0].message.content, metas
