# tools/quick_meds_rag_test.py

import os
from openai import OpenAI

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    vector_store_id = os.getenv("MEDS_VECTORSTORE_ID")
    if not vector_store_id:
        raise RuntimeError("MEDS_VECTORSTORE_ID not set")

    client = OpenAI(api_key=api_key)

    query = (
        "What are the common side effects of antihypertensive medications, "
        "and how serious are they?"
    )

    print(f"\n🔍 RAG smoke test")
    print(f"Vector store: {vector_store_id}")
    print(f"Query: {query}\n")

    # NOTE: correct Responses + file_search format:
    # tools = [{ "type": "file_search", "vector_store_ids": [...], "max_num_results": ... }]
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=(
            "You are a RAG debugging assistant.\n\n"
            "User question:\n"
            f"{query}\n\n"
            "Using ONLY the retrieved documents from the vector store, do this:\n"
            "1) Print a numbered list of the *top chunks* you used.\n"
            "2) For each chunk, show:\n"
            "   - source file name\n"
            "   - a short 1–2 sentence summary.\n"
            "3) Then give a short answer (3–5 sentences) to the question.\n"
            "Do NOT make up sources that were not retrieved."
        ),
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 5,
            }
        ],
        max_output_tokens=700,
    )

    print("🧠 Model + RAG answer:\n")
    print(response.output_text)

if __name__ == "__main__":
    main()
