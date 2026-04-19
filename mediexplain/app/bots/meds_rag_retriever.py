import os
from openai import OpenAI

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

def retrieve_med_chunks(query: str):
    """
    Uses the pre-built medication vector store to retrieve relevant passages.
    """
    client = get_client()
    vector_store_id = os.getenv("MEDS_VECTOR_STORE_ID")
    
    if not vector_store_id:
        raise RuntimeError("MEDS_VECTOR_STORE_ID missing in .env")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Retrieve medical guideline passages relevant to:\n{query}",
        tools=[{
            "type": "file_search",
            "file_search": {
                "vector_store_ids": [vector_store_id],
                "max_num_results": 6
            }
        }],
        max_output_tokens=600,
    )

    return response.output_text or ""
