import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_websearch(query: str):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=query,
        tools=[
            {"type": "web_search", "web_search": {"bing_query": {"q": query}}}
        ],
        max_output_tokens=2000
    )
    return response.output_text or ""
