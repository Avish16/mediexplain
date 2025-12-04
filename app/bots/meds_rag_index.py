# app/bots/meds_rag_index.py
"""
One-time script to build a vector store for medication side-effect PDFs.

Usage (from project root in Codespaces):

    python app/bots/meds_rag_index.py

It will:
- find all PDFs in app/bots/Data/
- create an OpenAI vector store
- upload + index the PDFs
- print the vector_store_id  → copy this and hardcode / put in secrets
"""

import os
import glob
from openai import OpenAI

try:
    import streamlit as st
except ImportError:
    st = None


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client.

    Priority:
    1. OPENAI_API_KEY from environment
    2. OPENAI_API_KEY from .streamlit/secrets.toml (same as your bots)
    """
    api_key = os.getenv("OPENAI_API_KEY")

    # Fall back to Streamlit secrets (like your other bots)
    if (not api_key) and (st is not None):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set.\n"
            "- Either export it in the terminal, e.g.\n"
            "    export OPENAI_API_KEY='your-key-here'\n"
            "- Or put it in .streamlit/secrets.toml as OPENAI_API_KEY."
        )

    return OpenAI(api_key=api_key)


def main():
    client = get_openai_client()

    # 1) Locate all the medication-knowledge PDFs
    base_dir = os.path.dirname(__file__)              # app/bots
    data_dir = os.path.join(base_dir, "Data/Research_Papers")         # app/bots/Data
    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))

    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in: {data_dir}")

    print("📂 Found the following PDFs to index:")
    for p in pdf_paths:
        print("   -", os.path.basename(p))

    # 2) Create a dedicated vector store for meds knowledge
    vs_name = "mediexplain_meds_knowledge_vs"
    print(f"\n🧠 Creating vector store: {vs_name} ...")
    vector_store = client.vector_stores.create(name=vs_name)
    vector_store_id = vector_store.id
    print("✅ Vector store created with ID:", vector_store_id)

    # 3) Upload + index all PDFs
    print("\n📤 Uploading PDFs into vector store (this may take a bit)...")

    file_streams = [open(p, "rb") for p in pdf_paths]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=file_streams,
        )
    finally:
        # Always close file handles
        for f in file_streams:
            try:
                f.close()
            except Exception:
                pass

    print("✅ Upload + indexing complete.")
    print("   Status:", batch.status)
    print("   File counts:", batch.file_counts)

    # 4) FINAL: tell you what to copy into your app
    print("\n🚀 IMPORTANT: save this vector_store_id somewhere safe.")
    print("You’ll use it in your meds/prescription bots for RAG queries.\n")
    print("VECTOR_STORE_ID =", vector_store_id)


if __name__ == "__main__":
    main()
