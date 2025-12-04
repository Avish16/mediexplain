# app/rag/retriever.py

import os
import sys

# --- Fix for ChromaDB sqlite3 issue (must be BEFORE chromadb import) ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils import embedding_functions

from app.rag.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
)


def _get_collection(api_key: str):
    """Return the Chroma collection, ready for text queries."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


def retrieve(query: str, k: int = 5, api_key: str | None = None):
    """
    Retrieve top-k relevant chunks for a text query.

    Returns:
        (documents, metadatas)
        where each is a list of length k
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set and no api_key passed to retrieve()."
        )

    collection = _get_collection(api_key)

    results = collection.query(
        query_texts=[query],
        n_results=k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas
