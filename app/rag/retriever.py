import chromadb
from chromadb.config import Settings
from openai import OpenAI

from app.bots.rag.config import (
    CHROMA_DB_DIR, COLLECTION_NAME, EMBED_MODEL
)

client = OpenAI()

def get_collection():
    chroma = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(allow_reset=False)
    )
    return chroma.get_collection(COLLECTION_NAME)

def embed_query(query: str):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query]
    )
    return resp.data[0].embedding

def retrieve(query: str, k: int = 5):
    col = get_collection()

    query_vec = embed_query(query)

    results = col.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas
