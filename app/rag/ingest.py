# app/rag/ingest.py

import os
import glob
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from app.rag.config import (
    HTML_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
)

client = OpenAI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_chroma_client():
    """
    Use DuckDB + Parquet backend for full compatibility with Codespaces.
    """
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_DB_DIR,
        )
    )

def extract_text_from_html(path: str) -> str:
    """
    Clean HTML → plain text.
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def chunk_text(text: str):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0

    return chunks

def embed_batch(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in response.data]

def build_index():
    """
    Build the RAG vector database.
    """
    chroma_client = get_chroma_client()

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        logging.info(f"Using existing collection: {COLLECTION_NAME}")
    except Exception:
        collection = chroma_client.create_collection(COLLECTION_NAME)
        logging.info(f"Created new collection: {COLLECTION_NAME}")

    paths = glob.glob(os.path.join(HTML_DIR, "*.html"))
    logging.info(f"Found {len(paths)} HTML files.")

    if not paths:
        logging.warning("⚠️  No HTML files found. Please download first.")
        return

    ids, texts, metas = [], [], []

    for path in tqdm(paths, desc="Processing HTML"):
        pmcid = os.path.basename(path).replace(".html", "")
        text = extract_text_from_html(path)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{pmcid}_chunk_{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metas.append({"pmcid": pmcid, "chunk_index": i})

    logging.info(f"Total chunks to embed: {len(texts)}")

    batch_size = 64
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        end = start + batch_size
        b_txt = texts[start:end]
        b_ids = ids[start:end]
        b_meta = metas[start:end]

        embeddings = embed_batch(b_txt)

        collection.add(
            ids=b_ids,
            embeddings=embeddings,
            documents=b_txt,
            metadatas=b_meta,
        )

    logging.info("✅ Successfully built RAG index.")


if __name__ == "__main__":
    build_index()
