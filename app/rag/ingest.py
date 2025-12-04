# app/rag/ingest.py

import os
import sys
import glob
import logging

# --- Fix for ChromaDB sqlite3 issue (must be BEFORE chromadb import) ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

from app.rag.config import (
    HTML_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def _extract_text_from_html(path: str) -> str:
    """Load an HTML file and convert it to cleaned plain text."""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove <script>, <style>, <noscript> etc.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


def _chunk_text(text: str):
    """Split long text into overlapping character chunks."""
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


def _get_collection(api_key: str):
    """
    Create or load the Chroma collection with an OpenAI embedding function.
    Chroma will compute & store embeddings automatically on .add().
    """
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

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


def build_index(api_key: str | None = None, force_rebuild: bool = False):
    """
    Build the Chroma index from HTML files under HTML_DIR.

    - Reads *.html from html/ folder
    - Extracts text with BeautifulSoup
    - Chunks text
    - Inserts chunks into Chroma (which embeds them)

    If the collection already contains data and force_rebuild=False,
    ingestion is skipped.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set and no api_key passed to build_index()."
        )

    collection = _get_collection(api_key)

    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        logging.info(
            f"Collection '{COLLECTION_NAME}' already has {existing_count} chunks; "
            f"skipping ingestion (set force_rebuild=True to rebuild)."
        )
        return

    if existing_count > 0 and force_rebuild:
        logging.info(
            f"force_rebuild=True: deleting existing collection data "
            f"({existing_count} chunks)."
        )
        # easiest way: delete and recreate collection
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        client.delete_collection(COLLECTION_NAME)
        collection = _get_collection(api_key)

    pattern = os.path.join(HTML_DIR, "*.html")
    paths = glob.glob(pattern)
    logging.info(f"Found {len(paths)} HTML files in {HTML_DIR}")

    if not paths:
        logging.warning("No HTML files found. Did you download them?")
        return

    for path in paths:
        filename = os.path.basename(path)       # e.g. PMC123456.html
        pmcid = filename.replace(".html", "")

        logging.info(f"Processing {filename} ...")

        try:
            text = _extract_text_from_html(path)
            if not text.strip():
                logging.warning(f"No text extracted from {filename}, skipping.")
                continue

            chunks = _chunk_text(text)
            if not chunks:
                logging.warning(f"No chunks created for {filename}, skipping.")
                continue

            ids = [f"{pmcid}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": pmcid, "chunk_index": i} for i in range(len(chunks))
            ]

            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )

            logging.info(f"Inserted {len(chunks)} chunks for {pmcid}")

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    final_count = collection.count()
    logging.info(
        f"✅ Finished building Chroma index from HTML. "
        f"Collection now has {final_count} chunks."
    )


if __name__ == "__main__":
    # Allows: python -m app.rag.ingest  (requires OPENAI_API_KEY env)
    build_index()
