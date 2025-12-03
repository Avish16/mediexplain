import os
import glob
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from app.bots.rag.config import (
    HTML_DIR, CHROMA_DB_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL
)

client = OpenAI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def extract_text_from_html(path):
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

def embed_batch(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in response.data]

def build_index():
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    chroma = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(allow_reset=False)
    )

    try:
        collection = chroma.get_collection(COLLECTION_NAME)
        logging.info("Using existing collection.")
    except Exception:
        collection = chroma.create_collection(COLLECTION_NAME)
        logging.info("Created new collection.")

    paths = glob.glob(os.path.join(HTML_DIR, "*.html"))
    logging.info(f"Found {len(paths)} HTML documents.")

    ids, texts, metas = [], [], []

    for path in tqdm(paths, desc="Processing HTML"):
        pmcid = os.path.basename(path).replace(".html", "")
        text = extract_text_from_html(path)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{pmcid}_{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metas.append({"pmcid": pmcid, "chunk_index": i})

    logging.info(f"Total chunks: {len(texts)}")

    batch = 64
    for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
        b_txt = texts[i:i+batch]
        b_ids = ids[i:i+batch]
        b_meta = metas[i:i+batch]
        vecs = embed_batch(b_txt)

        collection.add(
            ids=b_ids,
            documents=b_txt,
            embeddings=vecs,
            metadatas=b_meta,
        )

    logging.info("RAG index built successfully!")

if __name__ == "__main__":
    build_index()
