# app/rag/config.py

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --------- DATA LOCATIONS ---------
HTML_DIR = os.path.join(BASE_DIR, "html")                   # folder containing downloaded PMC HTML files
PDF_DIR = os.path.join(BASE_DIR, "pdfs")                    # optional PDF folder
CHROMA_DB_DIR = os.path.join(BASE_DIR, "mediexplain_chroma")

# --------- COLLECTION NAME ---------
COLLECTION_NAME = "med_articles"

# --------- CHUNKING ---------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------- MODELS ---------
EMBED_MODEL = "text-embedding-3-large"
RAG_LLM_MODEL = "gpt-4.1-mini"
