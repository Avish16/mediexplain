# app/rag/config.py

import os

# Package root: mediexplain/ (folder that contains app/, html/, mediexplain_chromadb/, …)
_HERE = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# --------- DATA LOCATIONS ---------
HTML_DIR = os.path.join(BASE_DIR, "html")

CHROMA_DB_DIR = os.path.join(BASE_DIR, "mediexplain_chromadb")

COLLECTION_NAME = "MediExplainPMC"

# --------- CHUNKING ---------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------- MODELS ---------
EMBED_MODEL = "text-embedding-3-small"

# Chat model used in RAG
RAG_LLM_MODEL = "gpt-4o-mini"
