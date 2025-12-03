import os

# Folders
HTML_DIR = "html"
PDF_DIR = "pdfs"
CHROMA_DB_DIR = "mediexplain_chroma"

# Chroma collection name
COLLECTION_NAME = "med_articles"

# RAG chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Models
EMBED_MODEL = "text-embedding-3-large"
RAG_LLM_MODEL = "gpt-4.1-mini"   # fast + good
