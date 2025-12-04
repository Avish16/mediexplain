# app/rag/config.py

import os

# Base directory (project root: folder that contains app/, html/, etc.)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --------- DATA LOCATIONS ---------
# Folder where your downloaded PMC HTML files live
HTML_DIR = os.path.join(BASE_DIR, "html")

# Folder where Chroma DB will store its files
CHROMA_DB_DIR = os.path.join(BASE_DIR, "mediexplain_chromadb")

# Name of the Chroma collection
COLLECTION_NAME = "MediExplainPMC"

# --------- CHUNKING ---------
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlapping characters between chunks

# --------- MODELS ---------
# Embedding model used by Chroma's OpenAIEmbeddingFunction
EMBED_MODEL = "text-embedding-3-small"
# app/rag/config.py

import os

# Base directory (project root: folder that contains app/, html/, etc.)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --------- DATA LOCATIONS ---------
# Folder where your downloaded PMC HTML files live
HTML_DIR = os.path.join(BASE_DIR, "html")

# Folder where Chroma DB will store its files
CHROMA_DB_DIR = os.path.join(BASE_DIR, "mediexplain_chromadb")

# Name of the Chroma collection
COLLECTION_NAME = "MediExplainPMC"

# --------- CHUNKING ---------
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlapping characters between chunks

# --------- MODELS ---------
# Embedding model used by Chroma's OpenAIEmbeddingFunction
EMBED_MODEL = "text-embedding-3-small"

# Chat model used in RAG
RAG_LLM_MODEL = "gpt-4o-mini"
