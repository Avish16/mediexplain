# app/rag/__init__.py

from .ingest import build_index
from .retriever import retrieve

__all__ = ["build_index", "retrieve"]
