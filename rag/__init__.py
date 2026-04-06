"""
RAG package — exposes the three public entry points used by app.py and setup.py.
"""

from rag.embeddings import build_index, load_index
from rag.reranker import train_reranker, load_reranker
from rag.pipeline import RAGPipeline

__all__ = [
    "build_index",
    "load_index",
    "train_reranker",
    "load_reranker",
    "RAGPipeline",
]
