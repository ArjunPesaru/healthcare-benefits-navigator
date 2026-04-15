"""
Builds and loads the FAISS vector index from text chunks.
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import DATA_PROC, VECTORSTORE

FAISS_INDEX_PATH  = os.path.join(VECTORSTORE, "index.faiss")
CHUNKS_META_PATH  = os.path.join(VECTORSTORE, "chunks_metadata.pkl")
CHUNKS_JSONL_PATH = os.path.join(DATA_PROC,   "chunks.jsonl")


def load_chunks_from_jsonl(path: str = CHUNKS_JSONL_PATH) -> list:
    """Load and return all chunks from a JSONL file, skipping blank lines."""
    chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(chunks=None, model_name="all-MiniLM-L6-v2"):
    """Embed all chunks and build a FAISS index. Saves to disk."""
    if chunks is None:
        chunks = load_chunks_from_jsonl()

    print(f"  Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    texts    = [c["text"] for c in chunks]

    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = embedder.encode(texts[i:i+batch_size], show_progress_bar=False)
        embeddings.extend(batch)
        print(f"\r  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks", end="")
    print()

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  FAISS index saved: {index.ntotal} vectors → {FAISS_INDEX_PATH}")
    return index, chunks, embedder


def load_index() -> tuple:
    """Load the FAISS index, chunk metadata, and sentence-transformer embedder from disk."""
    """Load FAISS index, chunks, and embedder from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. Run setup.py first."
        )
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_META_PATH, "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embedder
