"""
build_faiss_index.py
--------------------
Reads all PDFs from data/raw/pdfs/, chunks the text, generates
sentence-transformer embeddings, and saves a FAISS index + metadata
to data/processed/.

Usage:
    python src/build_faiss_index.py
"""

import os
import json
import pickle
import logging
from pathlib import Path

import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR        = Path("data/raw/pdfs")
SCRAPED_DIR    = Path("data/raw/scraped")
PROCESSED_DIR  = Path("data/processed")
FAISS_DIR      = Path("models/faiss")
INDEX_PATH     = FAISS_DIR / "faiss_index.bin"
METADATA_PATH  = PROCESSED_DIR / "chunks_metadata.pkl"

EMBED_MODEL    = "all-MiniLM-L6-v2"   # fast & accurate; swap for a larger model if needed
CHUNK_SIZE     = 500   # characters per chunk
CHUNK_OVERLAP  = 100   # overlap between consecutive chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        log.warning(f"Failed to read {pdf_path.name}: {e}")
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]   # drop tiny trailing chunks


def load_all_pdfs(pdf_dir: Path) -> list[dict]:
    """Load and chunk all PDFs from pdf_dir."""
    records = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDFs found in {pdf_dir}.")
        return records

    log.info(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")
    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            log.warning(f"No extractable text in {pdf_path.name}, skipping.")
            continue
        chunks = chunk_text(raw_text)
        for i, chunk in enumerate(chunks):
            records.append({
                "source":   pdf_path.name,
                "chunk_id": i,
                "text":     chunk,
                "type":     "pdf",
            })
        log.info(f"  {pdf_path.name}: {len(chunks)} chunks")
    return records


def load_scraped_texts(scraped_dir: Path) -> list[dict]:
    """Load and chunk all scraped .txt files from scraped_dir subdirectories."""
    records = []
    txt_files = sorted(scraped_dir.rglob("*.txt"))
    if not txt_files:
        log.warning(f"No scraped text files found in {scraped_dir}.")
        return records

    log.info(f"Found {len(txt_files)} scraped text file(s) in {scraped_dir}")
    for txt_path in tqdm(txt_files, desc="Loading scraped texts"):
        try:
            raw_text = txt_path.read_text(encoding="utf-8")
        except Exception as e:
            log.warning(f"Could not read {txt_path}: {e}")
            continue
        if not raw_text.strip():
            continue
        chunks = chunk_text(raw_text)
        provider = txt_path.parent.name   # folder name = provider
        for i, chunk in enumerate(chunks):
            records.append({
                "source":   txt_path.name,
                "provider": provider,
                "chunk_id": i,
                "text":     chunk,
                "type":     "scraped",
            })
        log.info(f"  {provider}/{txt_path.name}: {len(chunks)} chunks")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def build_index():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & chunk all PDFs + scraped texts
    records = load_all_pdfs(PDF_DIR)
    records += load_scraped_texts(SCRAPED_DIR)
    if not records:
        log.error("No data found from PDFs or scraped files. Aborting.")
        return
    log.info(f"Total chunks across all sources: {len(records)}")

    texts = [r["text"] for r in records]

    # 2. Generate embeddings
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    log.info("Generating embeddings (this may take a moment)…")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    embeddings = embeddings.astype(np.float32)
    log.info(f"Embedding matrix shape: {embeddings.shape}")

    # 3. Build FAISS index
    dim = embeddings.shape[1]
    # IndexFlatIP works for cosine similarity when vectors are normalized
    index = faiss.IndexFlatIP(dim)
    # Wrap with IDMap so we can retrieve by integer ID
    index_with_ids = faiss.IndexIDMap(index)
    ids = np.arange(len(records)).astype(np.int64)
    index_with_ids.add_with_ids(embeddings, ids)

    log.info(f"FAISS index built — total vectors: {index_with_ids.ntotal}")

    # 4. Save index + metadata
    faiss.write_index(index_with_ids, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(records, f)

    log.info(f"Index saved    → {INDEX_PATH}")
    log.info(f"Metadata saved → {METADATA_PATH}")
    log.info("Done! ✓")


# ── Quick search test ─────────────────────────────────────────────────────────

def test_search(query: str, top_k: int = 5):
    """Load the saved index and run a quick test query."""
    if not INDEX_PATH.exists():
        log.error("Index not found. Run build_index() first.")
        return

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        records = pickle.load(f)

    model = SentenceTransformer(EMBED_MODEL)
    q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q_vec, top_k)
    print(f"\nTop {top_k} results for: '{query}'\n{'─'*60}")
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
        r = records[idx]
        preview = r["text"][:200].replace("\n", " ")
        print(f"[{rank}] score={score:.4f} | source={r['source']} | chunk={r['chunk_id']}")
        print(f"    {preview}…\n")


if __name__ == "__main__":
    build_index()

    # Uncomment to test after building:
    # test_search("free gym membership benefits")
    # test_search("mental health resources covered near me")
    # test_search("grocery allowance for seniors")