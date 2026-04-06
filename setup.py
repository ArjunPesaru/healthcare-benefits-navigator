"""
One-time setup script.
Run this before launching the Streamlit app:

    python setup.py

What it does:
  1. Builds the MA plan data (CSVs + chunks)
  2. Embeds all chunks and builds the FAISS index
  3. Trains the XGBoost re-ranker model
"""

import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("ERROR: MISTRAL_API_KEY not set.")
    print("  Create a .env file with: MISTRAL_API_KEY=your_key_here")
    sys.exit(1)

print("=" * 55)
print("  MA Health Benefits Navigator — Setup")
print("=" * 55)

# Step 1: Build data
print("\n[1/3] Building plan data...")
from data_builder import run as build_data
_, df_cc, chunks = build_data()

# Step 2: Build FAISS index
print("\n[2/3] Building FAISS vector index...")
from rag.embeddings import build_index
build_index(chunks)

# Step 3: Train re-ranker
print("\n[3/3] Training XGBoost re-ranker...")
from rag.reranker import train_reranker
train_reranker(chunks)

print("\n" + "=" * 55)
print("  Setup complete! Run the app with:")
print("  streamlit run app.py")
print("=" * 55)
