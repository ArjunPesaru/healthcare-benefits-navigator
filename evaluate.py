"""
Offline evaluation of the MA Health Benefits Navigator RAG retrieval pipeline.

Computes NDCG@5, Hit@1, and MRR on a held-out set of evaluation queries
that are distinct from the 15 training queries in rag/reranker.py.

Evaluation covers the retrieval stack only (FAISS semantic search +
XGBoost re-ranking) — no Mistral API calls are made.

Usage:
    python evaluate.py

Requires: python setup.py (run once to build the FAISS index and train the model)
"""

import os
import sys
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Prevent OMP threading conflicts before faiss/torch import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Held-out evaluation queries ───────────────────────────────────────────────
# These 10 queries are NOT in the 15-query training set in rag/reranker.py.
# Each entry: (query_text, expected_tier, expected_carrier_substring, is_connectorcare)
EVAL_QUERIES = [
    ("Bronze HSA eligible plan Massachusetts",     "bronze",   None,               False),
    ("Platinum plan low out-of-pocket maximum",    "platinum", None,               False),
    ("WellSense Silver plan monthly cost",         "silver",   "wellsense",        False),
    ("ConnectorCare income below 150 percent FPL", None,       None,               True),
    ("Mass General Brigham Gold HMO",              "gold",     "mass general",     False),
    ("lowest premium Silver plan for age 55",      "silver",   None,               False),
    ("Bronze plan emergency room copay",           "bronze",   None,               False),
    ("UnitedHealthcare PPO Silver plan",           "silver",   "unitedhealthcare", False),
    ("mental health outpatient benefits coverage", None,       None,               False),
    ("Tufts Health Gold deductible plan",          "gold",     "tufts",            False),
]


def relevance_label(chunk: dict, tier: str, carrier: str, is_cc: bool) -> int:
    """
    Assign a 0–2 relevance label to a retrieved chunk given query intent.

    Relevance scale:
      2 — highly relevant (tier + carrier match, or cc query + cc chunk)
      1 — partially relevant (tier only, or carrier only)
      0 — irrelevant

    Args:
        chunk:    Retrieved chunk dict with a 'metadata' key.
        tier:     Expected metal tier string (e.g. 'silver'), or None.
        carrier:  Expected carrier name substring (e.g. 'tufts'), or None.
        is_cc:    True when the query targets ConnectorCare chunks.

    Returns:
        Integer relevance label in {0, 1, 2}.
    """
    meta = chunk.get("metadata", {})

    if is_cc:
        return 2 if meta.get("chunk_type") == "connectorcare" else 0

    chunk_tier    = meta.get("metal_tier", "").lower()
    chunk_carrier = meta.get("carrier",    "").lower()
    tier_hit    = bool(tier    and tier    == chunk_tier)
    carrier_hit = bool(carrier and carrier in chunk_carrier)

    if tier_hit and carrier_hit:
        return 2
    if tier_hit or carrier_hit:
        return 1
    return 0


def dcg(relevances: list) -> float:
    """
    Compute Discounted Cumulative Gain for an ordered list of relevance labels.

    Args:
        relevances: Relevance labels in retrieval order (best first).

    Returns:
        DCG score (higher is better).
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(
    retrieved: list,
    tier: str,
    carrier: str,
    is_cc: bool,
    k: int = 5,
) -> float:
    """
    Compute Normalised Discounted Cumulative Gain at rank k for one query.

    Args:
        retrieved:  Ordered list of retrieved chunk dicts (best first).
        tier:       Expected metal tier, or None.
        carrier:    Expected carrier substring, or None.
        is_cc:      True for ConnectorCare queries.
        k:          Rank cutoff (default 5).

    Returns:
        NDCG@k in [0.0, 1.0].
    """
    rels      = [relevance_label(c, tier, carrier, is_cc) for c in retrieved[:k]]
    ideal     = sorted(rels, reverse=True)
    idcg      = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def evaluate() -> dict:
    """
    Run the full retrieval evaluation and print a metrics table.

    Returns:
        dict with keys 'mean_ndcg5', 'hit_at_1', 'mrr' (all floats).
    """
    print("=" * 65)
    print("  MA Health Benefits Navigator — Retrieval Pipeline Evaluation")
    print("=" * 65)

    # ── Load pipeline artifacts ───────────────────────────────────────────────
    try:
        from rag.embeddings import load_index
        from rag.reranker   import load_reranker, extract_features
        import faiss
    except ImportError as exc:
        print(f"Import error: {exc}")
        sys.exit(1)

    print("\nLoading index and model...")
    try:
        index, chunks, embedder = load_index()
        xgb_model = load_reranker()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        print("Run 'python setup.py' first to build the index and train the model.")
        sys.exit(1)

    from config import TOP_K, TOP_N
    print(f"  Loaded {len(chunks)} chunks | {index.ntotal} FAISS vectors | TOP_K={TOP_K} TOP_N={TOP_N}")

    # ── Per-query evaluation ──────────────────────────────────────────────────
    ndcg_scores, hits, rrs = [], [], []

    col_w = 52
    print(f"\n{'Query':<{col_w}} {'NDCG@5':>7} {'Hit@1':>6} {'RR':>6}")
    print("-" * (col_w + 22))

    for query, tier, carrier, is_cc in EVAL_QUERIES:
        # Semantic retrieval
        q_emb = embedder.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, TOP_K)
        candidates = [
            (chunks[i], float(s))
            for s, i in zip(scores[0], indices[0])
            if i < len(chunks)
        ]

        # XGBoost re-ranking
        if candidates:
            X         = np.array([extract_features(query, c, s) for c, s in candidates])
            xgb_scores = xgb_model.predict(X)
            ranked    = sorted(zip(candidates, xgb_scores), key=lambda x: x[1], reverse=True)
            top_n     = [c for (c, _), _ in ranked[:TOP_N]]
        else:
            top_n = []

        # Metrics
        score = ndcg_at_k(top_n, tier, carrier, is_cc, k=5)
        ndcg_scores.append(score)

        top1_rel = relevance_label(top_n[0], tier, carrier, is_cc) if top_n else 0
        hits.append(1 if top1_rel > 0 else 0)

        rr = 0.0
        for rank, c in enumerate(top_n, 1):
            if relevance_label(c, tier, carrier, is_cc) > 0:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        label = (query[: col_w - 3] + "...") if len(query) > col_w else query
        print(f"{label:<{col_w}} {score:>7.3f} {top1_rel:>6} {rr:>6.3f}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mean_ndcg = float(np.mean(ndcg_scores))
    hit_at_1  = float(np.mean(hits))
    mrr       = float(np.mean(rrs))

    print("-" * (col_w + 22))
    print(f"\n{'Metric':<35} {'Value':>8}")
    print("-" * 44)
    print(f"{'Mean NDCG@5':<35} {mean_ndcg:>8.4f}")
    print(f"{'Hit@1 Rate':<35} {hit_at_1:>8.4f}")
    print(f"{'Mean Reciprocal Rank (MRR)':<35} {mrr:>8.4f}")
    print(f"{'Queries evaluated':<35} {len(EVAL_QUERIES):>8}")
    print("=" * 65)

    return {"mean_ndcg5": mean_ndcg, "hit_at_1": hit_at_1, "mrr": mrr}


if __name__ == "__main__":
    evaluate()
