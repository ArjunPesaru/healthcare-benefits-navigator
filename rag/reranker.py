"""
XGBoost re-ranker for the RAG pipeline.
Trains on labelled queries and re-ranks FAISS candidates.
"""

import os
import re
import pickle
import numpy as np
from xgboost import XGBRanker

from config import MODELS_DIR

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_reranker.pkl")

FEATURE_NAMES = [
    "faiss_score", "keyword_overlap", "tier_match", "carrier_match",
    "type_match", "is_connectorcare", "cc_query", "chunk_length",
    "premium_query", "has_age", "feedback_score",
]

TRAINING_QUERIES = [
    ("Specialist visit copay Silver plan",        "silver",   None,               False),
    ("Monthly premium 40 year old Gold plan",     "gold",     None,               False),
    ("What is ConnectorCare do I qualify",        None,       None,               True),
    ("Compare Bronze HMO plans Massachusetts",    "bronze",   None,               False),
    ("Harvard Pilgrim Silver plan benefits",      "silver",   "harvard pilgrim",  False),
    ("What does Platinum plan cover",             "platinum", None,               False),
    ("BCBS Gold plan cost age 30",                "gold",     "blue cross",       False),
    ("Tufts Bronze plan deductible",              "bronze",   "tufts",            False),
    ("HSA eligible plans Massachusetts",          None,       None,               False),
    ("UnitedHealthcare ER copay",                 None,       "unitedhealthcare", False),
    ("ConnectorCare 250 percent FPL income",      None,       None,               True),
    ("Fallon Health Gold plans",                  "gold",     "fallon",           False),
    ("Prescription drug coverage Silver",         "silver",   None,               False),
    ("WellSense mental health benefits",          None,       "wellsense",        False),
    ("Health insurance cost 60 year old",         None,       None,               False),
]


def extract_features(query, chunk, faiss_score):
    """Extract 11 query-chunk relevance features for re-ranking."""
    text = chunk.get("text", "").lower()
    meta = chunk.get("metadata", {})
    q    = query.lower()
    q_w  = set(q.split())
    t_w  = set(text.split())

    tiers    = ["bronze", "silver", "gold", "platinum", "catastrophic"]
    carriers = ["blue cross", "bcbs", "harvard pilgrim", "tufts", "fallon",
                "health new england", "hne", "wellsense", "mass general", "mgb",
                "unitedhealthcare"]

    q_tier    = next((t for t in tiers    if t in q), None)
    q_carrier = next((c for c in carriers if c in q), None)
    q_type    = next((p for p in ["hmo", "ppo", "epo"] if p in q), None)

    return [
        float(faiss_score),
        len(q_w & t_w) / max(len(q_w), 1),
        1.0 if q_tier    and q_tier    == meta.get("metal_tier", "").lower() else 0.0,
        1.0 if q_carrier and q_carrier in  meta.get("carrier",   "").lower() else 0.0,
        1.0 if q_type    and q_type    == meta.get("plan_type",  "").lower() else 0.0,
        1.0 if meta.get("chunk_type") == "connectorcare" else 0.0,
        1.0 if any(k in q for k in ["connectorcare", "subsidy", "fpl", "income", "afford"]) else 0.0,
        min(len(text) / 2000, 1.0),
        1.0 if any(k in q for k in ["premium", "cost", "price", "monthly", "afford"]) else 0.0,
        1.0 if re.search(r"\b\d{2}\b", q) else 0.0,
        float(meta.get("avg_feedback_score", 0.5)),
    ]


def train_reranker(chunks):
    """Train the XGBoost re-ranker on labelled training queries."""
    X, y, groups = [], [], []

    for query, rel_tier, rel_carrier, is_cc in TRAINING_QUERIES:
        gX, gy = [], []
        for chunk in chunks:
            meta  = chunk.get("metadata", {})
            feats = extract_features(query, chunk, 0.5)
            gX.append(feats)

            label = 0
            if is_cc and meta.get("chunk_type") == "connectorcare":
                label = 2
            elif rel_tier and rel_tier == meta.get("metal_tier", "").lower():
                label = 2 if (rel_carrier and rel_carrier in meta.get("carrier", "").lower()) else 1
            elif rel_carrier and rel_carrier in meta.get("carrier", "").lower():
                label = 1
            gy.append(label)

        X.extend(gX)
        y.extend(gy)
        groups.append(len(gX))

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        tree_method="hist",
        eval_metric="ndcg",
    )
    model.fit(np.array(X), np.array(y), group=np.array(groups), verbose=False)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"  XGBoost re-ranker saved → {MODEL_PATH}")
    return model


def load_reranker():
    """Load the trained XGBoost re-ranker from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Re-ranker model not found at {MODEL_PATH}. Run setup.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
