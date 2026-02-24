"""
Training Data Generation + Train/Test Split
Fixes:
  1. Real RAG scores instead of np.random.uniform
  2. Proper train/test split with evaluation
"""

import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from data_models import UserProfile


# ── Representative users for training ────────────────────────────────────────
# Expanded to 20 users so XGBoost has enough examples to learn from

TRAINING_USERS = [
    # Low income
    UserProfile(24, 18000, 1, "02108", False),
    UserProfile(30, 22000, 2, "02139", False),
    UserProfile(45, 25000, 3, "01060", True),
    UserProfile(55, 20000, 1, "02101", True),
    # Medium income
    UserProfile(28, 45000, 1, "02108", False),
    UserProfile(33, 52000, 2, "02115", False),
    UserProfile(40, 60000, 2, "01105", False),
    UserProfile(35, 70000, 3, "02139", False),
    UserProfile(29, 48000, 1, "02101", True),
    UserProfile(50, 58000, 2, "01060", True),
    UserProfile(38, 65000, 4, "02108", False),
    UserProfile(42, 55000, 3, "02115", True),
    # High income
    UserProfile(35, 90000, 2, "02108", False),
    UserProfile(45, 110000, 4, "02115", False),
    UserProfile(52, 95000, 2, "01105", True),
    UserProfile(60, 85000, 2, "01060", True),
    UserProfile(31, 120000, 1, "02101", False),
    UserProfile(48, 100000, 3, "02139", False),
    UserProfile(27, 88000, 1, "02108", False),
    UserProfile(56, 92000, 2, "01105", True),
]

# Queries that represent real user intents — used to get meaningful RAG scores
TRAINING_QUERIES = [
    "comprehensive coverage with wellness benefits",
    "gym membership fitness benefits",
    "mental health therapy counseling",
    "prescription drug coverage",
    "dental vision coverage",
    "chronic condition disease management",
    "low cost affordable premium plan",
    "transportation to medical appointments",
]


def compute_relevance_score(user: UserProfile, plan, rag_score: float) -> float:
    """
    Deterministic relevance score combining affordability, quality, and RAG signal.
    Range: 0–4 (matches XGBoost rank:pairwise label convention).
    """
    relevance = 2.0

    # Affordability: lower premium relative to income = higher score
    affordability = 1 - min((plan.monthly_premium * 12) / max(user.income, 1), 1.0)
    relevance += affordability * 1.5

    # Quality signal
    relevance += (plan.star_rating - 4.0) * 0.5

    # Benefit richness
    relevance += len(plan.additional_benefits) * 0.1

    # Chronic condition match: reward low OOP max
    if user.has_chronic_conditions and plan.out_of_pocket_max < 6000:
        relevance += 0.5

    # RAG relevance signal (real score, not random)
    relevance += rag_score * 0.8

    return float(min(max(relevance, 0.0), 4.0))


def generate_training_data(plans: List, rag_system) -> List[Dict]:
    """
    Build training examples using REAL RAG scores averaged across multiple queries.
    Each example = one (user, plan) pair with a deterministic relevance label.
    """
    all_plan_ids = [p.plan_id for p in plans]

    # ── Step 1: Get real RAG scores for each query, then average per plan ────
    print("Computing real RAG scores across training queries...")
    aggregated_rag_scores = {pid: [] for pid in all_plan_ids}

    for query in TRAINING_QUERIES:
        scores = rag_system.get_plan_relevance_scores(query, all_plan_ids, top_k=20)
        for pid, score in scores.items():
            aggregated_rag_scores[pid].append(score)

    # Average across all queries → stable signal, not query-dependent noise
    mean_rag_scores = {
        pid: float(np.mean(scores)) if scores else 0.0
        for pid, scores in aggregated_rag_scores.items()
    }

    print(f"  RAG score range: "
          f"{min(mean_rag_scores.values()):.3f} – {max(mean_rag_scores.values()):.3f}")

    # ── Step 2: Build (user, plan) examples ──────────────────────────────────
    training_examples = []

    for user in TRAINING_USERS:
        for plan in plans:
            if not plan.is_eligible(user):
                continue

            rag_score = mean_rag_scores.get(plan.plan_id, 0.0)
            relevance = compute_relevance_score(user, plan, rag_score)

            training_examples.append({
                'user': user,
                'plan': plan,
                'rag_score': rag_score,   # ✅ real score
                'relevance': relevance,
            })

    print(f"Generated {len(training_examples)} training examples")
    return training_examples


def split_training_data(
    training_examples: List[Dict],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split by user groups: 80% of users → train, 20% of users → test.
    This keeps each user's examples together (required for XGBoost group structure)
    and ensures the test set has complete user groups for NDCG computation.
    """
    from collections import defaultdict

    # Group examples by user
    user_groups = defaultdict(list)
    for ex in training_examples:
        key = (ex['user'].age, ex['user'].income)
        user_groups[key].append(ex)

    user_keys = list(user_groups.keys())

    # Reproducible shuffle
    rng = np.random.default_rng(random_state)
    rng.shuffle(user_keys)

    n_test_users = max(2, int(len(user_keys) * test_size))  # at least 2 test users
    test_keys  = set(user_keys[:n_test_users])
    train_keys = set(user_keys[n_test_users:])

    train_examples = [ex for k in train_keys for ex in user_groups[k]]
    test_examples  = [ex for k in test_keys  for ex in user_groups[k]]

    print(f"Split → {len(train_keys)} train users ({len(train_examples)} examples) | "
          f"{len(test_keys)} test users ({len(test_examples)} examples)")
    return train_examples, test_examples