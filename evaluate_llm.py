"""
LLM output quality evaluation for the MA Health Benefits Navigator.

Computes BLEU-1, BLEU-2, ROUGE-1, ROUGE-2, and ROUGE-L scores by comparing
the Mistral LLM's generated answers against human-authored reference answers
for 8 factual questions grounded in the 2025 MA Health Connector plan data.

Evaluation methodology
----------------------
- Reference answers are hand-authored from verified plan data in config.py
- The RAG pipeline generates one answer per question (age-30 neutral profile)
- BLEU uses add-1 (Laplace) smoothing to handle short hypothesis lengths
- ROUGE F1-scores are computed with Porter stemming for robustness
- Scores are averaged across all 8 questions and printed per-query and in aggregate

Usage:
    python evaluate_llm.py

Requires:
    python setup.py          (builds FAISS index and trains re-ranker)
    MISTRAL_API_KEY in .env  (live API call per reference question)
"""

import os
import sys
import warnings
from typing import List

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Run: pip install nltk rouge-score")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Reference Q&A pairs (ground truth from config.py / MA Health Connector) ──
# Each tuple: (question, reference_answer)
# Reference answers are concise (2–4 sentences) to match the LLM output style.
REFERENCE_QA = [
    (
        "What is ConnectorCare and who qualifies for it?",
        (
            "ConnectorCare is a subsidised Massachusetts health insurance programme "
            "for residents earning up to 500 percent FPL with zero dollar deductibles. "
            "Type 1 plans for incomes 0 to 100 percent FPL have a zero dollar monthly "
            "premium and no cost sharing for primary care, preventive care, and mental "
            "health visits."
        ),
    ),
    (
        "Which Bronze plans in Massachusetts are HSA eligible?",
        (
            "Several Bronze plans are HSA eligible including Blue Care Elect PPO Saver, "
            "HMO Blue New England Value, Harvard Pilgrim HMO Saver, Tufts Health Direct "
            "HMO Saver, and UHC Bronze HSA PPO. These plans have high deductibles of "
            "around 4500 to 5500 dollars making them compatible with a health savings account."
        ),
    ),
    (
        "What is the specialist visit copay for Silver plans?",
        (
            "Silver plan specialist visit copays in Massachusetts typically range from "
            "45 to 55 dollars per visit. Harvard Pilgrim HMO charges 45 dollars, "
            "Tufts Health charges 50 dollars, and Blue Cross Silver charges 50 dollars "
            "for specialist visits."
        ),
    ),
    (
        "Are premiums the same for men and women on Massachusetts plans?",
        (
            "Yes, premiums are identical for male and female enrollees under ACA "
            "Section 2701 which prohibits gender-based rate variation. All Massachusetts "
            "marketplace plans comply with this requirement, so men and women pay exactly "
            "the same monthly premium for the same plan."
        ),
    ),
    (
        "What is the deductible for a Platinum plan?",
        (
            "Platinum plans in Massachusetts have very low or zero dollar deductibles. "
            "The Blue Care Elect PPO 0 Platinum plan has a zero dollar deductible with "
            "a 3500 dollar out-of-pocket maximum and 15 dollar primary care copays."
        ),
    ),
    (
        "What is the emergency room copay for Bronze plans?",
        (
            "Bronze plan emergency room copays are 500 dollars per visit in Massachusetts. "
            "Plans such as Blue Care Elect PPO Saver, Harvard Pilgrim HMO Saver, and "
            "Tufts Health Direct HMO Saver all charge 500 dollars for emergency room "
            "services before the deductible is met."
        ),
    ),
    (
        "How much does a ConnectorCare Type 1 plan cost per month?",
        (
            "ConnectorCare Type 1 plans are available to individuals with household "
            "income between 0 and 100 percent FPL and have a zero dollar monthly premium "
            "and a zero dollar deductible. There is no cost sharing for primary care, "
            "preventive care, mental health services, or laboratory tests."
        ),
    ),
    (
        "What is the difference between an HMO and a PPO plan?",
        (
            "HMO plans require referrals from a primary care physician to see specialists "
            "and have lower premiums with restricted in-network coverage. PPO plans allow "
            "direct access to specialists without referrals and permit out-of-network care "
            "at higher cost sharing, typically with higher monthly premiums."
        ),
    ),
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    """Lowercase and whitespace-tokenise a string."""
    return text.lower().split()


def bleu_scores(reference: str, hypothesis: str) -> dict:
    """
    Compute BLEU-1 and BLEU-2 for a hypothesis against one reference.

    Uses add-1 (Laplace) smoothing (SmoothingFunction.method1) to handle
    short or zero-count n-gram cases gracefully.

    Args:
        reference:  Human-authored reference answer string.
        hypothesis: Model-generated answer string.

    Returns:
        dict with keys 'bleu1' and 'bleu2' (both floats in [0, 1]).
    """
    smooth = SmoothingFunction().method1
    ref_tok  = tokenise(reference)
    hyp_tok  = tokenise(hypothesis)
    return {
        "bleu1": sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0),
                               smoothing_function=smooth),
        "bleu2": sentence_bleu([ref_tok], hyp_tok, weights=(0.5, 0.5, 0, 0),
                               smoothing_function=smooth),
    }


def rouge_scores(reference: str, hypothesis: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        reference:  Human-authored reference answer string.
        hypothesis: Model-generated answer string.

    Returns:
        dict with keys 'rouge1', 'rouge2', 'rougeL' (F1 floats in [0, 1]).
    """
    scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = scorer.score(reference, hypothesis)
    return {
        "rouge1": results["rouge1"].fmeasure,
        "rouge2": results["rouge2"].fmeasure,
        "rougeL": results["rougeL"].fmeasure,
    }


# ── Main evaluation routine ───────────────────────────────────────────────────

def evaluate_llm() -> dict:
    """
    Query the RAG pipeline for each reference question, compute LLM output
    quality metrics (BLEU + ROUGE), and print a detailed report.

    Returns:
        dict of mean metric scores: bleu1, bleu2, rouge1, rouge2, rougeL.
    """
    print("=" * 70)
    print("  MA Health Benefits Navigator — LLM Output Quality Evaluation")
    print("=" * 70)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("\nERROR: MISTRAL_API_KEY not set. Create a .env file with your key.")
        sys.exit(1)

    try:
        from rag.embeddings import load_index
        from rag.reranker   import load_reranker
        from rag.pipeline   import RAGPipeline
    except ImportError as exc:
        print(f"Import error: {exc}")
        sys.exit(1)

    print("\nLoading index and model...")
    try:
        index, chunks, embedder = load_index()
        xgb_model               = load_reranker()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        print("Run 'python setup.py' first to build the index and train the model.")
        sys.exit(1)

    pipeline = RAGPipeline(index, chunks, embedder, xgb_model, api_key)
    # Neutral user profile — age 30, no tier/carrier preference
    filters = {"age": 30, "tier": "Any", "carrier": "Any"}

    print(f"  Pipeline ready | {len(chunks)} chunks | {index.ntotal} FAISS vectors\n")

    # ── Per-query evaluation ──────────────────────────────────────────────────
    all_scores: List[dict] = []
    col_w = 48

    print(f"{'Question':<{col_w}} {'B1':>5} {'B2':>5} {'R1':>5} {'R2':>5} {'RL':>5}")
    print("-" * (col_w + 28))

    for i, (question, reference) in enumerate(REFERENCE_QA, 1):
        print(f"  [{i}/{len(REFERENCE_QA)}] Querying: {question[:60]}...", end="\r")
        try:
            result    = pipeline.query(question, filters=filters)
            hypothesis = result.get("answer", "")
        except Exception as exc:
            print(f"\n  WARNING: query {i} failed ({exc}) — skipping")
            continue

        if not hypothesis.strip():
            print(f"\n  WARNING: empty answer for query {i} — skipping")
            continue

        b = bleu_scores(reference, hypothesis)
        r = rouge_scores(reference, hypothesis)
        merged = {**b, **r}
        all_scores.append(merged)

        label = (question[: col_w - 3] + "...") if len(question) > col_w else question
        print(
            f"{label:<{col_w}} "
            f"{b['bleu1']:>5.3f} {b['bleu2']:>5.3f} "
            f"{r['rouge1']:>5.3f} {r['rouge2']:>5.3f} {r['rougeL']:>5.3f}"
        )

    if not all_scores:
        print("No successful queries — cannot compute aggregate metrics.")
        return {}

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    import numpy as np
    means = {k: float(np.mean([s[k] for s in all_scores])) for k in all_scores[0]}

    print("-" * (col_w + 28))
    print(f"\n{'Metric':<30} {'Mean Score':>10}  {'Description'}")
    print("-" * 70)
    print(f"{'BLEU-1':<30} {means['bleu1']:>10.4f}  Unigram precision overlap")
    print(f"{'BLEU-2':<30} {means['bleu2']:>10.4f}  Bigram precision overlap")
    print(f"{'ROUGE-1':<30} {means['rouge1']:>10.4f}  Unigram F1 recall/precision")
    print(f"{'ROUGE-2':<30} {means['rouge2']:>10.4f}  Bigram F1 recall/precision")
    print(f"{'ROUGE-L':<30} {means['rougeL']:>10.4f}  Longest common subsequence F1")
    print(f"\n  Questions evaluated : {len(all_scores)} / {len(REFERENCE_QA)}")
    print(
        "  Note: LLM answers are conversational; scores reflect lexical overlap\n"
        "  with reference answers, not semantic correctness. ROUGE-L >= 0.30\n"
        "  indicates strong factual alignment for open-domain QA tasks."
    )
    print("=" * 70)

    return means


if __name__ == "__main__":
    evaluate_llm()
