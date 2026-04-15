"""
Full RAG pipeline: encode → FAISS search → XGBoost re-rank → Mistral LLM.
"""

import os
import re
import csv
import json
import numpy as np
import faiss
from datetime import datetime
from mistralai import Mistral

from config import FEEDBACK_DIR, MISTRAL_MODEL, TOP_K, TOP_N
from rag.reranker import extract_features

FEEDBACK_LOG  = os.path.join(FEEDBACK_DIR, "feedback_log.csv")
FEEDBACK_COLS = ["timestamp", "query", "answer", "rating", "chunk_ids", "comment"]

SYSTEM_PROMPT = (
    "You are a Massachusetts health insurance benefits navigator for plan year 2025. "
    "Use ONLY the context provided. If info is missing, direct to mahealthconnector.org "
    "or 1-877-623-6765. Always mention plan name and carrier. "
    "Premiums are identical for all genders (ACA §2701). "
    "Mention ConnectorCare if the user asks about cost assistance. "
    "The user profile (age, filters) in the prompt already contains all personal info — "
    "NEVER ask the user for their age or any other info; use the profile directly. "
    "Your 'answer' field: 2-3 sentences directly answering the question with key dollar amounts. "
    "Each 'why_ranked_here': 1 sentence max — key reason with one number (e.g. 'Lowest premium at $320/mo'). "
    "Respond ONLY with a valid JSON object — no markdown, no backticks — with this structure: "
    '{"answer": "<2-3 sentence narrative>", '
    '"ranked_plans": [{"rank": 1, "plan_name": "", "carrier": "", '
    '"plan_id": "", "metal_tier": "", "plan_type": "", '
    '"monthly_premium": "", "deductible": "", '
    '"primary_care_copay": "", "specialist_copay": "", '
    '"connector_care": "", '
    '"why_ranked_here": "<one sentence reason>"}]}'
)


class RAGPipeline:
    def __init__(self, index, chunks: list, embedder, xgb_model, mistral_api_key: str) -> None:
        """Initialise the RAG pipeline with pre-loaded index, chunks, and models."""
        self.index     = index
        self.chunks    = chunks
        self.embedder  = embedder
        self.xgb_model = xgb_model
        self.client    = Mistral(api_key=mistral_api_key)

    # Keywords that signal the user is asking about HSA-eligible plans.
    # Matched against the raw question so the enriched query context includes
    # "HSA-eligible" even when the user's UI filter is set to "Any".
    _HSA_KEYWORDS = {"hsa", "health savings", "savings account", "high deductible", "hdhp"}

    def encode_query(self, question: str, filters: dict = None) -> str:
        """Prepend user-profile context to the question for richer semantic search."""
        ctx = []
        if filters:
            if filters.get("age"):
                ctx.append(f"age {filters['age']}")
            if filters.get("tier") and filters["tier"] != "Any":
                ctx.append(filters["tier"])
            if filters.get("carrier") and filters["carrier"] != "Any":
                ctx.append(filters["carrier"])
            if filters.get("connectorcare"):
                ctx.append("ConnectorCare eligible")
            if filters.get("max_premium"):
                ctx.append(f"monthly premium under ${filters['max_premium']}")
        # Auto-detect HSA intent from the question itself
        q_lower = question.lower()
        if any(kw in q_lower for kw in self._HSA_KEYWORDS):
            ctx.append("HSA-eligible")
        return f"[{', '.join(ctx)}] {question}" if ctx else question

    def semantic_search(self, query_text: str, k: int = TOP_K) -> list:
        """Run FAISS nearest-neighbour search and return (chunk, score) pairs."""
        q_emb = self.embedder.encode([query_text]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, k)
        return [
            (self.chunks[i], float(s))
            for s, i in zip(scores[0], indices[0])
            if i < len(self.chunks)
        ]

    def filter_by_cost(self, candidates, age, max_premium):
        """Filter out plan chunks whose premium for this age exceeds max_premium."""
        if not max_premium or not age:
            return candidates
        filtered = []
        for chunk, score in candidates:
            meta = chunk.get("metadata", {})
            if chunk.get("chunk_type") != "plan":
                filtered.append((chunk, score))
                continue
            age_premiums = meta.get("age_premiums", {})
            # Find closest age key
            if age_premiums:
                closest_age = min(age_premiums.keys(), key=lambda a: abs(int(a) - age))
                premium = age_premiums[closest_age]
                if premium <= max_premium:
                    filtered.append((chunk, score))
            else:
                filtered.append((chunk, score))
        # Fallback: if every plan exceeds the budget, return the full candidate list
        # rather than an empty set — the LLM can still explain the cost gap.
        return filtered if filtered else candidates

    def rerank(self, query: str, candidates: list, n: int = TOP_N) -> list:
        """Re-rank FAISS candidates with XGBoost and return the top-n chunks."""
        if not candidates:
            return []
        X_re   = np.array([extract_features(query, c, s) for c, s in candidates])
        scores = self.xgb_model.predict(X_re)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for (c, _), _ in ranked[:n]]

    # Maximum characters accepted from the user's raw question.
    # Truncation happens before enrichment so the context tags are always appended.
    _MAX_QUESTION_LEN = 500  # characters — guards against oversized LLM prompts

    def query(self, question, filters=None):
        """End-to-end RAG: encode → retrieve → filter by cost → re-rank → LLM."""
        question   = question.strip()[:self._MAX_QUESTION_LEN]
        enriched   = self.encode_query(question, filters)
        candidates = self.semantic_search(enriched)

        # Cost filtering before re-ranking
        age         = filters.get("age")   if filters else None
        max_premium = filters.get("max_premium") if filters else None
        if max_premium and max_premium < 1200:  # only filter if user set a real limit
            candidates = self.filter_by_cost(candidates, age, max_premium)

        top_chunks = self.rerank(question, candidates)
        context    = "\n\n---\n\n".join(
            f"[Plan: {c.get('metadata', {}).get('plan_id', '?')}]\n{c['text']}"
            for c in top_chunks
        )

        # Build a plain-English user profile from filters so the LLM never asks again
        profile_parts = []
        if filters:
            if filters.get("age"):
                profile_parts.append(f"Age: {filters['age']}")
            if filters.get("tier") and filters["tier"] != "Any":
                profile_parts.append(f"Preferred metal tier: {filters['tier']}")
            if filters.get("carrier") and filters["carrier"] != "Any":
                profile_parts.append(f"Preferred carrier: {filters['carrier']}")
            if filters.get("connectorcare"):
                profile_parts.append("ConnectorCare eligible: Yes")
            if filters.get("max_premium"):
                profile_parts.append(f"Max monthly premium: ${filters['max_premium']}")
        profile_str = ("User profile: " + ", ".join(profile_parts) + "\n\n") if profile_parts else ""

        user_prompt = (
            f"{profile_str}"
            f"Context:\n{context[:3500]}\n\n"
            f"Question: {question}\n\n"
            "Use the user profile above when answering — do NOT ask for age or other info already provided. "
            "Return ONLY a JSON object (no backticks, no markdown) with keys: "
            "'answer' (narrative string) and 'ranked_plans' (array of plans found, "
            "ranked best to worst for this user, each with: rank, plan_name, carrier, plan_id, "
            "metal_tier, plan_type, monthly_premium, deductible, primary_care_copay, "
            "specialist_copay, connector_care, why_ranked_here)."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        resp = self.client.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages,
            max_tokens=2500,
            temperature=0.1,
        )
        raw       = resp.choices[0].message.content.strip()
        raw_clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            result = json.loads(raw_clean)
        except json.JSONDecodeError:
            # Truncated JSON — try to salvage the answer field via regex
            m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_clean)
            if m:
                answer_text = m.group(1).replace('\\"', '"')
            else:
                answer_text = (
                    "I wasn't able to generate a complete response. "
                    "Please try rephrasing your question or asking about a specific plan."
                )
            result = {"answer": answer_text, "ranked_plans": []}

        result["top_chunks"] = top_chunks
        return result

    def log_feedback(self, query: str, answer: str, rating: int, top_chunks: list, comment: str = "") -> None:
        """Append a user feedback record to the CSV feedback log."""
        row = {
            "timestamp": datetime.now().isoformat(),
            "query":     query,
            "answer":    answer[:300],
            "rating":    rating,
            "chunk_ids": json.dumps([c.get("chunk_id", "") for c in top_chunks]),
            "comment":   comment,
        }
        exists = os.path.exists(FEEDBACK_LOG)
        with open(FEEDBACK_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FEEDBACK_COLS)
            if not exists:
                w.writeheader()
            w.writerow(row)
