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
    "Respond ONLY with a valid JSON object — no markdown, no backticks — with this structure: "
    '{"answer": "<narrative recommendation>", '
    '"ranked_plans": [{"rank": 1, "plan_name": "", "carrier": "", '
    '"plan_id": "", "metal_tier": "", "plan_type": "", '
    '"monthly_premium": "", "deductible": "", '
    '"primary_care_copay": "", "specialist_copay": "", '
    '"connector_care": "", "why_ranked_here": ""}]}'
)


class RAGPipeline:
    def __init__(self, index, chunks, embedder, xgb_model, mistral_api_key):
        self.index     = index
        self.chunks    = chunks
        self.embedder  = embedder
        self.xgb_model = xgb_model
        self.client    = Mistral(api_key=mistral_api_key)

    def encode_query(self, question, filters=None):
        if not filters:
            return question
        ctx = []
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
        return f"[{', '.join(ctx)}] {question}" if ctx else question

    def semantic_search(self, query_text, k=TOP_K):
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
        return filtered if filtered else candidates  # fallback: keep all if none pass

    def rerank(self, query, candidates, n=TOP_N):
        if not candidates:
            return []
        X_re   = np.array([extract_features(query, c, s) for c, s in candidates])
        scores = self.xgb_model.predict(X_re)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for (c, _), _ in ranked[:n]]

    def query(self, question, filters=None):
        """End-to-end RAG: encode → retrieve → filter by cost → re-rank → LLM."""
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

        user_prompt = (
            f"Context:\n{context[:5000]}\n\n"
            f"Question: {question}\n\n"
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
            max_tokens=1200,
            temperature=0.1,
        )
        raw       = resp.choices[0].message.content.strip()
        raw_clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            result = json.loads(raw_clean)
        except json.JSONDecodeError:
            result = {"answer": raw, "ranked_plans": []}

        result["top_chunks"] = top_chunks
        return result

    def log_feedback(self, query, answer, rating, top_chunks, comment=""):
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
