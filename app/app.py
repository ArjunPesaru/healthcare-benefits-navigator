
import os, json, pickle, csv, re
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from datetime import datetime

BASE         = "/content/drive/MyDrive/Health_Benefits_Navigator"
VECTORSTORE  = f"{BASE}/data/vectorstore"
MODELS       = f"{BASE}/models"
FEEDBACK_LOG = f"{BASE}/data/feedback/feedback_log.csv"
MISTRAL_API_KEY = "kvNPtl3AaaGoGwWG9qka5xst8s8tQI8e"

SYSTEM_PROMPT = (
    "You are a helpful Massachusetts health insurance benefits navigator. "
    "Answer questions about MA health insurance plans for plan year 2025. "
    "Use only the context provided. If not found, direct to mahealthconnector.org "
    "or 1-877-623-6765. Always mention plan name and carrier. "
    "Premiums are identical for all genders (ACA §2701). "
    "Mention ConnectorCare if the user asks about cost assistance."
)

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index    = faiss.read_index(f"{VECTORSTORE}/index.faiss")
    with open(f"{VECTORSTORE}/chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(f"{MODELS}/xgb_reranker.pkl", "rb") as f:
        xgb = pickle.load(f)
    return embedder, index, chunks, xgb, Mistral(api_key=MISTRAL_API_KEY)

embedder, index, chunks, xgb_model, mistral_client = load_resources()
TOP_K, TOP_N = 15, 5


def extract_features(query, chunk, score):
    text = chunk.get("text", "").lower()
    meta = chunk.get("metadata", {})
    q    = query.lower()
    q_w  = set(q.split())
    t_w  = set(text.split())
    tiers    = ["bronze", "silver", "gold", "platinum", "catastrophic"]
    carriers = ["blue cross", "bcbs", "harvard pilgrim", "tufts", "fallon",
                "health new england", "hne", "wellsense", "mass general", "mgb", "unitedhealthcare"]
    q_tier    = next((t for t in tiers    if t in q), None)
    q_carrier = next((c for c in carriers if c in q), None)
    q_type    = next((p for p in ["hmo", "ppo", "epo"] if p in q), None)
    return [
        float(score),
        len(q_w & t_w) / max(len(q_w), 1),
        1.0 if q_tier    and q_tier    == meta.get("metal_tier", "").lower() else 0.0,
        1.0 if q_carrier and q_carrier in  meta.get("carrier",   "").lower() else 0.0,
        1.0 if q_type    and q_type    == meta.get("plan_type",  "").lower() else 0.0,
        1.0 if meta.get("chunk_type") == "connectorcare" else 0.0,
        1.0 if any(k in q for k in ["connectorcare", "subsidy", "fpl", "income", "afford"]) else 0.0,
        min(len(text) / 2000, 1.0),
        1.0 if any(k in q for k in ["premium", "cost", "price", "monthly", "afford"]) else 0.0,
        1.0 if re.search(r"\b\d{2}\b", q) else 0.0,
        0.5,
    ]


def rag(question, filters=None):
    enriched = question
    if filters:
        ctx = []
        if filters.get("age"):
            ctx.append(f"age {filters['age']}")
        if filters.get("tier") and filters["tier"] != "Any":
            ctx.append(filters["tier"])
        if filters.get("carrier") and filters["carrier"] != "Any":
            ctx.append(filters["carrier"])
        if filters.get("connectorcare"):
            ctx.append("ConnectorCare eligible")
        if ctx:
            enriched = f"[{', '.join(ctx)}] {question}"
    q_emb = embedder.encode([enriched]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, TOP_K)
    candidates   = [(chunks[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i < len(chunks)]
    X_re         = np.array([extract_features(question, c, s) for c, s in candidates])
    ranked       = sorted(zip(candidates, xgb_model.predict(X_re)), key=lambda x: x[1], reverse=True)
    top_chunks   = [c for (c, _), _ in ranked[:TOP_N]]
    context      = "\n\n---\n\n".join(c["text"] for c in top_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context[:4000]}\n\nQuestion: {question}"}
    ]
    resp = mistral_client.chat.complete(
        model="mistral-small-latest", messages=messages, max_tokens=512, temperature=0.1
    )
    return resp.choices[0].message.content.strip(), top_chunks


def log_feedback(query, answer, rating, top_chunks):
    row = {
        "timestamp": datetime.now().isoformat(),
        "query":     query,
        "answer":    answer[:300],
        "rating":    rating,
        "chunk_ids": json.dumps([c["chunk_id"] for c in top_chunks]),
        "comment":   "",
    }
    exists = os.path.exists(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# ─── App Layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MA Health Navigator", page_icon="", layout="wide")

with st.sidebar:
    st.title("Filter Plans")
    age     = st.slider("Your Age", 18, 64, 30)
    gender  = st.radio("Gender", ["Female", "Male"],
                       help="Premiums identical for all genders (ACA §2701)")
    tier    = st.selectbox("Metal Tier",
                           ["Any", "Platinum", "Gold", "Silver", "Bronze", "Catastrophic"])
    carrier = st.selectbox("Carrier",
                           ["Any", "Blue Cross Blue Shield MA", "Harvard Pilgrim",
                            "Tufts Health", "Fallon Health", "Health New England",
                            "WellSense", "Mass General Brigham", "UnitedHealthcare"])
    cc      = st.checkbox("ConnectorCare eligible", help="Subsidized plans up to 500% FPL")
    st.divider()
    if os.path.exists(FEEDBACK_LOG):
        fb = pd.read_csv(FEEDBACK_LOG)
        st.subheader("Feedback Summary")
        st.metric("Total Responses", len(fb))
        st.metric("Avg Rating", f"{fb['rating'].mean():.1f} / 5")
    st.divider()
    st.caption("LLM: Mistral AI  |  Data: CMS PUF 2025")

st.title("Massachusetts Health Insurance Navigator")
st.caption("Powered by Mistral AI · RAG + XGBoost Re-ranking · Plan Year 2025")

if "messages"    not in st.session_state: st.session_state.messages    = []
if "last_result" not in st.session_state: st.session_state.last_result = None

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    suggestions = [
        f"What is the premium for a {age}-year-old on a Silver HMO?",
        "What is ConnectorCare and how do I qualify?",
        "Compare Harvard Pilgrim and BCBS Gold plans",
        "Which plans are HSA eligible in Massachusetts?",
        "What is the ER copay on a Bronze plan?",
        "What does a Platinum plan cover vs Gold?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"s{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": s})
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Massachusetts health insurance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching plans..."):
            filters = {"age": age, "gender": gender, "tier": tier,
                        "carrier": carrier, "connectorcare": cc}
            answer, top_chunks = rag(prompt, filters)
            st.session_state.last_result = {
                "query": prompt, "answer": answer, "chunks": top_chunks
            }
        st.markdown(answer)
        if top_chunks:
            with st.expander("Sources"):
                for c in top_chunks:
                    m = c["metadata"]
                    st.caption(
                        f"- {m.get('carrier', '')} — "
                        f"{m.get('plan_name', m.get('plan_type', ''))} | CMS PUF 2025"
                    )
    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.last_result:
    st.divider()
    st.markdown("**Was this answer helpful?**")
    cols = st.columns([1, 1, 1, 1, 1, 4])
    for i, col in enumerate(cols[:5]):
        if col.button(f"{i+1} stars", key=f"r{i}"):
            r = st.session_state.last_result
            log_feedback(r["query"], r["answer"], i + 1, r["chunks"])
            st.success(f"Thank you! {i+1}/5 stars logged.")
