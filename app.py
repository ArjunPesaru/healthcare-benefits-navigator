"""
MA Health Benefits Navigator — Streamlit App
"""

import os
import torch
import streamlit as st
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY not found. Create a .env file with your key.")
    st.stop()

INDEX_PATH = os.path.join("data", "vectorstore", "faiss.index")
CHUNK_PATH = os.path.join("data", "vectorstore", "chunks.pkl")
MODEL_PATH = os.path.join("models", "xgb_reranker.pkl")


def auto_setup():
    """Run one-time setup if the FAISS index does not exist yet."""
    if not os.path.exists(INDEX_PATH):
        with st.spinner("Building index for the first time — this takes ~2 minutes..."):
            import setup as _setup  # noqa: F401


@st.cache_resource(show_spinner="Loading models and index…")
def load_resources():
    from rag.reranker import load_reranker
    from rag.embeddings import load_index
    from rag.pipeline import RAGPipeline
    xgb_model, _ = load_reranker(MODEL_PATH)
    index, chunks = load_index(INDEX_PATH, CHUNK_PATH)
    embedder = __import__(
        "sentence_transformers", fromlist=["SentenceTransformer"]
    ).SentenceTransformer("all-MiniLM-L6-v2")
    return RAGPipeline(index, chunks, embedder, xgb_model, MISTRAL_API_KEY)


def process_prompt(pipeline, prompt: str, filters: dict):
    """Call the RAG pipeline and append the result to st.session_state.messages."""
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching plans…"):
            result = pipeline.query(prompt, filters)
        answer = result.get("answer", "Sorry, I could not find an answer.")
        st.markdown(answer)
        plans = result.get("ranked_plans", [])
        if plans:
            render_plan_table(plans)
    st.session_state.messages.append({
        "role":   "user",
        "content": prompt,
    })
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "plans":   plans,
    })

st.set_page_config(page_title="MA Health Navigator", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stApp, section[data-testid="stMain"], [data-testid="stAppViewContainer"] {
        background-color: #000000 !important; color: #ffffff !important;
    }
    [data-testid="stHeader"] { background-color: #000000 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111111 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
    }
    p, span, label, div, h1, h2, h3, h4 { color: #ffffff !important; }
    [data-testid="stSelectbox"] > div > div {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
    }
    [data-baseweb="select"] span, [data-baseweb="select"] div,
    [data-baseweb="popover"] ul { background-color: #1a1a1a !important; }
    [data-baseweb="popover"] li { color: #ffffff !important; }
    [data-baseweb="popover"] li:hover { background-color: #2a2a2a !important; }
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: #ffffff !important; }
    [data-testid="stButton"] > button {
        background-color: #1a1a1a !important; color: #ffffff !important;
        border: 1px solid #2e2e2e !important; border-radius: 8px !important;
    }
    [data-testid="stButton"] > button:hover { background-color: #252525 !important; }
    [data-testid="stChatInput"] > div {
        background-color: #1a1a1a !important;
        border: 1px solid #2e2e2e !important; border-radius: 10px !important;
    }
    [data-testid="stChatInput"] textarea { background-color: #1a1a1a !important; color: #ffffff !important; }
    [data-testid="stCaptionContainer"] span { color: #888888 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Massachusetts Health Benefits Navigator")

with st.container(border=True):
    st.markdown(
        '<p style="color:#aaaaaa;font-size:11px;font-weight:600;'
        'letter-spacing:0.1em;margin:0 0 12px 0;">🟢 FILTER PLANS</p>',
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.0, 1.3, 1.8, 1.0])
    with col1:
        age = st.slider("Your Age", 18, 64, 30)
    with col2:
        gender = st.radio("Gender ⓘ", ["Female", "Male"], horizontal=True,
                          help="Premiums identical for all genders (ACA §2701)")
    with col3:
        tier = st.selectbox("Metal Tier", ["Any", "Platinum", "Gold", "Silver", "Bronze"])
    with col4:
        carrier = st.selectbox("Carrier",
                               ["Any", "Blue Cross Blue Shield MA", "Harvard Pilgrim",
                                "Tufts Health", "Fallon", "Health New England",
                                "WellSense", "Mass General Brigham", "UnitedHealthcare"])
    with col5:
        cc = st.checkbox("ConnectorCare eligible ⓘ",
                         help="Subsidised plans for households up to 500% FPL")

def render_plan_table(plans: list):
    """Render ranked plans as a styled DataFrame table."""
    if not plans:
        return
    rows = []
    for p in plans:
        rows.append({
            "Rank":        p.get("rank", ""),
            "Plan":        p.get("plan_name", ""),
            "Carrier":     p.get("carrier", ""),
            "Tier":        p.get("metal_tier", ""),
            "Type":        p.get("plan_type", ""),
            "Premium/mo":  p.get("monthly_premium", ""),
            "Deductible":  p.get("deductible", ""),
            "PCP Copay":   p.get("primary_care_copay", ""),
            "Spec Copay":  p.get("specialist_copay", ""),
            "ConnectorCare": p.get("connector_care", ""),
            "Why":         p.get("why_ranked_here", ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


if "messages"       not in st.session_state: st.session_state.messages       = []
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None

if not st.session_state.messages:
    st.markdown("💡 **Try asking:**")
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
        if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_prompt = s
            st.rerun()

# Re-render previous conversation turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("plans"):
            render_plan_table(msg["plans"])

auto_setup()
pipeline = load_resources()

filters = {
    "age":          age,
    "tier":         tier,
    "carrier":      carrier,
    "connectorcare": cc,
}

# Handle suggestion button click
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    process_prompt(pipeline, prompt, filters)
    st.rerun()

# Handle typed input
if user_input := st.chat_input("Ask about MA health plans…"):
    process_prompt(pipeline, user_input, filters)
