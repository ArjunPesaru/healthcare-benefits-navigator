"""
MA Health Benefits Navigator — Streamlit App

Run:  streamlit run app.py
Requires: python setup.py (run once to build index + models)
"""

import os
import torch  # must be imported before xgboost to prevent segfault
import streamlit as st
import pandas as pd

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error(
        "**MISTRAL_API_KEY not found.**  \n"
        "Create a `.env` file in the project root:  \n"
        "`MISTRAL_API_KEY=your_key_here`"
    )
    st.stop()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MA Health Navigator",
    page_icon="🏥",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp,
    section[data-testid="stMain"],
    .stApp > div,
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* ── Header / toolbar ── */
    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }

    /* ── Sidebar (if shown) ── */
    [data-testid="stSidebar"] {
        background-color: #0d0d0d !important;
    }

    /* ── Filter container ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111111 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
    }

    /* ── All text ── */
    p, span, label, div, h1, h2, h3, h4, h5, h6,
    .stMarkdown, [data-testid="stText"] {
        color: #ffffff !important;
    }

    /* ── Slider track & thumb — red accent ── */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background-color: #e05252 !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(1) > div {
        background: linear-gradient(to right, #e05252, #e05252) !important;
    }

    /* ── Selectbox / dropdown ── */
    [data-testid="stSelectbox"] > div > div {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
        color: #ffffff !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    [data-baseweb="popover"] ul {
        background-color: #1a1a1a !important;
    }
    [data-baseweb="popover"] li {
        color: #ffffff !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: #2a2a2a !important;
    }

    /* ── Radio buttons ── */
    [data-testid="stRadio"] label {
        color: #ffffff !important;
    }
    [data-testid="stRadio"] [data-baseweb="radio"] div {
        background-color: transparent !important;
    }

    /* ── Checkbox ── */
    [data-testid="stCheckbox"] label {
        color: #ffffff !important;
    }

    /* ── Suggestion buttons ── */
    [data-testid="stButton"] > button {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        text-align: left !important;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #252525 !important;
        border-color: #444444 !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background-color: #111111 !important;
        border-radius: 10px !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background-color: #1a1a1a !important;
        border-top: 1px solid #2a2a2a !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #1a1a1a !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 10px !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        background-color: #111111 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background-color: #111111 !important;
        border: 1px solid #2a2a2a !important;
    }

    /* ── Caption ── */
    [data-testid="stCaptionContainer"] span {
        color: #888888 !important;
    }

    /* ── Info / alert boxes ── */
    [data-testid="stInfo"] {
        background-color: #1a1a1a !important;
        border-left: 3px solid #e05252 !important;
        color: #ffffff !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] p { color: #aaaaaa !important; }
</style>
""", unsafe_allow_html=True)


# ── Auto-setup if artifacts are missing (runs once on cold start) ─────────────
def _auto_setup():
    from config import VECTORSTORE, MODELS_DIR
    import os
    index_path = os.path.join(VECTORSTORE, "index.faiss")
    model_path = os.path.join(MODELS_DIR,  "xgb_reranker.pkl")
    if os.path.exists(index_path) and os.path.exists(model_path):
        return
    with st.spinner("First-time setup: building index and model (< 2 min)…"):
        from data_builder import run as build_data
        _, _, chunks = build_data()
        from rag.embeddings import build_index
        build_index(chunks)
        from rag.reranker import train_reranker
        train_reranker(chunks)

_auto_setup()


# ── Load Resources (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and index…")
def load_resources():
    from rag.reranker   import load_reranker  # xgb first (torch already primed above)
    from rag.embeddings import load_index
    from rag.pipeline   import RAGPipeline

    xgb_model               = load_reranker()
    index, chunks, embedder = load_index()
    pipeline                = RAGPipeline(index, chunks, embedder, xgb_model, MISTRAL_API_KEY)
    return pipeline


try:
    pipeline = load_resources()
except Exception as e:
    st.error(f"**Failed to load resources:** {e}")
    st.stop()


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🏥 Massachusetts Health Benefits Navigator")

# ── Filter Bar ────────────────────────────────────────────────────────────────
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
        gender = st.radio(
            "Gender ⓘ",
            ["Female", "Male"],
            horizontal=True,
            help="Premiums are identical for all genders under ACA §2701",
        )

    with col3:
        tier = st.selectbox(
            "Metal Tier",
            ["Any", "Platinum", "Gold", "Silver", "Bronze"],
        )

    with col4:
        carrier = st.selectbox(
            "Carrier",
            ["Any", "Blue Cross Blue Shield MA", "Harvard Pilgrim",
             "Tufts Health", "Fallon", "Health New England",
             "WellSense", "Mass General Brigham", "UnitedHealthcare"],
        )

    with col5:
        cc = st.checkbox(
            "ConnectorCare eligible ⓘ",
            help="Subsidised plans for households earning up to 500% FPL",
        )


# ── Render Plan Table ─────────────────────────────────────────────────────────
def render_plan_table(ranked_plans):
    if not ranked_plans:
        return

    col_map = {
        "rank":              "Rank",
        "plan_name":         "Plan Name",
        "carrier":           "Carrier",
        "metal_tier":        "Tier",
        "plan_type":         "Type",
        "monthly_premium":   "Monthly Premium",
        "deductible":        "Deductible",
        "primary_care_copay":"PCP Copay",
        "specialist_copay":  "Specialist",
        "connector_care":    "ConnectorCare",
        "why_ranked_here":   "Why Recommended",
    }

    df = pd.DataFrame(ranked_plans)
    existing = [c for c in col_map if c in df.columns]
    df = df[existing].rename(columns=col_map)

    st.markdown("#### Plan Comparison")

    if "Monthly Premium" in df.columns and "Plan Name" in df.columns:
        top = df.iloc[0]
        if str(top.get("Monthly Premium", "")).strip() not in ("", "N/A", "nan"):
            st.info(
                f"**Top pick:** {top['Plan Name']}  ·  "
                f"**Monthly:** {top['Monthly Premium']}  ·  "
                f"**Tier:** {top.get('Tier','N/A')}  ·  "
                f"**Type:** {top.get('Type','N/A')}"
            )

    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Why these rankings?"):
        for _, row in df.iterrows():
            st.markdown(
                f"- **#{int(row['Rank'])} {row['Plan Name']}** "
                f"({row.get('Tier','?')}, {row.get('Type','?')}): "
                f"{row.get('Why Recommended','')}"
            )


# ── Session State ─────────────────────────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages       = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ── Process Prompt ────────────────────────────────────────────────────────────
def process_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching plans…"):
            filters = {
                "age":           age,
                "gender":        gender,
                "tier":          tier,
                "carrier":       carrier,
                "connectorcare": cc,
            }
            result = pipeline.query(prompt, filters)

        answer       = result.get("answer", "No answer generated.")
        ranked_plans = result.get("ranked_plans", [])
        top_chunks   = result.get("top_chunks", [])

        st.markdown(answer)
        render_plan_table(ranked_plans)

        if top_chunks:
            with st.expander("Sources (CMS PUF 2025)"):
                for c in top_chunks:
                    m = c.get("metadata", {})
                    st.caption(
                        f"• {m.get('carrier', '?')} — "
                        f"{m.get('plan_name', m.get('plan_id', '?'))}"
                    )

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      answer,
        "ranked_plans": ranked_plans,
    })
    pipeline.log_feedback(prompt, answer, rating=None, top_chunks=top_chunks)


# ── Suggestion Buttons ────────────────────────────────────────────────────────
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

# ── Re-render Chat History ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("ranked_plans"):
            render_plan_table(msg["ranked_plans"])

# ── Handle Pending Suggestion ─────────────────────────────────────────────────
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    process_prompt(prompt)

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Massachusetts health insurance…"):
    process_prompt(prompt)
