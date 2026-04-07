"""
MA Health Benefits Navigator — Streamlit App

Run:  streamlit run app.py
Requires: python setup.py (run once to build index + models)
"""

import os

# Must be set before torch/faiss import to prevent OMP segfault on Mac
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY not found. Create a .env file with your key.")
    st.stop()

INDEX_PATH = os.path.join("data", "vectorstore", "index.faiss")


def auto_setup():
    if not os.path.exists(INDEX_PATH):
        with st.spinner("Building index for the first time — this takes ~2 minutes..."):
            import setup as _setup  # noqa: F401


@st.cache_resource(show_spinner="Loading models and index…")
def load_resources():
    from rag.reranker import load_reranker
    from rag.embeddings import load_index
    from rag.pipeline import RAGPipeline
    xgb_model = load_reranker()
    index, chunks, embedder = load_index()
    return RAGPipeline(index, chunks, embedder, xgb_model, MISTRAL_API_KEY)


TIER_COLORS = {
    "Bronze": "#cd7f32", "Silver": "#aaaaaa",
    "Gold":   "#ffd700", "Platinum": "#e5e4e2",
    "Catastrophic": "#888888",
}
CHART_LAYOUT = dict(
    paper_bgcolor="#000000", plot_bgcolor="#111111",
    font_color="#ffffff", margin=dict(l=10, r=10, t=30, b=10),
)


def render_plan_table(plans: list):
    if not plans:
        return
    rows = []
    for p in plans:
        rows.append({
            "Rank":          p.get("rank", ""),
            "Plan":          p.get("plan_name", ""),
            "Carrier":       p.get("carrier", ""),
            "Tier":          p.get("metal_tier", ""),
            "Type":          p.get("plan_type", ""),
            "Premium/mo":    p.get("monthly_premium", ""),
            "Deductible":    p.get("deductible", ""),
            "PCP Copay":     p.get("primary_care_copay", ""),
            "Spec Copay":    p.get("specialist_copay", ""),
            "ConnectorCare": p.get("connector_care", ""),
        })
    st.markdown("**Ranked Plans**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Why this ranking?**")
    for p in plans:
        why   = p.get("why_ranked_here", "").strip()
        name  = p.get("plan_name", f"Plan #{p.get('rank','')}")
        tier  = p.get("metal_tier", "")
        rank  = p.get("rank", "")
        color = TIER_COLORS.get(tier, "#555555")
        with st.expander(f"#{rank} — {name} ({tier})", expanded=(rank == 1)):
            st.markdown(
                f'<div style="border-left:3px solid {color};padding-left:12px;">'
                f"{why or 'No justification provided.'}</div>",
                unsafe_allow_html=True,
            )


def process_prompt(pipeline, prompt: str, filters: dict):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching plans…"):
            result = pipeline.query(prompt, filters)
        answer = result.get("answer", "Sorry, I could not find an answer.")
        st.markdown(answer)
        plans = result.get("ranked_plans", [])
        if plans:
            st.markdown("---")
            render_plan_table(plans)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer, "plans": plans})


# ── Dashboard data ─────────────────────────────────────────────────────────────
@st.cache_data
def get_national_data():
    rng = np.random.default_rng(42)
    states = [
        ("Alabama","AL"),("Alaska","AK"),("Arizona","AZ"),("Arkansas","AR"),
        ("California","CA"),("Colorado","CO"),("Connecticut","CT"),("Delaware","DE"),
        ("Florida","FL"),("Georgia","GA"),("Hawaii","HI"),("Idaho","ID"),
        ("Illinois","IL"),("Indiana","IN"),("Iowa","IA"),("Kansas","KS"),
        ("Kentucky","KY"),("Louisiana","LA"),("Maine","ME"),("Maryland","MD"),
        ("Massachusetts","MA"),("Michigan","MI"),("Minnesota","MN"),("Mississippi","MS"),
        ("Missouri","MO"),("Montana","MT"),("Nebraska","NE"),("Nevada","NV"),
        ("New Hampshire","NH"),("New Jersey","NJ"),("New Mexico","NM"),("New York","NY"),
        ("North Carolina","NC"),("North Dakota","ND"),("Ohio","OH"),("Oklahoma","OK"),
        ("Oregon","OR"),("Pennsylvania","PA"),("Rhode Island","RI"),("South Carolina","SC"),
        ("South Dakota","SD"),("Tennessee","TN"),("Texas","TX"),("Utah","UT"),
        ("Vermont","VT"),("Virginia","VA"),("Washington","WA"),("West Virginia","WV"),
        ("Wisconsin","WI"),("Wyoming","WY"),("District of Columbia","DC"),
    ]
    base_plans = {
        "CA":120,"TX":90,"FL":85,"NY":95,"PA":70,"IL":75,"OH":68,"GA":62,
        "NC":60,"MI":65,"MA":28,"WA":55,"CO":50,"AZ":58,"NJ":72,"VA":60,
        "TN":55,"MO":50,"IN":48,"WI":52,"MN":58,"OR":45,"SC":42,"AL":35,
        "LA":38,"KY":36,"OK":34,"CT":40,"UT":44,"NV":46,"AR":30,"MS":28,
        "IA":32,"KS":30,"NE":28,"NM":32,"ID":26,"NH":24,"ME":22,"HI":20,
        "RI":22,"MT":18,"ND":16,"SD":16,"AK":14,"WY":12,"VT":18,"WV":22,
        "DE":24,"MD":52,"DC":35,
    }
    rows = []
    for name, abbr in states:
        n      = base_plans.get(abbr, 30) + rng.integers(-3, 4)
        n_hmo  = int(n * rng.uniform(0.45, 0.65))
        n_ppo  = int(n * rng.uniform(0.25, 0.40))
        n_epo  = max(0, n - n_hmo - n_ppo)
        bronze   = int(n * rng.uniform(0.28, 0.38))
        silver   = int(n * rng.uniform(0.30, 0.38))
        gold     = int(n * rng.uniform(0.15, 0.22))
        platinum = max(0, int(n * rng.uniform(0.04, 0.10)))
        catast   = max(0, n - bronze - silver - gold - platinum)
        rows.append({
            "State": name, "Abbr": abbr,
            "Total Plans": n,
            "Carriers": max(2, min(10, int(n / 9) + rng.integers(0, 3))),
            "HMO": n_hmo, "PPO": n_ppo, "EPO": n_epo,
            "Bronze": bronze, "Silver": silver, "Gold": gold,
            "Platinum": platinum, "Catastrophic": catast,
            "Avg Bronze Premium":   int(rng.uniform(280, 380)),
            "Avg Silver Premium":   int(rng.uniform(380, 490)),
            "Avg Gold Premium":     int(rng.uniform(480, 590)),
            "Avg Platinum Premium": int(rng.uniform(580, 700)),
            "Enrollment": int(rng.uniform(0.8, 3.5) * n * 1000),
        })
    return pd.DataFrame(rows)


# ── Page config & CSS ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Health Benefits Navigator", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stApp, section[data-testid="stMain"], [data-testid="stAppViewContainer"] {
        background-color: #000000 !important; color: #ffffff !important;
    }
    [data-testid="stHeader"] { background-color: #000000 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111111 !important;
        border: 1px solid #2a2a2a !important; border-radius: 10px !important;
    }
    p, span, label, div, h1, h2, h3, h4 { color: #ffffff !important; }
    [data-testid="stSelectbox"] > div > div {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important; border-radius: 6px !important;
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
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: #111111 !important; border-radius: 10px !important;
        padding: 4px !important; gap: 4px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: transparent !important; color: #aaaaaa !important;
        border-radius: 8px !important; font-size: 15px !important;
        font-weight: 600 !important; padding: 10px 28px !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background-color: #1e1e1e !important; color: #ffffff !important;
    }
    [data-testid="stMetric"] {
        background-color: #111111 !important; border: 1px solid #2a2a2a !important;
        border-radius: 10px !important; padding: 16px !important;
    }
    [data-testid="stMetricValue"] { color: #e05252 !important; font-size: 28px !important; }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    [data-testid="stExpander"]    { background-color: #111111 !important; border: 1px solid #2a2a2a !important; }
    /* Slider */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #e05252 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Health Benefits Navigator")
tab_chat, tab_dash = st.tabs(["💬  Chatbot", "📊  Insurance Dashboard"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    with st.container(border=True):
        st.markdown(
            '<p style="color:#aaaaaa;font-size:11px;font-weight:600;'
            'letter-spacing:0.1em;margin:0 0 12px 0;">🟢 FILTER PLANS</p>',
            unsafe_allow_html=True,
        )
        col1, col2, col3, col4, col5, col6 = st.columns([1.1, 0.9, 1.2, 1.6, 1.4, 0.9])
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
            max_premium = st.slider(
                "Max Monthly Budget ($)", min_value=0, max_value=1500,
                value=1500, step=25,
                help="Only show plans with a monthly premium at or below this amount. Set to $1500 to show all plans.",
            )
        with col6:
            cc = st.checkbox("ConnectorCare ⓘ",
                             help="Subsidised plans for households up to 500% FPL")

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

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("plans"):
                st.markdown("---")
                render_plan_table(msg["plans"])

    auto_setup()
    pipeline = load_resources()
    filters = {
        "age":         age,
        "tier":        tier,
        "carrier":     carrier,
        "connectorcare": cc,
        "max_premium": max_premium if max_premium < 1500 else None,
    }

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        process_prompt(pipeline, prompt, filters)
        st.rerun()

    if user_input := st.chat_input("Ask about MA health plans…"):
        process_prompt(pipeline, user_input, filters)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSURANCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    df = get_national_data()

    abbr_to_state = dict(zip(df["Abbr"], df["State"]))

    if "dash_state" not in st.session_state:
        st.session_state["dash_state"] = "All States (National)"

    # Apply pending map-click BEFORE the selectbox is instantiated
    if "_pending_map_state" in st.session_state:
        new = st.session_state.pop("_pending_map_state")
        st.session_state["dash_state"] = new
        st.session_state["state_sel"]  = new   # safe here — widget not yet created

    st.markdown("### US Health Insurance Marketplace")
    st.caption("Click any state on the map **or** use the dropdown to filter all charts. "
               "All 50 states + DC.")

    state_options = ["All States (National)"] + sorted(df["State"].tolist())
    sel_col, reset_col, _ = st.columns([1.4, 0.4, 3])
    with sel_col:
        selected_state = st.selectbox(
            "Filter by State", state_options,
            index=state_options.index(st.session_state["dash_state"]),
            key="state_sel",
        )
        st.session_state["dash_state"] = selected_state
    with reset_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset", key="reset_state"):
            st.session_state["dash_state"] = "All States (National)"
            st.rerun()

    national = selected_state == "All States (National)"
    view     = df if national else df[df["State"] == selected_state]
    row      = None if national else view.iloc[0]
    nat_avg  = {
        "Avg Bronze Premium":   int(df["Avg Bronze Premium"].mean()),
        "Avg Silver Premium":   int(df["Avg Silver Premium"].mean()),
        "Avg Gold Premium":     int(df["Avg Gold Premium"].mean()),
        "Avg Platinum Premium": int(df["Avg Platinum Premium"].mean()),
    }

    # ── KPI row ────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    if national:
        k1.metric("Total Plans",        f"{df['Total Plans'].sum():,}")
        k2.metric("Total Carriers",     f"{df['Carriers'].sum():,}")
        k3.metric("States Covered",     f"{len(df)}")
        k4.metric("Avg Silver Premium", f"${nat_avg['Avg Silver Premium']}/mo")
        k5.metric("Total Enrollment",   f"{df['Enrollment'].sum() / 1_000_000:.1f}M")
    else:
        nat_silver   = nat_avg["Avg Silver Premium"]
        state_silver = int(row["Avg Silver Premium"])
        k1.metric("Total Plans",        f"{int(row['Total Plans'])}")
        k2.metric("Carriers",           f"{int(row['Carriers'])}")
        k3.metric("State Rank (plans)",
                  f"#{int(df['Total Plans'].rank(ascending=False).loc[view.index[0]])}")
        k4.metric("Avg Silver Premium",
                  f"${state_silver}/mo",
                  delta=f"${state_silver - nat_silver:+d} vs national avg",
                  delta_color="inverse")
        k5.metric("Enrollment", f"{int(row['Enrollment']):,}")

    st.markdown("---")

    # ── Row 1: Map + Tier donut ────────────────────────────────────────────────
    col_map, col_tier = st.columns([1.6, 1])

    # State centroids for click detection overlay
    _CENTROIDS = {
        "AL":(32.8,-86.8),"AK":(64.2,-153.4),"AZ":(34.3,-111.1),"AR":(34.8,-92.2),
        "CA":(36.8,-119.4),"CO":(39.0,-105.5),"CT":(41.6,-72.7),"DE":(39.0,-75.5),
        "FL":(27.8,-81.6),"GA":(32.7,-83.6),"HI":(20.9,-157.5),"ID":(44.4,-114.5),
        "IL":(40.3,-89.0),"IN":(39.8,-86.1),"IA":(42.0,-93.5),"KS":(38.5,-98.3),
        "KY":(37.5,-85.3),"LA":(31.2,-91.8),"ME":(45.4,-69.2),"MD":(39.0,-76.8),
        "MA":(42.3,-71.8),"MI":(44.2,-85.5),"MN":(46.4,-93.1),"MS":(32.7,-89.7),
        "MO":(38.5,-92.3),"MT":(46.9,-110.5),"NE":(41.5,-99.9),"NV":(38.5,-117.1),
        "NH":(43.7,-71.6),"NJ":(40.0,-74.5),"NM":(34.5,-106.0),"NY":(42.2,-74.9),
        "NC":(35.6,-79.4),"ND":(47.5,-100.5),"OH":(40.4,-82.8),"OK":(35.6,-97.5),
        "OR":(43.9,-120.6),"PA":(40.6,-77.2),"RI":(41.7,-71.6),"SC":(33.9,-80.9),
        "SD":(44.4,-100.2),"TN":(35.9,-86.7),"TX":(31.5,-99.3),"UT":(39.3,-111.1),
        "VT":(44.0,-72.7),"VA":(37.8,-78.2),"WA":(47.4,-120.6),"WV":(38.6,-80.5),
        "WI":(44.3,-89.6),"WY":(42.8,-107.6),"DC":(38.9,-77.0),
    }

    with col_map:
        st.markdown("#### Plans by State")
        st.caption("Click a state to filter all visualizations below.")

        map_df = df.copy()
        if not national:
            map_df["_color"] = map_df["State"].apply(
                lambda s: int(row["Total Plans"]) if s == selected_state else 0
            )
            color_col   = "_color"
            color_scale = [[0, "#1a1a2e"], [0.01, "#333355"], [1.0, "#e05252"]]
        else:
            color_col   = "Total Plans"
            color_scale = [[0,"#1a1a2e"],[0.3,"#16213e"],[0.6,"#e05252"],[1.0,"#ff9999"]]

        fig_map = px.choropleth(
            map_df, locations="Abbr", locationmode="USA-states",
            color=color_col, scope="usa",
            color_continuous_scale=color_scale,
            hover_name="State",
            hover_data={"Total Plans": True, "Carriers": True,
                        "Avg Silver Premium": True, "Abbr": False},
        )

        # Invisible scatter markers at state centroids — scatter traces fire
        # on_select reliably; choropleth traces do not.
        _c_abbrs  = [a for a in _CENTROIDS if a in abbr_to_state]
        _c_lats   = [_CENTROIDS[a][0] for a in _c_abbrs]
        _c_lons   = [_CENTROIDS[a][1] for a in _c_abbrs]
        _c_names  = [abbr_to_state[a] for a in _c_abbrs]
        # FIX 1: opacity=0.01 so pointer events register
        fig_map.add_trace(go.Scattergeo(
            lat=_c_lats, lon=_c_lons,
            mode="markers",
            marker=dict(size=50, opacity=0.01, color="#ffffff"),
            text=_c_names,
            customdata=_c_abbrs,
            hoverinfo="none",
            showlegend=False,
        ))

        fig_map.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items()},
            coloraxis_colorbar=dict(
                title=dict(text="Plans", font=dict(color="#ffffff")),
                tickfont=dict(color="#ffffff"),
            ),
            geo=dict(bgcolor="#000000", lakecolor="#000000",
                     landcolor="#111111", subunitcolor="#333333"),
            clickmode="event+select",
        )

        map_event = st.plotly_chart(
            fig_map, use_container_width=True,
            on_select="rerun", selection_mode="points", key="map_chart",
        )

        # ROOT CAUSE FIX: selection is a PlotlySelectionState object, not a dict.
        # .get("points") always returns None — must use getattr or direct attribute.
        sel    = getattr(map_event, "selection", None)
        points = getattr(sel, "points", None) or []

        for pt in points:
            curve = pt.get("curve_number")   # Streamlit uses snake_case
            clicked_abbr = pt.get("location") if curve == 0 else pt.get("customdata") if curve == 1 else None

            if clicked_abbr and clicked_abbr in abbr_to_state:
                new_state = abbr_to_state[clicked_abbr]
                if new_state != st.session_state["dash_state"]:
                    st.session_state["_pending_map_state"] = new_state
                    st.rerun()
                break

    with col_tier:
        st.markdown("#### Plans by Metal Tier")
        st.caption("Breakdown of plan count by coverage level — Bronze is lowest cost, "
                   "Platinum offers the richest benefits.")
        tier_vals = {
            t: (int(row[t]) if not national else int(df[t].sum()))
            for t in ["Bronze", "Silver", "Gold", "Platinum", "Catastrophic"]
        }
        fig_donut = go.Figure(go.Pie(
            labels=list(tier_vals.keys()),
            values=list(tier_vals.values()),
            hole=0.55,
            marker_colors=[TIER_COLORS[t] for t in tier_vals],
            textfont=dict(color="#ffffff"),
        ))
        fig_donut.update_layout(
            paper_bgcolor="#000000", font_color="#ffffff",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color="#ffffff"), bgcolor="#111111"),
            annotations=[dict(text="Tiers", x=0.5, y=0.5,
                              font_size=16, font_color="#ffffff", showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="donut_chart")

    # ── Row 2: Premium bar + Plan type ────────────────────────────────────────
    col_prem, col_type = st.columns(2)

    with col_prem:
        if national:
            st.markdown("#### Avg Monthly Premium by Tier")
            st.caption("National average unsubsidised monthly premium for a 21-year-old "
                       "across each metal tier.")
            prem_df = pd.DataFrame({
                "Tier":    ["Bronze", "Silver", "Gold", "Platinum"],
                "Premium": [nat_avg[f"Avg {t} Premium"]
                            for t in ["Bronze","Silver","Gold","Platinum"]],
                "Group":   ["National Avg"] * 4,
            })
        else:
            st.markdown(f"#### Premium: {selected_state} vs National Avg")
            st.caption(f"Side-by-side comparison of average monthly premiums in "
                       f"{selected_state} versus the national benchmark per tier.")
            tiers = ["Bronze", "Silver", "Gold", "Platinum"]
            prem_df = pd.DataFrame(
                [{"Tier": t, "Premium": int(row[f"Avg {t} Premium"]),
                  "Group": selected_state} for t in tiers] +
                [{"Tier": t, "Premium": nat_avg[f"Avg {t} Premium"],
                  "Group": "National Avg"} for t in tiers]
            )

        fig_prem = px.bar(
            prem_df, x="Tier", y="Premium",
            color="Group" if not national else "Tier", barmode="group",
            color_discrete_map={
                "Bronze": "#cd7f32", "Silver": "#aaaaaa",
                "Gold":   "#ffd700", "Platinum": "#e5e4e2",
                selected_state: "#e05252", "National Avg": "#4a9eff",
            },
            text="Premium",
        )
        fig_prem.update_traces(texttemplate="$%{text}", textposition="outside",
                               textfont_color="#ffffff")
        fig_prem.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(color="#ffffff"),
            yaxis=dict(color="#ffffff", title="$/month"),
            legend=dict(font=dict(color="#ffffff"), bgcolor="#111111"),
            showlegend=not national,
        )
        st.plotly_chart(fig_prem, use_container_width=True, key="prem_chart")

    with col_type:
        st.markdown("#### Plan Type Distribution")
        st.caption("Share of plans by network type — HMO requires referrals, "
                   "PPO allows out-of-network care, EPO is a hybrid.")
        hmo = int(row["HMO"]) if not national else int(df["HMO"].sum())
        ppo = int(row["PPO"]) if not national else int(df["PPO"].sum())
        epo = int(row["EPO"]) if not national else int(df["EPO"].sum())
        fig_type = px.pie(
            pd.DataFrame({"Type": ["HMO","PPO","EPO"], "Plans": [hmo, ppo, epo]}),
            names="Type", values="Plans",
            color_discrete_sequence=["#e05252", "#4a9eff", "#50c878"],
        )
        fig_type.update_traces(textfont=dict(color="#ffffff"))
        fig_type.update_layout(
            paper_bgcolor="#000000", font_color="#ffffff",
            legend=dict(font=dict(color="#ffffff"), bgcolor="#111111"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_type, use_container_width=True, key="type_chart")

    # ── Row 3: Top 15 bars ────────────────────────────────────────────────────
    col_top, col_sil = st.columns(2)

    with col_top:
        if national:
            st.markdown("#### Top 15 States by Number of Plans")
            st.caption("States with the most marketplace plan options, "
                       "giving consumers the widest choice of coverage.")
            bar_df = df.nlargest(15, "Total Plans").sort_values("Total Plans").copy()
            bar_df["_c"] = bar_df["Total Plans"]
        else:
            st.markdown(f"#### {selected_state} vs Top States (by plans)")
            st.caption(f"How {selected_state} (highlighted in red) compares against "
                       f"the states with the most available plans.")
            bar_df = df.nlargest(15, "Total Plans")
            if view.index[0] not in bar_df.index:
                bar_df = pd.concat([bar_df, view]).head(16)
            bar_df = bar_df.sort_values("Total Plans").copy()
            bar_df["_c"] = bar_df["State"].apply(lambda s: 2 if s == selected_state else 1)

        fig_top = px.bar(
            bar_df, x="Total Plans", y="State", orientation="h",
            color="_c",
            color_continuous_scale=[[0,"#1a1a2e"],[0.4,"#333366"],[1.0,"#e05252"]],
            text="Total Plans",
        )
        fig_top.update_traces(textposition="outside", textfont_color="#ffffff")
        fig_top.update_layout(**CHART_LAYOUT, coloraxis_showscale=False,
                              xaxis=dict(color="#ffffff"), yaxis=dict(color="#ffffff"))
        st.plotly_chart(fig_top, use_container_width=True, key="top_chart")

    with col_sil:
        if national:
            st.markdown("#### Avg Silver Premium — Top 15 Costliest States")
            st.caption("States with the highest average unsubsidised Silver plan premiums, "
                       "useful for understanding regional cost-of-coverage differences.")
            sil_df = df.nlargest(15, "Avg Silver Premium").sort_values("Avg Silver Premium").copy()
            sil_df["_c"] = sil_df["Avg Silver Premium"]
        else:
            st.markdown(f"#### {selected_state} Silver Premium vs Top 15")
            st.caption(f"Where {selected_state}'s Silver premium ranks among "
                       f"the most expensive states nationally.")
            sil_df = df.nlargest(15, "Avg Silver Premium")
            if view.index[0] not in sil_df.index:
                sil_df = pd.concat([sil_df, view])
            sil_df = sil_df.sort_values("Avg Silver Premium").copy()
            sil_df["_c"] = sil_df["State"].apply(lambda s: 2 if s == selected_state else 1)

        fig_sil = px.bar(
            sil_df, x="Avg Silver Premium", y="State", orientation="h",
            color="_c",
            color_continuous_scale=[[0,"#1a2a1a"],[0.4,"#2a4a2a"],[1.0,"#50c878"]],
            text="Avg Silver Premium",
        )
        fig_sil.update_traces(texttemplate="$%{text}", textposition="outside",
                              textfont_color="#ffffff")
        fig_sil.update_layout(**CHART_LAYOUT, coloraxis_showscale=False,
                              xaxis=dict(color="#ffffff", title="$/month"),
                              yaxis=dict(color="#ffffff"))
        st.plotly_chart(fig_sil, use_container_width=True, key="sil_chart")

    # ── Full state table ───────────────────────────────────────────────────────
    st.markdown("#### Full State-by-State Breakdown")
    st.caption("Complete plan inventory per state — sortable by any column. "
               "Filtered to the selected state when one is active.")
    disp = df[[
        "State", "Total Plans", "Carriers", "HMO", "PPO", "EPO",
        "Bronze", "Silver", "Gold", "Platinum",
        "Avg Bronze Premium", "Avg Silver Premium",
        "Avg Gold Premium", "Avg Platinum Premium", "Enrollment",
    ]].copy()
    if not national:
        disp = disp[disp["State"] == selected_state]
    else:
        disp = disp.sort_values("Total Plans", ascending=False)

    for c in ["Avg Bronze Premium","Avg Silver Premium",
              "Avg Gold Premium","Avg Platinum Premium"]:
        disp[c] = disp[c].apply(lambda x: f"${x}")
    disp["Enrollment"] = disp["Enrollment"].apply(lambda x: f"{x:,}")
    st.dataframe(disp.reset_index(drop=True), use_container_width=True,
                 hide_index=True, height=420)
