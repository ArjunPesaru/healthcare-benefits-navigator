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
