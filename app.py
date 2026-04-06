"""
MA Health Benefits Navigator — Streamlit App
"""

import os
import torch
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY not found. Create a .env file with your key.")
    st.stop()

st.set_page_config(
    page_title="MA Health Navigator",
    page_icon="🏥",
    layout="wide",
)

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
</style>
""", unsafe_allow_html=True)

st.title("🏥 Massachusetts Health Benefits Navigator")
