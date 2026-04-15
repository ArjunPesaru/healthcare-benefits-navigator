"""
Shared pytest fixtures for the MA Health Benefits Navigator test suite.
"""
import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock

# Prevent OMP/threading conflicts before any FAISS or torch import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure project root is importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Minimal in-memory plan chunks (no disk I/O required) ─────────────────────

@pytest.fixture
def sample_plan_chunks():
    """Return a small list of realistic plan + ConnectorCare chunks for unit testing."""
    return [
        {
            "chunk_id": "plan_10000MA00010001-00",
            "chunk_type": "plan",
            "text": (
                "PLAN: Blue Care Elect PPO Saver "
                "Carrier: Blue Cross Blue Shield MA Silver plan "
                "monthly premium $430 deductible $1000 PCP $25 specialist $50"
            ),
            "metadata": {
                "plan_id":      "10000MA00010001-00",
                "carrier":      "Blue Cross Blue Shield MA",
                "plan_name":    "Blue Care Elect PPO Saver",
                "plan_type":    "PPO",
                "metal_tier":   "Silver",
                "hsa_eligible": "False",
                "connectorcare": "True",
                "deductible":   "$25",
                "age_premiums": {21: 430.0, 30: 486.05, 40: 573.19, 50: 757.66, 60: 852.12},
                "state":        "MA",
                "plan_year":    "2025",
            },
        },
        {
            "chunk_id": "plan_10000MA00020001-00",
            "chunk_type": "plan",
            "text": (
                "PLAN: HMO Blue New England Value "
                "Carrier: Blue Cross Blue Shield MA Bronze plan "
                "monthly premium $320 HSA eligible deductible $4500"
            ),
            "metadata": {
                "plan_id":      "10000MA00020001-00",
                "carrier":      "Blue Cross Blue Shield MA",
                "plan_name":    "HMO Blue New England Value",
                "plan_type":    "HMO",
                "metal_tier":   "Bronze",
                "hsa_eligible": "True",
                "connectorcare": "False",
                "deductible":   "$0",
                "age_premiums": {21: 320.0, 30: 363.2, 40: 426.66, 50: 563.84, 60: 635.0},
                "state":        "MA",
                "plan_year":    "2025",
            },
        },
        {
            "chunk_id": "plan_10100MA00010001-00",
            "chunk_type": "plan",
            "text": (
                "PLAN: Harvard Pilgrim HMO Gold "
                "Carrier: Harvard Pilgrim Gold plan HMO "
                "monthly premium $530 deductible $500 PCP $0 specialist $35"
            ),
            "metadata": {
                "plan_id":      "10100MA00010001-00",
                "carrier":      "Harvard Pilgrim",
                "plan_name":    "Harvard Pilgrim HMO Gold",
                "plan_type":    "HMO",
                "metal_tier":   "Gold",
                "hsa_eligible": "False",
                "connectorcare": "False",
                "deductible":   "$0",
                "age_premiums": {21: 530.0, 30: 601.55, 40: 707.09, 50: 933.86, 60: 1050.76},
                "state":        "MA",
                "plan_year":    "2025",
            },
        },
        {
            "chunk_id": "cc_Type_1",
            "chunk_type": "connectorcare",
            "text": "CONNECTORCARE PLAN: Type 1 Income 0-100% FPL premium $0 deductible $0",
            "metadata": {
                "plan_type":   "Type 1",
                "fpl_range":   "0-100%",
                "premium_min": 0,
                "state":       "MA",
                "plan_year":   "2025",
            },
        },
    ]


@pytest.fixture
def mock_pipeline(sample_plan_chunks):
    """
    Return a RAGPipeline instance with all heavy dependencies replaced by mocks.

    FAISS index, sentence-transformer, XGBoost, and Mistral client are all
    MagicMock objects so tests run offline without models on disk.
    """
    from rag.pipeline import RAGPipeline

    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.92, 0.85, 0.78, 0.65]]),
        np.array([[0, 1, 2, 3]]),
    )

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([[0.1] * 384])

    mock_xgb = MagicMock()
    mock_xgb.predict.return_value = np.array([0.9, 0.7, 0.5, 0.3])

    return RAGPipeline(
        index=mock_index,
        chunks=sample_plan_chunks,
        embedder=mock_embedder,
        xgb_model=mock_xgb,
        mistral_api_key="test-key-not-used",
    )


@pytest.fixture(scope="session")
def built_plan_data():
    """
    Build the master plan DataFrame and ConnectorCare DataFrame once per session.

    Calls the full data_builder pipeline (build_plan_attributes → build_benefits
    → build_rates → build_master) without writing any files to disk.
    """
    import pandas as pd
    from data_builder import (
        build_plan_attributes, build_benefits, build_rates, build_master,
    )
    from config import CONNECTORCARE_DATA

    df_attr   = build_plan_attributes()
    df_bcs    = build_benefits()
    df_rate   = build_rates()
    df_master = build_master(df_attr, df_bcs, df_rate)
    df_cc     = pd.DataFrame(CONNECTORCARE_DATA)
    return df_master, df_cc
