"""
Unit tests for rag/reranker.py.

Covers: feature extraction correctness, edge cases, model training/saving,
and load_reranker error handling.
"""
import os
import numpy as np
import pytest

from rag.reranker import extract_features, FEATURE_NAMES, train_reranker, load_reranker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def silver_chunk():
    return {
        "chunk_type": "plan",
        "text": (
            "silver plan Harvard Pilgrim HMO monthly premium $430 "
            "deductible $1000 PCP $25 specialist $50 hmo plan type"
        ),
        "metadata": {
            "metal_tier":         "silver",
            "carrier":            "Harvard Pilgrim",
            "plan_type":          "hmo",
            "chunk_type":         "plan",
            "avg_feedback_score": 0.8,
        },
    }


@pytest.fixture
def connectorcare_chunk():
    return {
        "chunk_type": "connectorcare",
        "text": "ConnectorCare Type 1 income 0-100% FPL premium $0 deductible $0",
        "metadata": {
            "chunk_type":         "connectorcare",
            "fpl_range":          "0-100%",
            "avg_feedback_score": 0.6,
        },
    }


# ── TestExtractFeatures ───────────────────────────────────────────────────────

class TestExtractFeatures:
    """Feature extraction returns correct values for all 11 features."""

    def test_returns_11_features(self, silver_chunk):
        feats = extract_features("silver plan cost", silver_chunk, 0.85)
        assert len(feats) == 11

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 11

    def test_all_features_are_floats(self, silver_chunk):
        feats = extract_features("silver plan", silver_chunk, 0.9)
        assert all(isinstance(f, float) for f in feats)

    def test_faiss_score_is_first_feature(self, silver_chunk):
        feats = extract_features("any query", silver_chunk, 0.75)
        assert feats[FEATURE_NAMES.index("faiss_score")] == pytest.approx(0.75)

    def test_tier_match_detected(self, silver_chunk):
        feats = extract_features("silver plan options", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("tier_match")] == 1.0

    def test_tier_mismatch(self, silver_chunk):
        feats = extract_features("gold plan benefits", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("tier_match")] == 0.0

    def test_carrier_match_detected(self, silver_chunk):
        feats = extract_features("harvard pilgrim plans", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("carrier_match")] == 1.0

    def test_carrier_mismatch(self, silver_chunk):
        feats = extract_features("tufts health bronze plan", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("carrier_match")] == 0.0

    def test_plan_type_match(self, silver_chunk):
        feats = extract_features("hmo silver plan", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("type_match")] == 1.0

    def test_connectorcare_chunk_flag_set(self, connectorcare_chunk):
        feats = extract_features("any query", connectorcare_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("is_connectorcare")] == 1.0

    def test_connectorcare_chunk_flag_not_set_for_plan(self, silver_chunk):
        feats = extract_features("any query", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("is_connectorcare")] == 0.0

    def test_cc_query_flag_from_subsidy_keyword(self, silver_chunk):
        feats = extract_features("connectorcare subsidy income fpl", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("cc_query")] == 1.0

    def test_cc_query_flag_not_set_for_generic_query(self, silver_chunk):
        feats = extract_features("silver plan premium", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("cc_query")] == 0.0

    def test_premium_query_flag(self, silver_chunk):
        feats = extract_features("monthly premium cost", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("premium_query")] == 1.0

    def test_premium_query_flag_not_set(self, silver_chunk):
        feats = extract_features("specialist referral required", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("premium_query")] == 0.0

    def test_age_in_query_detected(self, silver_chunk):
        feats = extract_features("plan for 35 year old", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("has_age")] == 1.0

    def test_no_age_in_query(self, silver_chunk):
        feats = extract_features("silver specialist copay", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("has_age")] == 0.0

    def test_keyword_overlap_is_normalised(self, silver_chunk):
        feats = extract_features("silver hmo plan premium", silver_chunk, 0.5)
        assert 0.0 <= feats[FEATURE_NAMES.index("keyword_overlap")] <= 1.0

    def test_chunk_length_is_normalised(self, silver_chunk):
        feats = extract_features("query", silver_chunk, 0.5)
        assert 0.0 <= feats[FEATURE_NAMES.index("chunk_length")] <= 1.0

    def test_feedback_score_from_metadata(self, silver_chunk):
        feats = extract_features("query", silver_chunk, 0.5)
        assert feats[FEATURE_NAMES.index("feedback_score")] == pytest.approx(0.8)

    def test_feedback_score_defaults_to_half(self, silver_chunk):
        # Chunk with no avg_feedback_score in metadata
        chunk = {**silver_chunk, "metadata": {**silver_chunk["metadata"]}}
        del chunk["metadata"]["avg_feedback_score"]
        feats = extract_features("query", chunk, 0.5)
        assert feats[FEATURE_NAMES.index("feedback_score")] == pytest.approx(0.5)

    def test_empty_query_does_not_raise(self, silver_chunk):
        feats = extract_features("", silver_chunk, 0.0)
        assert len(feats) == 11

    def test_empty_chunk_text_does_not_raise(self):
        chunk = {"chunk_type": "plan", "text": "", "metadata": {}}
        feats = extract_features("silver plan", chunk, 0.5)
        assert len(feats) == 11


# ── TestLoadReranker ──────────────────────────────────────────────────────────

class TestLoadReranker:
    def test_raises_file_not_found_when_model_missing(self, tmp_path, monkeypatch):
        """load_reranker raises FileNotFoundError pointing user to setup.py."""
        import rag.reranker as reranker_mod
        monkeypatch.setattr(reranker_mod, "MODEL_PATH", str(tmp_path / "no_model.pkl"))
        with pytest.raises(FileNotFoundError, match="setup.py"):
            load_reranker()


# ── TestTrainReranker ─────────────────────────────────────────────────────────

class TestTrainReranker:
    def test_trains_and_saves_model_file(self, tmp_path, monkeypatch, sample_plan_chunks):
        """train_reranker writes a pickle file and returns a usable model."""
        import rag.reranker as reranker_mod
        model_path = str(tmp_path / "test_reranker.pkl")
        monkeypatch.setattr(reranker_mod, "MODEL_PATH", model_path)

        model = train_reranker(sample_plan_chunks)

        assert os.path.exists(model_path)
        X = np.array([extract_features("silver plan", c, 0.5) for c in sample_plan_chunks])
        scores = model.predict(X)
        assert len(scores) == len(sample_plan_chunks)

    def test_saved_model_can_be_loaded(self, tmp_path, monkeypatch, sample_plan_chunks):
        """Model saved by train_reranker can be reloaded and used for inference."""
        import rag.reranker as reranker_mod
        model_path = str(tmp_path / "test_reranker.pkl")
        monkeypatch.setattr(reranker_mod, "MODEL_PATH", model_path)

        train_reranker(sample_plan_chunks)
        loaded = load_reranker()

        X = np.array([extract_features("query", c, 0.5) for c in sample_plan_chunks])
        scores = loaded.predict(X)
        assert len(scores) == len(sample_plan_chunks)
        assert all(isinstance(float(s), float) for s in scores)
