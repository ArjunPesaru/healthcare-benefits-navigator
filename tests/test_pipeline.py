"""
Unit tests for rag/pipeline.py.

Covers: query context enrichment, cost-based candidate filtering,
XGBoost re-ranking, and feedback CSV persistence.
"""
import os
import csv
import pytest
from unittest.mock import MagicMock

from rag.pipeline import RAGPipeline, FEEDBACK_COLS


# ── TestEncodeQuery ───────────────────────────────────────────────────────────

class TestEncodeQuery:
    """Tests for RAGPipeline.encode_query — user-profile context enrichment."""

    def test_no_filters_returns_bare_question(self, mock_pipeline):
        result = mock_pipeline.encode_query("What is the deductible?")
        assert result == "What is the deductible?"

    def test_age_filter_prepended(self, mock_pipeline):
        result = mock_pipeline.encode_query("plans", filters={"age": 35})
        assert "age 35" in result

    def test_tier_filter_prepended(self, mock_pipeline):
        result = mock_pipeline.encode_query("plans", filters={"tier": "Silver"})
        assert "Silver" in result

    def test_any_tier_not_prepended(self, mock_pipeline):
        result = mock_pipeline.encode_query("plans", filters={"tier": "Any"})
        assert "Any" not in result

    def test_carrier_filter_prepended(self, mock_pipeline):
        result = mock_pipeline.encode_query("plans", filters={"carrier": "Tufts Health"})
        assert "Tufts Health" in result

    def test_any_carrier_not_prepended(self, mock_pipeline):
        result = mock_pipeline.encode_query("plans", filters={"carrier": "Any"})
        assert "Any" not in result

    def test_connectorcare_flag_prepended(self, mock_pipeline):
        result = mock_pipeline.encode_query("plans", filters={"connectorcare": True})
        assert "ConnectorCare eligible" in result

    def test_max_premium_appended(self, mock_pipeline):
        result = mock_pipeline.encode_query("plans", filters={"max_premium": 400})
        assert "$400" in result

    def test_hsa_keyword_auto_detected(self, mock_pipeline):
        result = mock_pipeline.encode_query("best HSA account plan")
        assert "HSA-eligible" in result

    def test_health_savings_account_keyword(self, mock_pipeline):
        result = mock_pipeline.encode_query("health savings account options")
        assert "HSA-eligible" in result

    def test_high_deductible_hdhp_keyword(self, mock_pipeline):
        result = mock_pipeline.encode_query("high deductible plan benefits")
        assert "HSA-eligible" in result

    def test_no_hsa_keyword_no_tag(self, mock_pipeline):
        result = mock_pipeline.encode_query("Silver plan specialist copay")
        assert "HSA-eligible" not in result

    def test_question_remains_in_output(self, mock_pipeline):
        q = "What is the ER copay?"
        result = mock_pipeline.encode_query(q, filters={"age": 30})
        assert q in result

    def test_multiple_filters_all_present(self, mock_pipeline):
        result = mock_pipeline.encode_query(
            "coverage",
            filters={"age": 40, "tier": "Gold", "carrier": "Fallon", "connectorcare": True},
        )
        assert "age 40" in result
        assert "Gold" in result
        assert "Fallon" in result
        assert "ConnectorCare eligible" in result


# ── TestFilterByCost ──────────────────────────────────────────────────────────

class TestFilterByCost:
    """Tests for RAGPipeline.filter_by_cost — budget-based candidate filtering."""

    def _plan_candidates(self, sample_plan_chunks):
        return [(c, 0.9) for c in sample_plan_chunks if c["chunk_type"] == "plan"]

    def test_removes_plans_over_budget(self, mock_pipeline, sample_plan_chunks):
        # At age 30: Silver=$486, Bronze=$363, Gold=$602
        # max_premium=500 → Gold removed
        candidates = self._plan_candidates(sample_plan_chunks)
        filtered = mock_pipeline.filter_by_cost(candidates, age=30, max_premium=500)
        tiers = [c["metadata"]["metal_tier"] for c, _ in filtered]
        assert "Gold" not in tiers

    def test_keeps_plans_under_budget(self, mock_pipeline, sample_plan_chunks):
        # At age 21: Silver=$430, Bronze=$320, Gold=$530
        # max_premium=450 → Silver and Bronze kept
        candidates = self._plan_candidates(sample_plan_chunks)
        filtered = mock_pipeline.filter_by_cost(candidates, age=21, max_premium=450)
        tiers = [c["metadata"]["metal_tier"] for c, _ in filtered]
        assert "Silver" in tiers
        assert "Bronze" in tiers

    def test_no_age_returns_all_unchanged(self, mock_pipeline, sample_plan_chunks):
        candidates = self._plan_candidates(sample_plan_chunks)
        filtered = mock_pipeline.filter_by_cost(candidates, age=None, max_premium=200)
        assert len(filtered) == len(candidates)

    def test_no_max_premium_returns_all_unchanged(self, mock_pipeline, sample_plan_chunks):
        candidates = self._plan_candidates(sample_plan_chunks)
        filtered = mock_pipeline.filter_by_cost(candidates, age=30, max_premium=None)
        assert len(filtered) == len(candidates)

    def test_fallback_when_all_plans_exceed_budget(self, mock_pipeline, sample_plan_chunks):
        """If every plan exceeds the budget limit, return the full list as fallback."""
        candidates = self._plan_candidates(sample_plan_chunks)
        filtered = mock_pipeline.filter_by_cost(candidates, age=21, max_premium=1)
        assert len(filtered) == len(candidates)

    def test_connectorcare_chunks_always_pass_through(self, mock_pipeline, sample_plan_chunks):
        """Non-plan chunks bypass cost filtering unconditionally."""
        all_candidates = [(c, 0.9) for c in sample_plan_chunks]
        filtered = mock_pipeline.filter_by_cost(all_candidates, age=21, max_premium=1)
        cc_in_result = any(c["chunk_type"] == "connectorcare" for c, _ in filtered)
        assert cc_in_result

    def test_scores_are_preserved(self, mock_pipeline, sample_plan_chunks):
        """(chunk, score) tuples preserve their original score values."""
        candidates = [(c, float(i + 1) * 0.2) for i, c in enumerate(
            [c for c in sample_plan_chunks if c["chunk_type"] == "plan"]
        )]
        filtered = mock_pipeline.filter_by_cost(candidates, age=21, max_premium=500)
        for chunk, score in filtered:
            assert isinstance(score, float)


# ── TestRerank ────────────────────────────────────────────────────────────────

class TestRerank:
    """Tests for RAGPipeline.rerank — XGBoost-based candidate re-ordering."""

    def test_empty_candidates_returns_empty_list(self, mock_pipeline):
        assert mock_pipeline.rerank("query", []) == []

    def test_returns_at_most_top_n_items(self, mock_pipeline, sample_plan_chunks):
        from config import TOP_N
        candidates = [(c, 0.5) for c in sample_plan_chunks]
        result = mock_pipeline.rerank("silver plan", candidates)
        assert len(result) <= TOP_N

    def test_each_result_is_a_chunk_dict(self, mock_pipeline, sample_plan_chunks):
        candidates = [(c, 0.5) for c in sample_plan_chunks]
        result = mock_pipeline.rerank("plan", candidates)
        for item in result:
            assert "chunk_id" in item
            assert "text" in item
            assert "metadata" in item

    def test_single_candidate_returned(self, mock_pipeline, sample_plan_chunks):
        candidates = [(sample_plan_chunks[0], 0.9)]
        mock_pipeline.xgb_model.predict.return_value = [0.9]
        result = mock_pipeline.rerank("query", candidates)
        assert len(result) == 1


# ── TestLogFeedback ───────────────────────────────────────────────────────────

class TestLogFeedback:
    """Tests for RAGPipeline.log_feedback — CSV append-only persistence."""

    def test_creates_file_on_first_call(self, mock_pipeline, tmp_path, monkeypatch):
        import rag.pipeline as pipeline_mod
        log_path = str(tmp_path / "feedback_log.csv")
        monkeypatch.setattr(pipeline_mod, "FEEDBACK_LOG", log_path)

        mock_pipeline.log_feedback("test query", "test answer", 5, [], "good")
        assert os.path.exists(log_path)

    def test_csv_contains_all_required_columns(self, mock_pipeline, tmp_path, monkeypatch):
        import rag.pipeline as pipeline_mod
        log_path = str(tmp_path / "feedback_log.csv")
        monkeypatch.setattr(pipeline_mod, "FEEDBACK_LOG", log_path)

        mock_pipeline.log_feedback("query", "answer", 4, [], "")
        with open(log_path) as f:
            fieldnames = csv.DictReader(f).fieldnames
        assert set(fieldnames) == set(FEEDBACK_COLS)

    def test_appends_multiple_rows(self, mock_pipeline, tmp_path, monkeypatch):
        import rag.pipeline as pipeline_mod
        log_path = str(tmp_path / "feedback_log.csv")
        monkeypatch.setattr(pipeline_mod, "FEEDBACK_LOG", log_path)

        mock_pipeline.log_feedback("q1", "a1", 5, [], "")
        mock_pipeline.log_feedback("q2", "a2", 3, [], "needs work")
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_row_values_are_recorded_correctly(self, mock_pipeline, tmp_path, monkeypatch,
                                               sample_plan_chunks):
        import rag.pipeline as pipeline_mod
        log_path = str(tmp_path / "feedback_log.csv")
        monkeypatch.setattr(pipeline_mod, "FEEDBACK_LOG", log_path)

        mock_pipeline.log_feedback(
            "my query", "my answer", 5, [sample_plan_chunks[0]], "great"
        )
        with open(log_path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["query"] == "my query"
        assert row["rating"] == "5"
        assert row["comment"] == "great"
        assert row["answer"] == "my answer"

    def test_header_written_only_once(self, mock_pipeline, tmp_path, monkeypatch):
        """Calling log_feedback multiple times must not duplicate the CSV header."""
        import rag.pipeline as pipeline_mod
        log_path = str(tmp_path / "feedback_log.csv")
        monkeypatch.setattr(pipeline_mod, "FEEDBACK_LOG", log_path)

        for i in range(3):
            mock_pipeline.log_feedback(f"q{i}", f"a{i}", i + 1, [], "")

        with open(log_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        # 1 header + 3 data rows
        assert len(lines) == 4
