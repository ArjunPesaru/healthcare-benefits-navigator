"""
Unit tests for data_builder.py.

Covers: helper functions, CSV builders, master dataset construction,
chunk building, and data quality validation.
"""
import pandas as pd
import pytest

from data_builder import (
    get_issuer_id,
    get_plan_id,
    format_copay,
    build_plan_attributes,
    build_benefits,
    build_rates,
    build_master,
    build_chunks,
    validate_data,
)
from config import MA_PLANS, CONNECTORCARE_DATA, TARGET_BENEFITS


# ── TestHelperFunctions ───────────────────────────────────────────────────────

class TestHelperFunctions:
    """Tests for pure helper functions used during data construction."""

    def test_get_issuer_id_returns_integer(self):
        carrier = MA_PLANS[0][0]
        assert isinstance(get_issuer_id(carrier), int)

    def test_get_issuer_id_base_value(self):
        carrier = MA_PLANS[0][0]
        assert get_issuer_id(carrier) >= 10000

    def test_get_issuer_id_is_unique_per_carrier(self):
        carriers = list(dict.fromkeys(c for c, *_ in MA_PLANS))
        ids = [get_issuer_id(c) for c in carriers]
        assert len(ids) == len(set(ids)), "Each carrier must have a unique issuer ID"

    def test_get_plan_id_contains_ma(self):
        pid = get_plan_id(MA_PLANS[0][0], 0)
        assert "MA" in pid

    def test_get_plan_id_ends_with_suffix(self):
        pid = get_plan_id(MA_PLANS[0][0], 0)
        assert pid.endswith("0001-00")

    def test_get_plan_id_sequential_index(self):
        carrier = MA_PLANS[0][0]
        pid0 = get_plan_id(carrier, 0)
        pid1 = get_plan_id(carrier, 1)
        assert pid0 != pid1

    def test_format_copay_string_passthrough(self):
        assert format_copay("20% coins") == "20% coins"

    def test_format_copay_zero(self):
        assert format_copay(0) == "$0"

    def test_format_copay_positive_integer(self):
        assert format_copay(25) == "$25"

    def test_format_copay_positive_float(self):
        assert format_copay(12.5) == "$12.5"


# ── TestBuildPlanAttributes ───────────────────────────────────────────────────

class TestBuildPlanAttributes:
    def test_row_count_matches_plan_list(self):
        df = build_plan_attributes()
        assert len(df) == len(MA_PLANS)

    def test_required_columns_present(self):
        df = build_plan_attributes()
        for col in ["PlanId", "PlanType", "MetalLevel", "IsHSAEligible", "StateCode"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_state_code_is_ma(self):
        df = build_plan_attributes()
        assert (df["StateCode"] == "MA").all()

    def test_business_year_is_2025(self):
        df = build_plan_attributes()
        assert (df["BusinessYear"] == "2025").all()

    def test_no_dental_only_plans(self):
        df = build_plan_attributes()
        assert (df["DentalOnlyPlan"] == "No").all()

    def test_plan_ids_are_unique(self):
        df = build_plan_attributes()
        assert df["PlanId"].nunique() == len(df)

    def test_hsa_values_are_yes_or_no(self):
        df = build_plan_attributes()
        assert set(df["IsHSAEligible"].unique()).issubset({"Yes", "No"})


# ── TestBuildBenefits ─────────────────────────────────────────────────────────

class TestBuildBenefits:
    def test_covers_all_target_benefits(self):
        df = build_benefits()
        present = df["BenefitName"].unique()
        for b in TARGET_BENEFITS:
            assert b in present, f"Missing benefit: {b}"

    def test_every_benefit_is_covered(self):
        df = build_benefits()
        assert (df["IsCovered"] == "Covered").all()

    def test_plan_ids_match_attributes(self):
        df_attr = build_plan_attributes()
        df_bcs  = build_benefits()
        attr_ids = set(df_attr["PlanId"])
        bcs_ids  = set(df_bcs["PlanId"])
        assert bcs_ids == attr_ids

    def test_rows_produced_for_every_plan(self):
        df = build_benefits()
        assert df["PlanId"].nunique() == len(MA_PLANS)


# ── TestBuildRates ────────────────────────────────────────────────────────────

class TestBuildRates:
    def test_age_range_spans_18_to_64(self):
        df = build_rates()
        ages = df["Age"].astype(int).unique()
        assert 18 in ages
        assert 64 in ages

    def test_all_premiums_are_positive(self):
        df = build_rates()
        df["IndividualRate"] = pd.to_numeric(df["IndividualRate"])
        assert (df["IndividualRate"] > 0).all()

    def test_platinum_more_expensive_than_bronze_on_average(self):
        df_rate = build_rates()
        df_attr = build_plan_attributes()
        merged = df_rate.merge(df_attr[["PlanId", "MetalLevel"]], on="PlanId")
        merged["IndividualRate"] = pd.to_numeric(merged["IndividualRate"])
        avg_bronze   = merged[merged["MetalLevel"] == "Bronze"]["IndividualRate"].mean()
        avg_platinum = merged[merged["MetalLevel"] == "Platinum"]["IndividualRate"].mean()
        assert avg_platinum > avg_bronze

    def test_gold_more_expensive_than_silver_on_average(self):
        df_rate = build_rates()
        df_attr = build_plan_attributes()
        merged = df_rate.merge(df_attr[["PlanId", "MetalLevel"]], on="PlanId")
        merged["IndividualRate"] = pd.to_numeric(merged["IndividualRate"])
        avg_silver = merged[merged["MetalLevel"] == "Silver"]["IndividualRate"].mean()
        avg_gold   = merged[merged["MetalLevel"] == "Gold"]["IndividualRate"].mean()
        assert avg_gold > avg_silver


# ── TestBuildChunks ───────────────────────────────────────────────────────────

class TestBuildChunks:
    def test_plan_chunk_count_matches_master_dataframe(self, built_plan_data):
        df_master, df_cc = built_plan_data
        chunks      = build_chunks(df_master, df_cc)
        plan_chunks = [c for c in chunks if c["chunk_type"] == "plan"]
        assert len(plan_chunks) == len(df_master)

    def test_connectorcare_chunk_count_matches_config(self, built_plan_data):
        df_master, df_cc = built_plan_data
        chunks   = build_chunks(df_master, df_cc)
        cc_chunks = [c for c in chunks if c["chunk_type"] == "connectorcare"]
        assert len(cc_chunks) == len(CONNECTORCARE_DATA)

    def test_all_chunks_have_required_keys(self, built_plan_data):
        df_master, df_cc = built_plan_data
        chunks = build_chunks(df_master, df_cc)
        for chunk in chunks:
            for key in ("chunk_id", "chunk_type", "text", "metadata"):
                assert key in chunk, f"Chunk missing key: {key}"

    def test_plan_chunks_include_age_premiums(self, built_plan_data):
        df_master, df_cc = built_plan_data
        chunks      = build_chunks(df_master, df_cc)
        plan_chunks = [c for c in chunks if c["chunk_type"] == "plan"]
        for chunk in plan_chunks:
            assert "age_premiums" in chunk["metadata"]
            assert isinstance(chunk["metadata"]["age_premiums"], dict)

    def test_no_chunk_has_empty_text(self, built_plan_data):
        df_master, df_cc = built_plan_data
        chunks = build_chunks(df_master, df_cc)
        for chunk in chunks:
            assert chunk["text"].strip() != "", f"Empty text in chunk: {chunk['chunk_id']}"

    def test_chunk_ids_are_unique(self, built_plan_data):
        df_master, df_cc = built_plan_data
        chunks = build_chunks(df_master, df_cc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


# ── TestValidateData ──────────────────────────────────────────────────────────

class TestValidateData:
    """validate_data should print a quality report without raising exceptions."""

    def test_valid_master_data_passes_without_exception(self, built_plan_data):
        df_master, _ = built_plan_data
        validate_data(df_master)  # must not raise

    def test_dataframe_with_missing_columns_does_not_raise(self):
        df_bad = pd.DataFrame({"metal_tier": ["Silver"], "plan_year": [2025]})
        validate_data(df_bad)  # graceful degradation — warnings only

    def test_dataframe_with_nulls_does_not_raise(self, built_plan_data):
        df_master, _ = built_plan_data
        df_bad = df_master.copy()
        df_bad.loc[df_bad.index[0], "plan_name"] = None
        validate_data(df_bad)  # must not raise

    def test_accepts_empty_dataframe(self):
        validate_data(pd.DataFrame())  # must not raise
