"""
Builds the MA Health Plans dataset (CSV files + processed chunks).
Run once before setup.py to generate the raw data files.
"""

import os
import json
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from config import (
    DATA_RAW, DATA_PROC, MA_PLANS, MA_AGE_MULTIPLIERS,
    BASE_PREMIUMS, CONNECTORCARE_DATA, TARGET_BENEFITS,
)


# ── Helper Functions ───────────────────────────────────────────────────────────

def get_issuer_id(carrier):
    unique_carriers = list(dict.fromkeys(c for c, *_ in MA_PLANS))
    return 10000 + unique_carriers.index(carrier) * 100


def get_plan_id(carrier, idx):
    return f"{get_issuer_id(carrier)}MA{idx+1:04d}0001-00"


def format_copay(val):
    if isinstance(val, str):
        return val
    return f"${val}" if val > 0 else "$0"


# ── Build Plan Attributes CSV ──────────────────────────────────────────────────

def build_plan_attributes():
    rows = []
    for i, p in enumerate(MA_PLANS):
        carrier, plan_name, plan_type, metal, hsa, referral = p[:6]
        rows.append({
            "StateCode":                       "MA",
            "IssuerId":                        str(get_issuer_id(carrier)),
            "IssuerMarketPlaceMarketingName":  carrier,
            "PlanMarketingName":               plan_name,
            "PlanType":                        plan_type,
            "MetalLevel":                      metal,
            "IsHSAEligible":                   "Yes" if hsa else "No",
            "IsReferralRequiredForSpecialist": "Yes" if referral else "No",
            "NationalNetwork":                 "No",
            "WellnessProgramOffered":          "Yes",
            "DentalOnlyPlan":                  "No",
            "NetworkId":                       f"MAN{get_issuer_id(carrier)}",
            "FormularyId":                     f"MAF{get_issuer_id(carrier)}",
            "PlanId":                          get_plan_id(carrier, i),
            "BusinessYear":                    "2025",
        })
    return pd.DataFrame(rows)


# ── Build Benefits & Cost Sharing CSV ─────────────────────────────────────────

def build_benefits():
    benefit_rows = []
    for i, p in enumerate(MA_PLANS):
        carrier, plan_name, plan_type, metal, hsa, referral, \
        ded, oop, pcp, spec, er, urgent, generic, pref, nonpref, specialty, inp, img, lab = p
        plan_id = get_plan_id(carrier, i)

        benefits_map = {
            "Primary Care Visit to Treat an Injury or Illness":             (format_copay(pcp),       "No Charge"),
            "Specialist Visit":                                              (format_copay(spec),      "No Charge"),
            "Emergency Room Services":                                       (format_copay(er),        "No Charge"),
            "Urgent Care Centers or Facilities":                             (format_copay(urgent),    "No Charge"),
            "Mental/Behavioral Health Outpatient Services":                  (format_copay(pcp),       "No Charge"),
            "Substance Abuse Disorder Outpatient Services":                  (format_copay(pcp),       "No Charge"),
            "Prenatal and Postnatal Care":                                   ("$0",                    "No Charge"),
            "Generic Drugs":                                                 (format_copay(generic),   "No Charge"),
            "Preferred Brand Drugs":                                         (format_copay(pref),      "No Charge"),
            "Non-Preferred Brand Drugs":                                     (format_copay(nonpref),   "No Charge"),
            "Specialty Drugs":                                               (format_copay(specialty), "No Charge"),
            "Inpatient Hospital Services (e.g. Hospital Stay)":             (inp,                     "No Charge"),
            "Outpatient Surgery Physician/Surgical Services":                (format_copay(spec),      "No Charge"),
            "Rehabilitative Physical Therapy":                               (format_copay(spec),      "No Charge"),
            "Rehabilitative Occupational and Rehabilitative Speech Therapy": (format_copay(spec),      "No Charge"),
            "Skilled Nursing Facility":                                      (inp if isinstance(inp, str) else f"${inp}", "No Charge"),
            "Preventive Care/Screening/Immunization":                        ("$0",                    "No Charge"),
            "Pediatric Dental Check-Up for a Child":                        ("$0",                    "No Charge"),
            "Routine Eye Exam (Adult)":                                      (format_copay(spec),      "No Charge"),
            "Imaging (CT/PET Scans, MRIs)":                                 (format_copay(img),       "No Charge"),
            "Laboratory Outpatient and Professional Services":               (format_copay(lab),       "No Charge"),
        }

        for benefit, (copay_val, coins_val) in benefits_map.items():
            benefit_rows.append({
                "StateCode":     "MA",
                "IssuerId":      str(get_issuer_id(carrier)),
                "PlanId":        plan_id,
                "BenefitName":   benefit,
                "CopayInnTier1": copay_val,
                "CoinsInnTier1": coins_val,
                "IsCovered":     "Covered",
                "BusinessYear":  "2025",
            })
    return pd.DataFrame(benefit_rows)


# ── Build Rate PUF (age-based premiums) ───────────────────────────────────────

def build_rates():
    ages = list(range(18, 65))
    rate_rows = []
    for i, p in enumerate(MA_PLANS):
        carrier, plan_name, plan_type, metal = p[:4]
        plan_id = get_plan_id(carrier, i)
        base    = BASE_PREMIUMS.get(metal, 400)
        for age in ages:
            multiplier = MA_AGE_MULTIPLIERS.get(age, 1.0)
            rate_rows.append({
                "StateCode":      "MA",
                "IssuerId":       str(get_issuer_id(carrier)),
                "PlanId":         plan_id,
                "Age":            str(age),
                "IndividualRate": round(base * multiplier, 2),
                "Tobacco":        "Tobacco User/Non-Tobacco User",
                "RatingAreaId":   "Rating Area 1",
                "BusinessYear":   "2025",
            })
    return pd.DataFrame(rate_rows)


# ── Build Master Dataset ───────────────────────────────────────────────────────

def build_master(df_attr, df_bcs, df_rate):
    df_attr = df_attr.copy()
    df_attr = df_attr[df_attr["DentalOnlyPlan"].str.upper() != "YES"]
    df_attr = df_attr.rename(columns={
        "IssuerMarketPlaceMarketingName":  "carrier_name",
        "PlanMarketingName":               "plan_name",
        "PlanType":                        "plan_type",
        "MetalLevel":                      "metal_tier",
        "IsHSAEligible":                   "hsa_eligible",
        "IsReferralRequiredForSpecialist": "referral_required",
        "NationalNetwork":                 "national_network",
        "WellnessProgramOffered":          "wellness_program",
    })
    for col in ["hsa_eligible", "referral_required", "national_network"]:
        if col in df_attr.columns:
            df_attr[col] = df_attr[col].str.upper().map(
                {"YES": True, "NO": False, "Y": True, "N": False}
            )
    df_attr["metal_tier"] = df_attr["metal_tier"].str.title()

    # Pivot benefits to wide format
    df_bcs["benefit_col"] = (
        df_bcs["BenefitName"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    copay   = df_bcs.pivot_table(index="PlanId", columns="benefit_col", values="CopayInnTier1", aggfunc="first").add_prefix("copay_")
    coins   = df_bcs.pivot_table(index="PlanId", columns="benefit_col", values="CoinsInnTier1", aggfunc="first").add_prefix("coins_")
    covered = df_bcs.pivot_table(index="PlanId", columns="benefit_col", values="IsCovered",     aggfunc="first").add_prefix("covered_")
    df_benefits = copay.join(coins, how="outer").join(covered, how="outer").reset_index()

    # Pivot rates to wide format
    df_rate["IndividualRate"] = pd.to_numeric(df_rate["IndividualRate"], errors="coerce")
    df_rate["Age_num"] = pd.to_numeric(df_rate["Age"], errors="coerce")
    target_ages = [21, 27, 30, 35, 40, 45, 50, 55, 60, 64]
    df_rates = (
        df_rate[df_rate["Age_num"].isin(target_ages)]
        .groupby(["PlanId", "Age_num"])["IndividualRate"]
        .mean()
        .unstack("Age_num")
        .reset_index()
    )
    df_rates.columns = ["PlanId"] + [f"premium_age{int(a)}" for a in df_rates.columns[1:]]

    df_master = (
        df_attr
        .merge(df_benefits, on="PlanId", how="left")
        .merge(df_rates,    on="PlanId", how="left")
    )

    df_master["state"]                    = "MA"
    df_master["plan_year"]                = 2025
    df_master["gender_affects_premium"]   = False
    df_master["glp1_obesity_covered"]     = False
    df_master["pediatric_dental"]         = True
    df_master["maternity_covered"]        = True
    df_master["preventive_care_cost"]     = "$0"
    df_master["insulin_generic"]          = "$0"
    df_master["insulin_brand_max"]        = "$25"
    df_master["connectorcare_compatible"] = (
        df_master["metal_tier"].str.lower() == "silver"
    )
    df_master["data_source"] = "CMS State-Based Exchange PUF 2025"
    return df_master


# ── Build RAG Text Chunks ──────────────────────────────────────────────────────

def plan_to_text(row):
    def v(col, default="N/A"):
        val = row.get(col, default)
        return default if pd.isna(val) else str(val)

    def dollar(col, default="See plan"):
        val = row.get(col)
        if val is None or pd.isna(val) or str(val).strip() in ("", "nan", "Not Covered"):
            return default
        s = str(val).strip()
        return s if s.startswith("$") else f"${s}"

    age_cols_present = [
        f"premium_age{a}" for a in [21, 27, 30, 35, 40, 45, 50, 55, 60, 64]
        if f"premium_age{a}" in row.index
    ]
    premium_lines = "\n".join([
        f"  Age {a.replace('premium_age', '')}: {dollar(a)}"
        for a in age_cols_present
    ])

    pcp       = dollar("copay_primary_care_visit_to_treat_an_injury_or_illness")
    spec      = dollar("copay_specialist_visit")
    er        = dollar("copay_emergency_room_services")
    urg       = dollar("copay_urgent_care_centers_or_facilities")
    mh        = dollar("copay_mental_behavioral_health_outpatient_services")
    inp       = dollar("copay_inpatient_hospital_services_e_g_hospital_stay")
    img       = dollar("copay_imaging_ct_pet_scans_mris")
    lab       = dollar("copay_laboratory_outpatient_and_professional_services")
    gen       = dollar("copay_generic_drugs")
    pref      = dollar("copay_preferred_brand_drugs")
    nonpref   = dollar("copay_non_preferred_brand_drugs")
    specialty = dollar("copay_specialty_drugs")
    eye       = dollar("copay_routine_eye_exam_adult")
    cc_compat = "Yes" if str(row.get("connectorcare_compatible", "")).upper() == "TRUE" else "No"
    premiums  = premium_lines if premium_lines else "  See mahealthconnector.org"

    return f"""
PLAN: {v("plan_name")}
Carrier: {v("carrier_name")} | Plan ID: {v("PlanId")}
Plan Type: {v("plan_type")} | Metal Tier: {v("metal_tier")}
State: Massachusetts | Plan Year: 2025
HSA Eligible: {v("hsa_eligible")} | Referral Required: {v("referral_required")}
ConnectorCare Compatible: {cc_compat}

MONTHLY PREMIUMS (unsubsidized):
{premiums}
Note: Premiums identical for male and female (ACA §2701)

COST SHARING:
  Primary Care   : {pcp}
  Specialist     : {spec}
  Emergency Room : {er}
  Urgent Care    : {urg}
  Mental Health  : {mh}
  Inpatient Hosp : {inp}
  Imaging        : {img}
  Lab Services   : {lab}
  Preventive Care: $0 (ACA mandate)

PRESCRIPTIONS:
  Generic        : {gen}
  Preferred Brand: {pref}
  Non-Preferred  : {nonpref}
  Specialty      : {specialty}
  GLP-1 Obesity  : Not covered (all MA carriers Jan 2026)
  Insulin Generic: $0 (PACT Act) | Insulin Brand: $25 max (PACT Act)

ADDITIONAL:
  Maternity/Prenatal : Covered (ACA EHB)
  Pediatric Dental   : Included (ACA EHB)
  Routine Eye Exam   : {eye}
""".strip()


def cc_to_text(row):
    return (
        f"CONNECTORCARE PLAN: {row['plan_type']} — Income {row['fpl_range']} FPL\n"
        f"Source: MA Health Connector ConnectorCare Overview 2025, Table 4\n"
        f"All ConnectorCare plans: $0 deductible | Benefits same across all carriers\n"
        f"Eligibility: Income {row['fpl_range']} FPL | MA resident\n\n"
        f"PREMIUMS & OOP:\n"
        f"  Lowest Premium: ${row['premium_min']}/mo | Deductible: $0\n"
        f"  Medical OOP (Ind/Fam): ${row['oop_medical_ind']} / ${row['oop_medical_fam']}\n"
        f"  Rx OOP (Ind/Fam): ${row['oop_rx_ind']} / ${row['oop_rx_fam']}\n\n"
        f"COPAYS:\n"
        f"  PCP: $0 | Preventive: $0 | Mental Health: $0 | Lab/X-Ray: $0\n"
        f"  Specialist: ${row['specialist']} | Urgent Care: ${row['urgent_care']}\n"
        f"  ER: ${row['er']} | Outpatient Surgery: ${row['outpatient_surgery']}\n"
        f"  Inpatient: ${row['inpatient']} | Imaging: ${row['imaging']}\n"
        f"  Rx Generic: ${row['rx_generic']} | Preferred Brand: ${row['rx_pref_brand']}\n"
        f"  Non-Preferred: ${row['rx_nonpref_brand']} | Specialty: ${row['rx_specialty']}\n"
        f"  Chronic Condition Meds (asthma, diabetes, hypertension): $0"
    )


def build_chunks(df_master, df_cc):
    chunks = []
    for idx, row in df_master.iterrows():
        # Store per-age premiums in metadata for cost filtering
        age_premiums = {}
        for a in [21, 27, 30, 35, 40, 45, 50, 55, 60, 64]:
            col = f"premium_age{a}"
            if col in row.index and not pd.isna(row.get(col)):
                age_premiums[a] = float(row[col])

        chunks.append({
            "chunk_id":   f"plan_{row.get('PlanId', idx)}",
            "chunk_type": "plan",
            "text":       plan_to_text(row),
            "metadata": {
                "plan_id":       str(row.get("PlanId", "")),
                "carrier":       str(row.get("carrier_name", "")),
                "plan_name":     str(row.get("plan_name", "")),
                "plan_type":     str(row.get("plan_type", "")),
                "metal_tier":    str(row.get("metal_tier", "")),
                "hsa_eligible":  str(row.get("hsa_eligible", "")),
                "connectorcare": str(row.get("connectorcare_compatible", "")),
                "deductible":    str(row.get("copay_primary_care_visit_to_treat_an_injury_or_illness", "N/A")),
                "age_premiums":  age_premiums,
                "state":         "MA",
                "plan_year":     "2025",
            }
        })

    for _, row in df_cc.iterrows():
        chunks.append({
            "chunk_id":   f"cc_{row['plan_type'].replace(' ', '_')}",
            "chunk_type": "connectorcare",
            "text":       cc_to_text(row),
            "metadata": {
                "plan_type":   row["plan_type"],
                "fpl_range":   row["fpl_range"],
                "premium_min": row["premium_min"],
                "state":       "MA",
                "plan_year":   "2025",
            }
        })
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("Building MA health plan data...")

    df_attr = build_plan_attributes()
    df_bcs  = build_benefits()
    df_rate = build_rates()

    df_attr.to_csv(os.path.join(DATA_RAW, "Plan_Attributes_PUF_MA.csv"),       index=False)
    df_bcs.to_csv( os.path.join(DATA_RAW, "Benefits_Cost_Sharing_PUF_MA.csv"), index=False)
    df_rate.to_csv(os.path.join(DATA_RAW, "Rate_PUF_MA.csv"),                  index=False)
    print(f"  Saved raw CSVs to {DATA_RAW}")

    df_master = build_master(df_attr, df_bcs, df_rate)
    df_cc     = pd.DataFrame(CONNECTORCARE_DATA)

    df_master.to_csv(os.path.join(DATA_PROC, "ma_plans_2025.csv"), index=False)
    df_cc.to_csv(    os.path.join(DATA_PROC, "connectorcare.csv"),  index=False)
    print(f"  Saved processed data: {len(df_master)} plans, {len(df_cc)} ConnectorCare types")

    chunks = build_chunks(df_master, df_cc)
    chunks_path = os.path.join(DATA_PROC, "chunks.jsonl")
    with open(chunks_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    n_plans = sum(1 for c in chunks if c["chunk_type"] == "plan")
    n_cc    = sum(1 for c in chunks if c["chunk_type"] == "connectorcare")
    print(f"  Built {len(chunks)} chunks ({n_plans} plan + {n_cc} ConnectorCare)")
    print("Data build complete.")
    return df_master, df_cc, chunks


if __name__ == "__main__":
    run()
