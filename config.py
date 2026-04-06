"""
Configuration constants for the MA Health Benefits Navigator.

Data sources:
  - CMS Benefits & Cost Sharing PUF 2025 (benefit categories, cost-sharing)
  - MA Health Connector 2025 filings (ConnectorCare plan parameters, Table 4)
  - MA DOI 2025 (licensed carrier list)
  - CMS MA state-specific age curve (2:1 statutory cap)
"""

import os

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_RAW      = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC     = os.path.join(BASE_DIR, "data", "processed")
VECTORSTORE   = os.path.join(BASE_DIR, "data", "vectorstore")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
FEEDBACK_DIR  = os.path.join(BASE_DIR, "feedback")

for _d in [DATA_RAW, DATA_PROC, VECTORSTORE, MODELS_DIR, FEEDBACK_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Mistral ────────────────────────────────────────────────────────────────────
MISTRAL_MODEL = "mistral-small-latest"
TOP_K = 15   # number of FAISS candidates retrieved before re-ranking
TOP_N = 5    # top chunks passed to the LLM after XGBoost re-ranking

# ── MA Licensed Carriers (MA DOI 2025) ────────────────────────────────────────
MA_CARRIERS = [
    "Blue Cross Blue Shield MA",
    "Harvard Pilgrim",
    "Tufts Health",
    "Fallon",
    "Health New England",
    "WellSense",
    "Mass General Brigham",
    "UnitedHealthcare",
]

# ── Benefits extracted from CMS Benefits & Cost Sharing PUF ───────────────────
TARGET_BENEFITS = [
    "Primary Care Visit to Treat an Injury or Illness",
    "Specialist Visit",
    "Emergency Room Services",
    "Urgent Care Centers or Facilities",
    "Mental/Behavioral Health Outpatient Services",
    "Substance Abuse Disorder Outpatient Services",
    "Prenatal and Postnatal Care",
    "Generic Drugs",
    "Preferred Brand Drugs",
    "Non-Preferred Brand Drugs",
    "Specialty Drugs",
    "Inpatient Hospital Services (e.g. Hospital Stay)",
    "Outpatient Surgery Physician/Surgical Services",
    "Rehabilitative Physical Therapy",
    "Rehabilitative Occupational and Rehabilitative Speech Therapy",
    "Skilled Nursing Facility",
    "Preventive Care/Screening/Immunization",
    "Pediatric Dental Check-Up for a Child",
    "Routine Eye Exam (Adult)",
    "Imaging (CT/PET Scans, MRIs)",
    "Laboratory Outpatient and Professional Services",
]

# ── MA Age Multipliers (CMS state-specific curve, 2:1 cap) ────────────────────
MA_AGE_MULTIPLIERS = {
    **{a: 0.635 for a in range(0, 18)},
    **{a: 0.762 for a in range(18, 21)},
    21: 1.000, 22: 1.000, 23: 1.000, 24: 1.000, 25: 1.004, 26: 1.024,
    27: 1.048, 28: 1.087, 29: 1.119, 30: 1.135, 31: 1.159, 32: 1.183,
    33: 1.198, 34: 1.214, 35: 1.230, 36: 1.254, 37: 1.278, 38: 1.302,
    39: 1.317, 40: 1.333, 41: 1.357, 42: 1.397, 43: 1.429, 44: 1.468,
    45: 1.516, 46: 1.579, 47: 1.651, 48: 1.683, 49: 1.714, 50: 1.762,
    51: 1.817, 52: 1.873, 53: 1.937,
    **{a: 2.000 for a in range(54, 65)},
}

# Base monthly premiums by metal tier (MA 2025, age-21 reference rate, before subsidies)
BASE_PREMIUMS = {
    "Catastrophic": 220,
    "Bronze":       320,
    "Silver":       430,
    "Gold":         530,
    "Platinum":     640,
}

# ── ConnectorCare Plan Parameters (MA Health Connector PDF Table 4) ────────────
CONNECTORCARE_DATA = [
    {"plan_type": "Type 1",  "fpl_range": "0–100%",   "premium_min": 0,
     "deductible": 0,   "oop_medical_ind": 0,    "oop_medical_fam": 0,
     "oop_rx_ind": 250, "oop_rx_fam": 500,
     "preventive": 0,  "pcp": 0,  "specialist": 0,   "mental_health_out": 0,
     "speech_ot_pt": 0,  "er": 0,   "urgent_care": 0,  "outpatient_surgery": 0,
     "inpatient": 0,    "imaging": 0, "lab_xray": 0,   "snf": 0,
     "rx_generic": 0,  "rx_pref_brand": 0, "rx_nonpref_brand": 0, "rx_specialty": 0},

    {"plan_type": "Type 2A", "fpl_range": "100–150%", "premium_min": 0,
     "deductible": 0,   "oop_medical_ind": 750,  "oop_medical_fam": 1500,
     "oop_rx_ind": 500, "oop_rx_fam": 1000,
     "preventive": 0,  "pcp": 0,  "specialist": 18,  "mental_health_out": 0,
     "speech_ot_pt": 10, "er": 50,  "urgent_care": 18, "outpatient_surgery": 50,
     "inpatient": 50,   "imaging": 30, "lab_xray": 0,  "snf": 0,
     "rx_generic": 10, "rx_pref_brand": 20, "rx_nonpref_brand": 40, "rx_specialty": 40},

    {"plan_type": "Type 2B", "fpl_range": "150–200%", "premium_min": 51,
     "deductible": 0,   "oop_medical_ind": 750,  "oop_medical_fam": 1500,
     "oop_rx_ind": 500, "oop_rx_fam": 1000,
     "preventive": 0,  "pcp": 0,  "specialist": 18,  "mental_health_out": 0,
     "speech_ot_pt": 10, "er": 50,  "urgent_care": 18, "outpatient_surgery": 50,
     "inpatient": 50,   "imaging": 30, "lab_xray": 0,  "snf": 0,
     "rx_generic": 10, "rx_pref_brand": 20, "rx_nonpref_brand": 40, "rx_specialty": 40},

    {"plan_type": "Type 3A", "fpl_range": "200–250%", "premium_min": 99,
     "deductible": 0,   "oop_medical_ind": 1500, "oop_medical_fam": 3000,
     "oop_rx_ind": 750, "oop_rx_fam": 1500,
     "preventive": 0,  "pcp": 0,  "specialist": 22,  "mental_health_out": 0,
     "speech_ot_pt": 20, "er": 100, "urgent_care": 22, "outpatient_surgery": 125,
     "inpatient": 250,  "imaging": 60, "lab_xray": 0,  "snf": 0,
     "rx_generic": 12.5, "rx_pref_brand": 25, "rx_nonpref_brand": 50, "rx_specialty": 50},

    {"plan_type": "Type 3B", "fpl_range": "250–300%", "premium_min": 147,
     "deductible": 0,   "oop_medical_ind": 1500, "oop_medical_fam": 3000,
     "oop_rx_ind": 750, "oop_rx_fam": 1500,
     "preventive": 0,  "pcp": 0,  "specialist": 22,  "mental_health_out": 0,
     "speech_ot_pt": 20, "er": 100, "urgent_care": 22, "outpatient_surgery": 125,
     "inpatient": 250,  "imaging": 60, "lab_xray": 0,  "snf": 0,
     "rx_generic": 12.5, "rx_pref_brand": 25, "rx_nonpref_brand": 50, "rx_specialty": 50},

    {"plan_type": "Type 3C", "fpl_range": "300–400%", "premium_min": 226,
     "deductible": 0,   "oop_medical_ind": 1500, "oop_medical_fam": 3000,
     "oop_rx_ind": 750, "oop_rx_fam": 1500,
     "preventive": 0,  "pcp": 0,  "specialist": 22,  "mental_health_out": 0,
     "speech_ot_pt": 20, "er": 100, "urgent_care": 22, "outpatient_surgery": 125,
     "inpatient": 250,  "imaging": 60, "lab_xray": 0,  "snf": 0,
     "rx_generic": 12.5, "rx_pref_brand": 25, "rx_nonpref_brand": 50, "rx_specialty": 50},

    {"plan_type": "Type 3D", "fpl_range": "400–500%", "premium_min": 264,
     "deductible": 0,   "oop_medical_ind": 1500, "oop_medical_fam": 3000,
     "oop_rx_ind": 750, "oop_rx_fam": 1500,
     "preventive": 0,  "pcp": 0,  "specialist": 22,  "mental_health_out": 0,
     "speech_ot_pt": 20, "er": 100, "urgent_care": 22, "outpatient_surgery": 125,
     "inpatient": 250,  "imaging": 60, "lab_xray": 0,  "snf": 0,
     "rx_generic": 12.5, "rx_pref_brand": 25, "rx_nonpref_brand": 50, "rx_specialty": 50},
]

# ── Real MA Plans (source: mahealthconnector.org 2025) ────────────────────────
# Columns: carrier, plan_name, plan_type, metal_tier, hsa_eligible, referral_required,
#          deductible_ind, oop_ind, pcp, specialist, er, urgent_care,
#          generic, pref_brand, nonpref_brand, specialty, inpatient, imaging, lab
MA_PLANS = [
    ("Blue Cross Blue Shield MA", "Blue Care Elect PPO Saver",       "PPO", "Bronze",   True,  False, 5000, 8700, 30,  60,  500, 75, 15, 55, 110, 200, "20% coins", 70, 20),
    ("Blue Cross Blue Shield MA", "Blue Care Elect PPO 1000",        "PPO", "Silver",   False, False, 1000, 7900, 25,  50,  300, 50, 15, 50, 100, 175, "250/adm",   60, 20),
    ("Blue Cross Blue Shield MA", "Blue Care Elect PPO 500",         "PPO", "Gold",     False, False, 500,  5500, 20,  40,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Blue Cross Blue Shield MA", "Blue Care Elect PPO 0",           "PPO", "Platinum", False, False, 0,    3500, 15,  30,  150, 30, 5,  35, 70,  100, "100/adm",   35, 10),
    ("Blue Cross Blue Shield MA", "HMO Blue New England Value",      "HMO", "Bronze",   True,  True,  4500, 8700, 0,   50,  500, 75, 15, 55, 110, 200, "20% coins", 70, 20),
    ("Blue Cross Blue Shield MA", "HMO Blue New England",            "HMO", "Silver",   False, True,  750,  7900, 0,   40,  300, 50, 15, 50, 100, 175, "250/adm",   60, 20),
    ("Blue Cross Blue Shield MA", "HMO Blue New England Plus",       "HMO", "Gold",     False, True,  0,    5000, 0,   35,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Harvard Pilgrim",           "Harvard Pilgrim HMO Saver",       "HMO", "Bronze",   True,  True,  4500, 8700, 0,   60,  500, 75, 20, 60, 120, 200, "20% coins", 75, 25),
    ("Harvard Pilgrim",           "Harvard Pilgrim HMO",             "HMO", "Silver",   False, True,  1500, 7900, 0,   45,  300, 50, 15, 55, 110, 175, "300/adm",   65, 20),
    ("Harvard Pilgrim",           "Harvard Pilgrim HMO Gold",        "HMO", "Gold",     False, True,  500,  5500, 0,   35,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Harvard Pilgrim",           "Harvard Pilgrim PPO",             "PPO", "Silver",   False, False, 2000, 7900, 30,  50,  300, 50, 15, 55, 110, 175, "300/adm",   65, 20),
    ("Tufts Health",              "Tufts Health Direct HMO Saver",   "HMO", "Bronze",   True,  True,  5000, 8700, 0,   65,  500, 75, 20, 60, 120, 200, "20% coins", 75, 25),
    ("Tufts Health",              "Tufts Health Direct HMO",         "HMO", "Silver",   False, True,  2000, 7900, 0,   50,  300, 50, 15, 55, 110, 175, "300/adm",   65, 20),
    ("Tufts Health",              "Tufts Health Direct Gold",        "HMO", "Gold",     False, True,  500,  5500, 0,   40,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Tufts Health",              "Tufts Health Premier PPO",        "PPO", "Gold",     False, False, 250,  5000, 25,  40,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Fallon",                    "Fallon Health Direct Care HMO",   "HMO", "Silver",   False, True,  1500, 7900, 0,   45,  300, 50, 15, 55, 110, 175, "300/adm",   65, 20),
    ("Fallon",                    "Fallon Health Direct Gold",       "HMO", "Gold",     False, True,  0,    5500, 0,   35,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Fallon",                    "Fallon Health SelectCare Bronze",  "HMO", "Bronze",   True,  True,  5000, 8700, 0,   65,  500, 75, 20, 60, 120, 200, "20% coins", 75, 25),
    ("Health New England",        "HNE HMO Silver",                  "HMO", "Silver",   False, True,  2000, 7900, 0,   50,  300, 50, 15, 55, 110, 175, "300/adm",   65, 20),
    ("Health New England",        "HNE HMO Gold",                    "HMO", "Gold",     False, True,  500,  5500, 0,   40,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Health New England",        "HNE HMO Bronze",                  "HMO", "Bronze",   True,  True,  5000, 8700, 0,   65,  500, 75, 20, 60, 120, 200, "20% coins", 75, 25),
    ("WellSense",                 "WellSense Health Plan Silver",    "HMO", "Silver",   False, True,  1500, 7900, 0,   45,  300, 50, 15, 55, 110, 175, "300/adm",   65, 20),
    ("WellSense",                 "WellSense Health Plan Bronze",    "HMO", "Bronze",   True,  True,  5000, 8700, 0,   65,  500, 75, 20, 60, 120, 200, "20% coins", 75, 25),
    ("Mass General Brigham",      "MGB Health Plan Silver HMO",      "HMO", "Silver",   False, True,  2000, 7900, 0,   50,  300, 50, 15, 55, 110, 175, "300/adm",   65, 20),
    ("Mass General Brigham",      "MGB Health Plan Gold HMO",        "HMO", "Gold",     False, True,  500,  5500, 0,   40,  250, 40, 10, 45, 90,  150, "200/adm",   50, 15),
    ("Mass General Brigham",      "MGB Health Plan Bronze HMO",      "HMO", "Bronze",   True,  True,  5000, 8700, 0,   65,  500, 75, 20, 60, 120, 200, "20% coins", 75, 25),
    ("UnitedHealthcare",          "UHC Navigate HMO Silver",         "HMO", "Silver",   False, True,  2500, 8050, 0,   55,  350, 55, 15, 55, 110, 175, "300/adm",   65, 20),
    ("UnitedHealthcare",          "UHC Navigate HMO Gold",           "HMO", "Gold",     False, True,  750,  5800, 0,   40,  275, 40, 10, 45, 90,  150, "200/adm",   55, 15),
    ("UnitedHealthcare",          "UHC Choice Plus PPO Silver",      "PPO", "Silver",   False, False, 2000, 8050, 30,  55,  350, 55, 15, 55, 110, 175, "300/adm",   65, 20),
    ("UnitedHealthcare",          "UHC Bronze HSA PPO",              "PPO", "Bronze",   True,  False, 5500, 8700, 0,   0,   0,   0,  0,  0,  0,   0,   "20% coins", 0,  0),
]
