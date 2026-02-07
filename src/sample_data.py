"""
Sample Massachusetts Health Plans
"""

from data_models import HealthPlan, PlanType, IncomeLevel

def create_sample_ma_plans():
    """Create sample MA health insurance plans"""
    
    plans = []
    
    # Plan 1: MassHealth
    plans.append(HealthPlan(
        plan_id="MA-MASSHEALTH-001",
        plan_name="MassHealth Standard",
        carrier="MassHealth",
        plan_type=PlanType.HMO,
        monthly_premium=0.0,
        annual_deductible=0.0,
        out_of_pocket_max=0.0,
        primary_care_copay=0.0,
        specialist_copay=0.0,
        emergency_room_copay=0.0,
        prescription_coverage={'generic': 0, 'brand': 0, 'specialty': 0},
        dental_included=True,
        vision_included=True,
        income_eligibility=[IncomeLevel.LOW],
        age_min=19,
        age_max=64,
        benefits_description="Comprehensive coverage for low-income MA residents. Includes medical, dental, vision at no cost.",
        additional_benefits=[
            "Free transportation to medical appointments",
            "Fitness membership (YMCA, Planet Fitness)",
            "Nutrition counseling and healthy food program",
            "Mental health and substance abuse services",
            "Diabetes prevention program"
        ],
        network_size="large",
        hospital_access=["Mass General", "Beth Israel", "Boston Medical Center"],
        star_rating=4.2,
        member_satisfaction=85.0
    ))
    
    # Plan 2: Blue Cross Blue Shield
    plans.append(HealthPlan(
        plan_id="MA-BCBS-HMO-001",
        plan_name="HMO Blue New England",
        carrier="Blue Cross Blue Shield MA",
        plan_type=PlanType.HMO,
        monthly_premium=425.00,
        annual_deductible=2000.0,
        out_of_pocket_max=6000.0,
        primary_care_copay=25.0,
        specialist_copay=50.0,
        emergency_room_copay=250.0,
        prescription_coverage={'generic': 10, 'brand': 40, 'specialty': 100},
        dental_included=False,
        vision_included=False,
        income_eligibility=[IncomeLevel.MEDIUM, IncomeLevel.HIGH],
        age_min=18,
        age_max=64,
        benefits_description="Affordable HMO with extensive provider network. Requires PCP referral for specialists.",
        additional_benefits=[
            "Blue365 wellness discounts",
            "24/7 nurse line",
            "Weight management program",
            "Gym membership discounts",
            "Acupuncture coverage (12 visits/year)"
        ],
        network_size="large",
        hospital_access=["All major MA hospitals"],
        star_rating=4.5,
        member_satisfaction=88.0
    ))
    
    # Plan 3: Harvard Pilgrim
    plans.append(HealthPlan(
        plan_id="MA-HPHC-PPO-001",
        plan_name="Harvard Pilgrim Independence Plan",
        carrier="Harvard Pilgrim Health Care",
        plan_type=PlanType.PPO,
        monthly_premium=575.00,
        annual_deductible=1500.0,
        out_of_pocket_max=7000.0,
        primary_care_copay=30.0,
        specialist_copay=60.0,
        emergency_room_copay=300.0,
        prescription_coverage={'generic': 15, 'brand': 50, 'specialty': 150},
        dental_included=True,
        vision_included=True,
        income_eligibility=[IncomeLevel.MEDIUM, IncomeLevel.HIGH],
        age_min=18,
        age_max=64,
        benefits_description="Flexible PPO with no referral requirements. Access to Harvard-affiliated providers.",
        additional_benefits=[
            "Fitness Your Way gym network",
            "Fertility treatment coverage",
            "Chiropractic care (20 visits/year)",
            "Mental health app subscription",
            "Diabetes management program",
            "Online therapy sessions"
        ],
        network_size="medium",
        hospital_access=["Mass General Brigham", "Dana-Farber"],
        star_rating=4.7,
        member_satisfaction=92.0
    ))
    
    # Plan 4: Tufts Health
    plans.append(HealthPlan(
        plan_id="MA-TUFTS-HMO-001",
        plan_name="Tufts Health Together",
        carrier="Tufts Health Plan",
        plan_type=PlanType.HMO,
        monthly_premium=385.00,
        annual_deductible=1800.0,
        out_of_pocket_max=5500.0,
        primary_care_copay=20.0,
        specialist_copay=45.0,
        emergency_room_copay=200.0,
        prescription_coverage={'generic': 10, 'brand': 35, 'specialty': 80},
        dental_included=True,
        vision_included=False,
        income_eligibility=[IncomeLevel.MEDIUM, IncomeLevel.HIGH],
        age_min=18,
        age_max=64,
        benefits_description="Value-focused HMO with strong preventive care benefits.",
        additional_benefits=[
            "SilverSneakers fitness for 55+",
            "$100 monthly healthy food card",
            "Free annual flu shot",
            "Telehealth visits included",
            "Smoking cessation program",
            "Chronic care management"
        ],
        network_size="large",
        hospital_access=["Tufts Medical", "Lahey Hospital"],
        star_rating=4.3,
        member_satisfaction=86.0
    ))
    
    # Plan 5: Fallon Health
    plans.append(HealthPlan(
        plan_id="MA-FALLON-EPO-001",
        plan_name="Fallon Community Health",
        carrier="Fallon Health",
        plan_type=PlanType.EPO,
        monthly_premium=445.00,
        annual_deductible=2500.0,
        out_of_pocket_max=6500.0,
        primary_care_copay=25.0,
        specialist_copay=55.0,
        emergency_room_copay=275.0,
        prescription_coverage={'generic': 12, 'brand': 45, 'specialty': 120},
        dental_included=False,
        vision_included=True,
        income_eligibility=[IncomeLevel.MEDIUM, IncomeLevel.HIGH],
        age_min=18,
        age_max=64,
        benefits_description="Regional EPO serving Central/Western MA with community health focus.",
        additional_benefits=[
            "Free YMCA membership",
            "Meal delivery after hospital discharge",
            "Home health aide services",
            "Virtual physical therapy"
        ],
        network_size="medium",
        hospital_access=["UMass Memorial", "Saint Vincent"],
        star_rating=4.1,
        member_satisfaction=83.0
    ))
    
    return plans