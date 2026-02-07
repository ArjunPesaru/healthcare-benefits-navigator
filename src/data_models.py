"""
Data Models for Healthcare Benefits Navigator
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class PlanType(Enum):
    HMO = "HMO"
    PPO = "PPO"
    EPO = "EPO"
    POS = "POS"
    HDHP = "High Deductible Health Plan"

class IncomeLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class UserProfile:
    age: int
    income: float
    household_size: int
    zip_code: str
    has_chronic_conditions: bool
    current_plan_id: Optional[str] = None
    
    def get_income_level(self) -> IncomeLevel:
        fpl_2024 = {1: 15060, 2: 20440, 3: 25820, 4: 31200}
        base_fpl = fpl_2024.get(min(self.household_size, 4), 31200)
        additional_persons = max(0, self.household_size - 4)
        fpl = base_fpl + (additional_persons * 5180)
        fpl_percentage = (self.income / fpl) * 100
        
        if fpl_percentage < 200:
            return IncomeLevel.LOW
        elif fpl_percentage < 400:
            return IncomeLevel.MEDIUM
        else:
            return IncomeLevel.HIGH

@dataclass
class HealthPlan:
    plan_id: str
    plan_name: str
    carrier: str
    plan_type: PlanType
    monthly_premium: float
    annual_deductible: float
    out_of_pocket_max: float
    primary_care_copay: float
    specialist_copay: float
    emergency_room_copay: float
    prescription_coverage: Dict[str, float]
    dental_included: bool
    vision_included: bool
    income_eligibility: List[IncomeLevel]
    age_min: int
    age_max: int
    benefits_description: str
    additional_benefits: List[str]
    network_size: str
    hospital_access: List[str]
    star_rating: float
    member_satisfaction: float
    
    def is_eligible(self, user: UserProfile) -> bool:
        income_match = user.get_income_level() in self.income_eligibility
        age_match = self.age_min <= user.age <= self.age_max
        return income_match and age_match