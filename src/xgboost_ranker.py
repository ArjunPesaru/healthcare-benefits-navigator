"""
XGBoost Ranking System
"""

import numpy as np
from typing import List, Dict, Tuple
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class PlanFeatureExtractor:
    def extract_features(self, user, plan, rag_score: float = 0.0) -> np.ndarray:
        features = []
        
        # Cost features
        income_ratio = plan.monthly_premium * 12 / max(user.income, 1)
        features.extend([
            plan.monthly_premium,
            plan.annual_deductible,
            plan.out_of_pocket_max,
            income_ratio,
            plan.primary_care_copay,
            plan.specialist_copay,
            plan.emergency_room_copay
        ])
        
        # Coverage features
        features.extend([
            1 if plan.dental_included else 0,
            1 if plan.vision_included else 0,
            len(plan.additional_benefits),
            plan.prescription_coverage.get('generic', 0),
            plan.prescription_coverage.get('brand', 0),
            plan.prescription_coverage.get('specialty', 0)
        ])
        
        # Quality features
        features.extend([
            plan.star_rating,
            plan.member_satisfaction,
            {'small': 1, 'medium': 2, 'large': 3}.get(plan.network_size, 2),
            len(plan.hospital_access)
        ])
        
        # User-plan match
        features.extend([
            1 if user.has_chronic_conditions else 0,
            abs(user.age - 45),
            1 if user.get_income_level() in plan.income_eligibility else 0
        ])
        
        # Plan type encoding
        plan_type_encoding = {
            'HMO': [1, 0, 0, 0, 0],
            'PPO': [0, 1, 0, 0, 0],
            'EPO': [0, 0, 1, 0, 0],
            'POS': [0, 0, 0, 1, 0],
            'High Deductible Health Plan': [0, 0, 0, 0, 1]
        }
        features.extend(plan_type_encoding.get(plan.plan_type.value, [0, 0, 0, 0, 0]))
        
        features.append(rag_score)
        
        total_annual_cost = plan.monthly_premium * 12 + plan.annual_deductible * 0.5
        affordability = 1 - min(total_annual_cost / user.income, 1.0)
        features.append(affordability)
        
        return np.array(features)

class XGBoostPlanRanker:
    def __init__(self):
        self.model = None
        self.feature_extractor = PlanFeatureExtractor()
        self.scaler = StandardScaler()
    
    def prepare_training_data(self, training_examples: List[Dict]) -> Tuple:
        X_list = []
        y_list = []
        groups = []
        
        current_group = []
        current_user = None
        
        for example in training_examples:
            user = example['user']
            plan = example['plan']
            rag_score = example.get('rag_score', 0.0)
            relevance = example['relevance']
            
            features = self.feature_extractor.extract_features(user, plan, rag_score)
            X_list.append(features)
            y_list.append(relevance)
            
            if current_user != user:
                if current_group:
                    groups.append(len(current_group))
                current_group = [example]
                current_user = user
            else:
                current_group.append(example)
        
        if current_group:
            groups.append(len(current_group))
        
        X = np.array(X_list)
        y = np.array(y_list)
        groups = np.array(groups)
        
        X = self.scaler.fit_transform(X)
        
        return X, y, groups
    
    def train(self, training_examples: List[Dict]) -> None:
        X_train, y_train, groups_train = self.prepare_training_data(training_examples)
        
        params = {
            'objective': 'rank:pairwise',
            'eta': 0.1,
            'max_depth': 6,
            'eval_metric': 'ndcg@10'
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(groups_train)
        
        self.model = xgb.train(params, dtrain, num_boost_round=100)
        print("XGBoost training completed")
    
    def rank_plans(self, user, plans: List, rag_scores: Dict[str, float] = None) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model not trained")
        
        if rag_scores is None:
            rag_scores = {plan.plan_id: 0.0 for plan in plans}
        
        X_list = []
        plan_ids = []
        
        for plan in plans:
            rag_score = rag_scores.get(plan.plan_id, 0.0)
            features = self.feature_extractor.extract_features(user, plan, rag_score)
            X_list.append(features)
            plan_ids.append(plan.plan_id)
        
        X = np.array(X_list)
        X = self.scaler.transform(X)
        
        dmatrix = xgb.DMatrix(X)
        scores = self.model.predict(dmatrix)
        
        ranked = sorted(zip(plan_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked