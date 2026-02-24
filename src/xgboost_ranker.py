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
        affordability = 1 - min(total_annual_cost / max(user.income, 1), 1.0)
        features.append(affordability)

        return np.array(features, dtype=np.float32)


class XGBoostPlanRanker:
    def __init__(self):
        self.model = None
        self.feature_extractor = PlanFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_training_data(self, examples: List[Dict], fit_scaler: bool = False) -> Tuple:
        """
        Convert examples → (X, y, groups).
        fit_scaler=True only for training data; False for test/inference.
        """
        X_list, y_list, groups = [], [], []
        current_user, current_group_size = None, 0

        for ex in examples:
            user, plan = ex['user'], ex['plan']
            features = self.feature_extractor.extract_features(
                user, plan, ex.get('rag_score', 0.0)
            )
            X_list.append(features)
            y_list.append(ex['relevance'])

            if current_user != user:
                if current_group_size > 0:
                    groups.append(current_group_size)
                current_user = user
                current_group_size = 1
            else:
                current_group_size += 1

        if current_group_size > 0:
            groups.append(current_group_size)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # ✅ Fit scaler ONLY on training data to prevent leakage
        if fit_scaler:
            X = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X = self.scaler.transform(X)

        return X, y, np.array(groups)

    def train(self, training_examples: List[Dict]) -> None:
        X_train, y_train, groups_train = self.prepare_training_data(
            training_examples, fit_scaler=True   # ✅ fit only on train
        )

        params = {
            'objective': 'rank:pairwise',
            'eta': 0.1,
            'max_depth': 6,
            'eval_metric': 'ndcg@5',
            'seed': 42,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(groups_train)

        self.model = xgb.train(
            params, dtrain,
            num_boost_round=100,
            verbose_eval=False
        )
        print("✓ XGBoost training complete")

    def evaluate(self, test_examples: List[Dict]) -> Dict:
        """
        Evaluate on held-out test examples.
        Returns NDCG@5 and a per-user breakdown.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_test, y_test, groups_test = self.prepare_training_data(
            test_examples, fit_scaler=False   # ✅ only transform, never fit
        )

        dtest = xgb.DMatrix(X_test, label=y_test)
        dtest.set_group(groups_test)

        # XGBoost built-in NDCG evaluation
        eval_result = self.model.eval(dtest)
        # eval_result is a string like: "[0]\teval-ndcg@5:0.9134"
        ndcg_score = float(eval_result.split(':')[-1].strip())

        # ── Per-user Precision@3 ──────────────────────────────────────────────
        scores = self.model.predict(dtest)
        precision_scores = []
        idx = 0
        for g_size in groups_test:
            g_scores = scores[idx: idx + g_size]
            g_labels = y_test[idx: idx + g_size]
            top3_idx = np.argsort(g_scores)[::-1][:3]
            # "relevant" = label >= median label for this group
            threshold = np.median(g_labels)
            hits = sum(1 for i in top3_idx if g_labels[i] >= threshold)
            precision_scores.append(hits / 3)
            idx += g_size

        metrics = {
            'ndcg@5':        round(ndcg_score, 4),
            'precision@3':   round(float(np.mean(precision_scores)), 4),
            'test_examples': len(test_examples),
            'num_users':     len(groups_test),
        }

        print("\n📊 XGBoost Evaluation on Test Set:")
        print(f"   NDCG@5       : {metrics['ndcg@5']}")
        print(f"   Precision@3  : {metrics['precision@3']}")
        print(f"   Test examples: {metrics['test_examples']} across {metrics['num_users']} users")

        return metrics

    def rank_plans(self, user, plans: List, rag_scores: Dict[str, float] = None) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("Model not trained.")

        rag_scores = rag_scores or {p.plan_id: 0.0 for p in plans}

        X_list, plan_ids = [], []
        for plan in plans:
            features = self.feature_extractor.extract_features(
                user, plan, rag_scores.get(plan.plan_id, 0.0)
            )
            X_list.append(features)
            plan_ids.append(plan.plan_id)

        X = self.scaler.transform(np.array(X_list, dtype=np.float32))
        scores = self.model.predict(xgb.DMatrix(X))

        return sorted(zip(plan_ids, scores), key=lambda x: x[1], reverse=True)