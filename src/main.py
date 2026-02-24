"""
Healthcare Benefits Navigator - Main System
"""

import os
from dotenv import load_dotenv
load_dotenv()

from data_models import UserProfile
from sample_data import create_sample_ma_plans
from rag_system import BenefitRAGSystem
from xgboost_ranker import XGBoostPlanRanker
from plan_comparator import PlanComparator
from mistral_integration import MistralLLMIntegration, EnhancedRAGSystemWithMistral
from training_utils import generate_training_data, split_training_data  # ✅ new


class HealthcareNavigator:
    def __init__(self):
        print("🏥 Initializing Healthcare Benefits Navigator...")

        # Load plans
        self.plans = create_sample_ma_plans()
        print(f"✓ Loaded {len(self.plans)} health plans")

        # Initialize RAG first — needed to compute real RAG scores for training
        self.rag_system = BenefitRAGSystem()
        documents = self.rag_system.prepare_documents(self.plans)
        self.rag_system.build_index(documents)
        print(f"✓ RAG system ready with {len(documents)} documents")

        # ✅ Generate training data with REAL RAG scores
        all_examples = generate_training_data(self.plans, self.rag_system)

        # ✅ Train/test split
        train_examples, test_examples = split_training_data(all_examples, test_size=0.2)

        # Train XGBoost
        self.xgb_ranker = XGBoostPlanRanker()
        self.xgb_ranker.train(train_examples)
        print(f"✓ XGBoost trained on {len(train_examples)} examples")

        # ✅ Evaluate on held-out test set
        self.eval_metrics = self.xgb_ranker.evaluate(test_examples)

        # Initialize comparator
        self.comparator = PlanComparator(self.rag_system, self.xgb_ranker)

        # Initialize Mistral
        self.mistral_rag = EnhancedRAGSystemWithMistral(self.rag_system)
        print("✓ Mistral AI ready")

        print("\n✅ System fully initialized!\n")

    def ask_question(self, question: str, user_age: int = 35, user_income: float = 60000):
        print(f"\n💬 Question: {question}\n")
        user_context = {
            'age': user_age,
            'income': user_income,
            'household_size': 2,
            'has_chronic_conditions': False
        }
        result = self.mistral_rag.answer_question(question, user_context)
        print(f"🤖 Answer:\n{result['answer']}\n")
        print(f"📚 Sources: {result['num_sources']}")
        return result

    def compare_plans(self, age, income, household_size, has_chronic, current_plan_id):
        user = UserProfile(age, income, household_size, "02108", has_chronic, current_plan_id)
        current_plan = next((p for p in self.plans if p.plan_id == current_plan_id), None)
        if not current_plan:
            print(f"❌ Plan {current_plan_id} not found")
            return None

        print(f"\n📊 Comparing plans for: Age={age}, Income=${income:,}, HH={household_size}")
        print(f"   Current: {current_plan.plan_name}\n")

        results = self.comparator.compare_plans(
            user=user,
            current_plan=current_plan,
            all_plans=self.plans,
            user_query="comprehensive coverage with wellness benefits"
        )

        user_dict = {'age': age, 'income': income,
                     'household_size': household_size, 'has_chronic_conditions': has_chronic}
        explanation = self.mistral_rag.mistral.generate_plan_comparison_summary(results, user_dict)

        print("=" * 80)
        print("🤖 AI ANALYSIS")
        print("=" * 80)
        print(explanation)
        print("=" * 80)

        print(f"\n📈 Key Metrics:")
        print(f"   Eligible Plans : {results['eligible_plans_count']}")
        print(f"   Current Cost   : ${results['cost_analysis']['current_annual_cost']:,.2f}/year")
        print(f"\n📊 Model Evaluation (test set):")
        print(f"   NDCG@5         : {self.eval_metrics['ndcg@5']}")
        print(f"   Precision@3    : {self.eval_metrics['precision@3']}")

        print(f"\n🎯 Top 3 Recommendations:")
        for rec in results['top_recommendations'][:3]:
            print(f"\n   {rec['rank']}. {rec['plan_name']}")
            print(f"      Premium: ${rec['monthly_premium']:.2f}/month")
            print(f"      Savings: ${rec['annual_savings']:.2f}/year")
            print(f"      Score  : {rec['xgb_score']:.2f}")

        return results


def main():
    print("=" * 80)
    print("🏥 HEALTHCARE BENEFITS NAVIGATOR - DEMO")
    print("=" * 80)

    nav = HealthcareNavigator()

    print("\n" + "=" * 80)
    print("DEMO 1: Natural Language Q&A")
    print("=" * 80)
    nav.ask_question("What gym membership benefits are available in Massachusetts plans?")

    print("\n" + "=" * 80)
    print("DEMO 2: Plan Comparison with AI")
    print("=" * 80)
    nav.compare_plans(
        age=35, income=60000, household_size=2,
        has_chronic=False, current_plan_id="MA-BCBS-HMO-001"
    )

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()