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
import numpy as np

class HealthcareNavigator:
    def __init__(self):
        print("🏥 Initializing Healthcare Benefits Navigator...")
        
        # Load plans
        self.plans = create_sample_ma_plans()
        print(f"✓ Loaded {len(self.plans)} health plans")
        
        # Initialize RAG
        self.rag_system = BenefitRAGSystem()
        documents = self.rag_system.prepare_documents(self.plans)
        self.rag_system.build_index(documents)
        print(f"✓ RAG system ready with {len(documents)} documents")
        
        # Initialize XGBoost
        self.xgb_ranker = XGBoostPlanRanker()
        training_data = self._generate_training_data()
        self.xgb_ranker.train(training_data)
        print(f"✓ XGBoost trained with {len(training_data)} examples")
        
        # Initialize comparator
        self.comparator = PlanComparator(self.rag_system, self.xgb_ranker)
        
        # Initialize Mistral
        self.mistral_rag = EnhancedRAGSystemWithMistral(self.rag_system)
        print("✓ Mistral AI ready")
        
        print("\n✅ System fully initialized!\n")
    
    def _generate_training_data(self):
        """Generate synthetic training data"""
        training_examples = []
        
        users = [
            UserProfile(28, 45000, 1, "02108", False),
            UserProfile(35, 75000, 3, "02115", False),
            UserProfile(52, 60000, 2, "01105", True),
            UserProfile(24, 28000, 1, "02139", False),
        ]
        
        for user in users:
            for plan in self.plans:
                if not plan.is_eligible(user):
                    continue
                
                relevance = 2.0
                affordability = 1 - min((plan.monthly_premium * 12) / user.income, 1.0)
                relevance += affordability * 1.5
                relevance += (plan.star_rating - 4.0) * 0.5
                relevance += len(plan.additional_benefits) * 0.1
                
                if user.has_chronic_conditions and plan.out_of_pocket_max < 6000:
                    relevance += 0.5
                
                relevance = min(max(relevance, 0), 4)
                
                training_examples.append({
                    'user': user,
                    'plan': plan,
                    'rag_score': np.random.uniform(0.5, 0.9),
                    'relevance': relevance
                })
        
        return training_examples
    
    def ask_question(self, question: str, user_age: int = 35, user_income: float = 60000):
        """Ask a question about benefits"""
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
    
    def compare_plans(self, age: int, income: float, household_size: int, 
                     has_chronic: bool, current_plan_id: str):
        """Compare plans for a user"""
        user = UserProfile(age, income, household_size, "02108", has_chronic, current_plan_id)
        
        current_plan = next((p for p in self.plans if p.plan_id == current_plan_id), None)
        if not current_plan:
            print(f"❌ Plan {current_plan_id} not found")
            return None
        
        print(f"\n📊 Comparing plans for:")
        print(f"   Age: {age}, Income: ${income:,}, Household: {household_size}")
        print(f"   Current: {current_plan.plan_name}\n")
        
        results = self.comparator.compare_plans(
            user=user,
            current_plan=current_plan,
            all_plans=self.plans,
            user_query="comprehensive coverage with wellness benefits"
        )
        
        # Generate Mistral explanation
        user_dict = {
            'age': age,
            'income': income,
            'household_size': household_size,
            'has_chronic_conditions': has_chronic
        }
        
        explanation = self.mistral_rag.mistral.generate_plan_comparison_summary(
            results, user_dict
        )
        
        print("=" * 80)
        print("🤖 AI ANALYSIS")
        print("=" * 80)
        print(explanation)
        print("=" * 80)
        
        print(f"\n📈 Key Metrics:")
        print(f"   Eligible Plans: {results['eligible_plans_count']}")
        print(f"   Current Cost: ${results['cost_analysis']['current_annual_cost']:,.2f}/year")
        
        print(f"\n🎯 Top 3 Recommendations:")
        for rec in results['top_recommendations'][:3]:
            print(f"\n   {rec['rank']}. {rec['plan_name']}")
            print(f"      Premium: ${rec['monthly_premium']:.2f}/month")
            print(f"      Savings: ${rec['annual_savings']:.2f}/year")
            print(f"      Score: {rec['xgb_score']:.2f}")
        
        return results

def main():
    print("=" * 80)
    print("🏥 HEALTHCARE BENEFITS NAVIGATOR - DEMO")
    print("=" * 80)
    
    # Initialize
    nav = HealthcareNavigator()
    
    # Demo 1: Ask questions
    print("\n" + "=" * 80)
    print("DEMO 1: Natural Language Q&A")
    print("=" * 80)
    
    nav.ask_question("What gym membership benefits are available in Massachusetts plans?")
    
    # Demo 2: Compare plans
    print("\n" + "=" * 80)
    print("DEMO 2: Plan Comparison with AI")
    print("=" * 80)
    
    nav.compare_plans(
        age=35,
        income=60000,
        household_size=2,
        has_chronic=False,
        current_plan_id="MA-BCBS-HMO-001"
    )
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()