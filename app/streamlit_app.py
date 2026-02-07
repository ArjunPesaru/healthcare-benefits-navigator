"""
Streamlit Web App for Healthcare Benefits Navigator
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

from data_models import UserProfile
from sample_data import create_sample_ma_plans
from rag_system import BenefitRAGSystem
from xgboost_ranker import XGBoostPlanRanker
from plan_comparator import PlanComparator
from mistral_integration import EnhancedRAGSystemWithMistral
import numpy as np

# Page config
st.set_page_config(
    page_title="MA Health Insurance Navigator",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .ai-response {
        background-color: #1f77b4;
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def initialize_system():
    """Initialize the system once"""
    plans = create_sample_ma_plans()
    
    rag_system = BenefitRAGSystem()
    documents = rag_system.prepare_documents(plans)
    rag_system.build_index(documents)
    
    xgb_ranker = XGBoostPlanRanker()
    
    # Generate training data
    training_examples = []
    users = [
        UserProfile(28, 45000, 1, "02108", False),
        UserProfile(35, 75000, 3, "02115", False),
        UserProfile(52, 60000, 2, "01105", True),
    ]
    
    for user in users:
        for plan in plans:
            if not plan.is_eligible(user):
                continue
            
            relevance = 2.0
            affordability = 1 - min((plan.monthly_premium * 12) / user.income, 1.0)
            relevance += affordability * 1.5
            relevance = min(max(relevance, 0), 4)
            
            training_examples.append({
                'user': user,
                'plan': plan,
                'rag_score': np.random.uniform(0.5, 0.9),
                'relevance': relevance
            })
    
    xgb_ranker.train(training_examples)
    
    comparator = PlanComparator(rag_system, xgb_ranker)
    mistral_rag = EnhancedRAGSystemWithMistral(rag_system)
    
    return plans, rag_system, xgb_ranker, comparator, mistral_rag

# Main app
def main():
    st.markdown('<h1 class="main-header">🏥 Massachusetts Health Insurance Navigator</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>
            Compare plans and discover benefits using <strong>AI-powered analysis</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not os.getenv('MISTRAL_API_KEY'):
        st.error("⚠️ MISTRAL_API_KEY not set. Please check your .env file.")
        return
    
    # Initialize system
    with st.spinner("🤖 Loading AI system..."):
        plans, rag_system, xgb_ranker, comparator, mistral_rag = initialize_system()
    
    st.success("✅ System ready!")
    
    # Tabs
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Compare Plans"])
    
    # Tab 1: Q&A
    with tab1:
        st.header("💬 Ask About Benefits")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What gym benefits are available?"
            )
        
        with col2:
            age = st.number_input("Your age", 18, 100, 35)
        
        if st.button("🤖 Ask AI", type="primary", use_container_width=True):
            if question:
                with st.spinner("🤖 AI is thinking..."):
                    user_context = {
                        'age': age,
                        'income': 60000,
                        'household_size': 2,
                        'has_chronic_conditions': False
                    }
                    
                    result = mistral_rag.answer_question(question, user_context)
                    
                    st.markdown(f"""
                    <div class='ai-response'>
                        <strong>🤖 AI Answer:</strong><br><br>
                        {result['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"**Source {i}** (Score: {source['score']:.2f})")
                            st.text(source['text'])
                            st.divider()
    
    # Tab 2: Plan Comparison
    with tab2:
        st.header("📊 Compare Health Plans")
        
        with st.form("user_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", 18, 100, 35)
                income = st.number_input("Annual Income ($)", 0, 500000, 60000, 5000)
            
            with col2:
                household = st.number_input("Household Size", 1, 10, 2)
                chronic = st.checkbox("I have chronic conditions")
            
            with col3:
                current_plan = st.selectbox(
                    "Current Plan",
                    options=[p.plan_id for p in plans],
                    format_func=lambda x: next(p.plan_name for p in plans if p.plan_id == x)
                )
            
            submitted = st.form_submit_button("🔍 Compare Plans", use_container_width=True)
        
        if submitted:
            user = UserProfile(age, income, household, "02108", chronic, current_plan)
            
            current = next(p for p in plans if p.plan_id == current_plan)
            
            with st.spinner("🤖 Analyzing plans with AI..."):
                results = comparator.compare_plans(
                    user=user,
                    current_plan=current,
                    all_plans=plans,
                    user_query="comprehensive coverage"
                )
                
                # Generate AI explanation
                user_dict = {
                    'age': age,
                    'income': income,
                    'household_size': household,
                    'has_chronic_conditions': chronic
                }
                
                explanation = mistral_rag.mistral.generate_plan_comparison_summary(
                    results, user_dict
                )
                
                # Display results
                st.subheader("🤖 AI Analysis")
                st.markdown(f"""
                <div class='ai-response'>
                    {explanation}
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                st.subheader("📈 Key Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Eligible Plans", results['eligible_plans_count'])
                
                with col2:
                    st.metric("Current Annual Cost", 
                             f"${results['cost_analysis']['current_annual_cost']:,.0f}")
                
                with col3:
                    st.metric("Income Burden", 
                             results['cost_analysis']['income_burden_current'])
                
                # Recommendations
                st.subheader("🎯 Top Recommendations")
                
                for rec in results['top_recommendations'][:3]:
                    with st.expander(f"#{rec['rank']} - {rec['plan_name']}", expanded=(rec['rank']==1)):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Monthly Premium", f"${rec['monthly_premium']:.2f}")
                        
                        with col2:
                            savings = rec['annual_savings']
                            st.metric("Annual Savings", f"${abs(savings):,.0f}",
                                     delta=f"{'Save' if savings > 0 else 'Cost'} ${abs(savings):,.0f}")
                        
                        with col3:
                            st.metric("Match Score", f"{rec['xgb_score']:.2f}")
                        
                        st.markdown("**Why This Plan:**")
                        for reason in rec['rationale']:
                            st.markdown(f"- {reason}")

if __name__ == "__main__":
    main()