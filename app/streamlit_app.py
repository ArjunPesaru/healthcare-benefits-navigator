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
from training_utils import generate_training_data, split_training_data  # ✅ replaces inline loop

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MA Health Insurance Navigator",
    page_icon="🏥",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .ai-response {
        background-color: #f0f7ff;
        color: #1a1a1a;
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.75rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── System init (cached — runs once per session) ──────────────────────────────
@st.cache_resource
def initialize_system():
    plans = create_sample_ma_plans()

    # RAG must be built first — training_utils needs it for real score computation
    rag_system = BenefitRAGSystem()
    documents = rag_system.prepare_documents(plans)
    rag_system.build_index(documents)

    # ✅ Real RAG scores, no random noise
    all_examples = generate_training_data(plans, rag_system)

    # ✅ Proper train/test split
    train_examples, test_examples = split_training_data(all_examples, test_size=0.2)

    xgb_ranker = XGBoostPlanRanker()
    xgb_ranker.train(train_examples)

    # ✅ Evaluate on held-out test set
    eval_metrics = xgb_ranker.evaluate(test_examples)

    comparator = PlanComparator(rag_system, xgb_ranker)
    mistral_rag = EnhancedRAGSystemWithMistral(rag_system)

    return plans, rag_system, xgb_ranker, comparator, mistral_rag, eval_metrics


# ── Main app ──────────────────────────────────────────────────────────────────
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

    if not os.getenv('MISTRAL_API_KEY'):
        st.error("⚠️ MISTRAL_API_KEY not set. Please check your .env file.")
        return

    with st.spinner("🤖 Loading AI system..."):
        plans, rag_system, xgb_ranker, comparator, mistral_rag, eval_metrics = initialize_system()

    st.success("✅ System ready!")

    # ── Model metrics banner ──────────────────────────────────────────────────
    with st.expander("📊 Model Performance (Test Set)", expanded=True):
        if eval_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("NDCG@5",          str(eval_metrics.get('ndcg@5', 'N/A')),
                      help="Ranking quality: closer to 1.0 is better")
            m2.metric("Precision@3",     str(eval_metrics.get('precision@3', 'N/A')),
                      help="Fraction of top-3 recommendations that are relevant")
            m3.metric("Test Examples",   str(eval_metrics.get('test_examples', 'N/A')))
            m4.metric("Users Evaluated", str(eval_metrics.get('num_users', 'N/A')))
        else:
            st.warning("Evaluation metrics not available.")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Compare Plans"])

    # ── Tab 1: Q&A ────────────────────────────────────────────────────────────
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
            else:
                st.warning("Please enter a question first.")

    # ── Tab 2: Plan Comparison ────────────────────────────────────────────────
    with tab2:
        st.header("📊 Compare Health Plans")

        with st.form("user_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age     = st.number_input("Age", 18, 100, 35)
                income  = st.number_input("Annual Income ($)", 0, 500000, 60000, 5000)
            with col2:
                household = st.number_input("Household Size", 1, 10, 2)
                chronic   = st.checkbox("I have chronic conditions")
            with col3:
                current_plan = st.selectbox(
                    "Current Plan",
                    options=[p.plan_id for p in plans],
                    format_func=lambda x: next(p.plan_name for p in plans if p.plan_id == x)
                )

            submitted = st.form_submit_button("🔍 Compare Plans", use_container_width=True)

        if submitted:
            user    = UserProfile(age, income, household, "02108", chronic, current_plan)
            current = next(p for p in plans if p.plan_id == current_plan)

            with st.spinner("🤖 Analyzing plans with AI..."):
                results = comparator.compare_plans(
                    user=user,
                    current_plan=current,
                    all_plans=plans,
                    user_query="comprehensive coverage with wellness benefits"
                )

                user_dict = {
                    'age': age, 'income': income,
                    'household_size': household,
                    'has_chronic_conditions': chronic
                }
                explanation = mistral_rag.mistral.generate_plan_comparison_summary(
                    results, user_dict
                )

            # AI narrative
            st.subheader("🤖 AI Analysis")
            st.markdown(f"""
            <div class='ai-response'>
                {explanation}
            </div>
            """, unsafe_allow_html=True)

            # Key metrics
            st.subheader("📈 Key Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Eligible Plans",    results['eligible_plans_count'])
            c2.metric("Current Annual Cost",
                      f"${results['cost_analysis']['current_annual_cost']:,.0f}")
            c3.metric("Market Average",
                      f"${results['cost_analysis']['average_market_cost']:,.0f}")
            c4.metric("Income Burden",     results['cost_analysis']['income_burden_current'])
            c5.metric("Model NDCG@5",      eval_metrics['ndcg@5'],
                      help="Ranking model quality on held-out test set")

            # Recommendations
            st.subheader("🎯 Top Recommendations")
            for rec in results['top_recommendations'][:3]:
                with st.expander(f"#{rec['rank']} — {rec['plan_name']}", expanded=(rec['rank'] == 1)):
                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Monthly Premium",  f"${rec['monthly_premium']:.2f}")

                    savings = rec['annual_savings']
                    col2.metric("Annual Savings",
                                f"${abs(savings):,.0f}",
                                delta=f"{'↓ Save' if savings > 0 else '↑ Extra'} ${abs(savings):,.0f}")

                    col3.metric("XGBoost Score",    f"{rec['xgb_score']:.3f}")
                    col4.metric("RAG Score",        f"{rec['rag_score']:.3f}")

                    st.markdown("**Why This Plan:**")
                    for reason in rec['rationale']:
                        st.markdown(f"- {reason}")

                    if rec['key_benefits']:
                        st.markdown("**Key Benefits:**")
                        for b in rec['key_benefits']:
                            st.markdown(f"- {b}")


if __name__ == "__main__":
    main()