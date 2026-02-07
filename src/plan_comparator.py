"""
Plan Comparison Engine
"""

from typing import List, Dict
import numpy as np

class PlanComparator:
    def __init__(self, rag_system, xgb_ranker):
        self.rag_system = rag_system
        self.xgb_ranker = xgb_ranker
    
    def compare_plans(self, user, current_plan, all_plans: List, user_query: str = None) -> Dict:
        # Filter eligible plans
        eligible_plans = [p for p in all_plans if p.is_eligible(user)]
        
        print(f"User eligible for {len(eligible_plans)} out of {len(all_plans)} plans")
        
        # Get RAG scores
        rag_scores = {}
        if user_query:
            rag_scores = self.rag_system.get_plan_relevance_scores(
                user_query, 
                [p.plan_id for p in eligible_plans]
            )
        
        # Get XGBoost rankings
        xgb_rankings = self.xgb_ranker.rank_plans(user, eligible_plans, rag_scores)
        xgb_scores = {plan_id: score for plan_id, score in xgb_rankings}
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_plan, eligible_plans, xgb_rankings[:5], rag_scores
        )
        
        # Cost analysis
        cost_analysis = self._analyze_costs(user, current_plan, eligible_plans)
        
        return {
            'current_plan': current_plan.plan_id,
            'eligible_plans_count': len(eligible_plans),
            'rag_scores': rag_scores,
            'xgb_scores': xgb_scores,
            'top_recommendations': recommendations,
            'cost_analysis': cost_analysis
        }
    
    def _generate_recommendations(self, current_plan, eligible_plans, top_ranked, rag_scores) -> List[Dict]:
        recommendations = []
        plan_dict = {p.plan_id: p for p in eligible_plans}
        plan_dict[current_plan.plan_id] = current_plan
        
        for rank, (plan_id, xgb_score) in enumerate(top_ranked, 1):
            if plan_id == current_plan.plan_id:
                continue
                
            plan = plan_dict[plan_id]
            
            cost_savings = (current_plan.monthly_premium * 12) - (plan.monthly_premium * 12)
            
            rationale = []
            if cost_savings > 0:
                rationale.append(f"Save ${cost_savings:.2f} annually")
            
            if plan.star_rating > current_plan.star_rating:
                rationale.append(f"Higher rating ({plan.star_rating:.1f} vs {current_plan.star_rating:.1f})")
            
            if len(plan.additional_benefits) > len(current_plan.additional_benefits):
                extra = len(plan.additional_benefits) - len(current_plan.additional_benefits)
                rationale.append(f"{extra} more benefits")
            
            recommendations.append({
                'rank': rank,
                'plan_id': plan_id,
                'plan_name': plan.plan_name,
                'carrier': plan.carrier,
                'xgb_score': float(xgb_score),
                'rag_score': float(rag_scores.get(plan_id, 0)),
                'monthly_premium': float(plan.monthly_premium),
                'annual_savings': float(cost_savings),
                'rationale': rationale,
                'key_benefits': plan.additional_benefits[:3]
            })
        
        return recommendations
    
    def _analyze_costs(self, user, current_plan, alternative_plans) -> Dict:
        current_annual_cost = current_plan.monthly_premium * 12
        premiums = [p.monthly_premium * 12 for p in alternative_plans]
        
        plan_costs = [(p, p.monthly_premium * 12) for p in alternative_plans]
        plan_costs.sort(key=lambda x: x[1])
        cheapest = plan_costs[0] if plan_costs else None
        
        return {
            'current_annual_cost': float(current_annual_cost),
            'average_market_cost': float(np.mean(premiums)) if premiums else 0,
            'cheapest_plan': {
                'name': cheapest[0].plan_name,
                'annual_cost': float(cheapest[1]),
                'savings': float(current_annual_cost - cheapest[1])
            } if cheapest else None,
            'income_burden_current': f"{(current_annual_cost / user.income * 100):.1f}%"
        }