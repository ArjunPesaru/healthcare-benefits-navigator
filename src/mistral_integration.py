"""
Mistral AI Integration - v1.12.0
"""

import os
from typing import List, Dict, Optional, Tuple
from mistralai import Mistral

class MistralLLMIntegration:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-large-latest"
        print(f"✓ Mistral AI initialized with model: {self.model}")
    
    def generate_benefit_answer(self, query: str, retrieved_docs: List[Tuple[str, float, str]], 
                               user_profile: Optional[Dict] = None) -> str:
        # Prepare context
        context_parts = []
        for i, (doc, score, plan_id) in enumerate(retrieved_docs[:5], 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{doc}\n")
        
        context = "\n".join(context_parts)
        
        system_prompt = """You are a helpful healthcare insurance assistant for Massachusetts health plans.

Your role:
1. Answer questions about health insurance benefits clearly
2. Use information from provided sources
3. Explain complex terms in simple language
4. Be empathetic and patient

Guidelines:
- Always cite sources
- If unclear, say so honestly
- Focus on practical, actionable information
- Use simple, jargon-free language"""

        user_context = ""
        if user_profile:
            user_context = f"""
User Context:
- Age: {user_profile.get('age', 'N/A')}
- Income: ${user_profile.get('income', 0):,}
- Household: {user_profile.get('household_size', 'N/A')}
- Chronic Conditions: {'Yes' if user_profile.get('has_chronic_conditions') else 'No'}

"""

        user_prompt = f"""{user_context}Question: {query}

Information from Insurance Plans:
{context}

Please provide a clear answer based on the information above. Cite sources."""

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️  Mistral API error: {e}")
            return f"Error: {e}\n\nRelevant info:\n{context[:500]}..."
    
    def generate_plan_comparison_summary(self, comparison_results: Dict, user_profile: Dict) -> str:
        current_plan = comparison_results.get('current_plan', 'Unknown')
        eligible_count = comparison_results.get('eligible_plans_count', 0)
        recommendations = comparison_results.get('top_recommendations', [])[:3]
        cost_analysis = comparison_results.get('cost_analysis', {})
        
        context = f"""
User Profile:
- Age: {user_profile.get('age')}
- Annual Income: ${user_profile.get('income'):,}
- Household Size: {user_profile.get('household_size')}
- Chronic Conditions: {'Yes' if user_profile.get('has_chronic_conditions') else 'No'}

Current Plan: {current_plan}
Current Annual Cost: ${cost_analysis.get('current_annual_cost', 0):,.2f}
Market Average: ${cost_analysis.get('average_market_cost', 0):,.2f}
Income Burden: {cost_analysis.get('income_burden_current', 'N/A')}

Eligible Plans: {eligible_count}

Top 3 Recommendations:
"""
        
        for rec in recommendations:
            context += f"""
{rec['rank']}. {rec['plan_name']} ({rec['carrier']})
   - Monthly Premium: ${rec['monthly_premium']:.2f}
   - Annual Savings: ${rec['annual_savings']:.2f}
   - Match Score: {rec['xgb_score']:.2f}
   - Why: {', '.join(rec['rationale'])}
"""
        
        prompt = f"""{context}

Provide a comprehensive summary of this comparison.

Include:
1. Assessment of current situation
2. Top 3 recommended plans and why
3. Cost differences and savings
4. Important considerations
5. Next steps

Use warm, helpful tone that's easy to understand."""

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a healthcare insurance advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️  Mistral API error: {e}")
            return f"Unable to generate summary: {e}"

class EnhancedRAGSystemWithMistral:
    def __init__(self, rag_system, mistral_api_key: Optional[str] = None):
        self.rag_system = rag_system
        self.mistral = MistralLLMIntegration(api_key=mistral_api_key)
    
    def answer_question(self, query: str, user_profile: Optional[Dict] = None, top_k: int = 5) -> Dict:
        retrieved_docs = self.rag_system.retrieve(query, top_k=top_k)
        
        answer = self.mistral.generate_benefit_answer(
            query=query,
            retrieved_docs=retrieved_docs,
            user_profile=user_profile
        )
        
        sources = [
            {
                'text': doc[:200] + "..." if len(doc) > 200 else doc,
                'score': float(score),
                'plan_id': plan_id
            }
            for doc, score, plan_id in retrieved_docs
        ]
        
        return {
            'question': query,
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources)
        }