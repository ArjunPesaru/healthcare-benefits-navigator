"""
RAG System for Healthcare Benefits
"""

import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import faiss

class BenefitRAGSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.benefit_documents = []
        self.plan_mapping = {}
    
    def prepare_documents(self, plans) -> List[str]:
        documents = []
        for plan in plans:
            doc = f"Plan: {plan.plan_name} by {plan.carrier}. {plan.benefits_description}"
            documents.append(doc)
            self.plan_mapping[len(documents) - 1] = plan.plan_id
            
            for benefit in plan.additional_benefits:
                benefit_doc = f"{plan.plan_name} includes: {benefit}"
                documents.append(benefit_doc)
                self.plan_mapping[len(documents) - 1] = plan.plan_id
        
        self.benefit_documents = documents
        return documents
    
    def build_index(self, documents: List[str]) -> None:
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.benefit_documents):
                doc = self.benefit_documents[idx]
                plan_id = self.plan_mapping[idx]
                results.append((doc, float(score), plan_id))
        
        return results
    
    def get_plan_relevance_scores(self, query: str, plan_ids: List[str], top_k: int = 20) -> Dict[str, float]:
        results = self.retrieve(query, top_k=top_k)
        
        plan_scores = {pid: 0.0 for pid in plan_ids}
        plan_counts = {pid: 0 for pid in plan_ids}
        
        for doc, score, plan_id in results:
            if plan_id in plan_scores:
                plan_scores[plan_id] += score
                plan_counts[plan_id] += 1
        
        for plan_id in plan_scores:
            if plan_counts[plan_id] > 0:
                plan_scores[plan_id] /= plan_counts[plan_id]
        
        return plan_scores