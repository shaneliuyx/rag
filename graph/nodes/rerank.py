from __future__ import annotations
from typing import List, Dict, Any
from models.rerank_bge import rerank


class RerankNode:
    def __init__(self, top_k: int = 10) -> None:
        # default top_k is final cap; we'll rerank larger candidate sets upstream
        self.top_k = top_k

    def __call__(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        docs = [h["text"] for h in hits]
        ranking = rerank(query, docs, top_k=self.top_k)
        return [
            {**hits[idx], "score": float(score)}
            for idx, score in ranking
        ]
