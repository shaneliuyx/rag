from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class RunState(BaseModel):
    query: str
    is_complex: bool = False
    reason: Optional[str] = None
    rewritten_query: Optional[str] = None
    candidates: List[Dict[str, Any]] = []  # raw retrieval hits
    reranked: List[Dict[str, Any]] = []   # reranked hits
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    confidence: float = 0.0
