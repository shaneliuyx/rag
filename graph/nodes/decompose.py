from __future__ import annotations
from typing import List, Dict


class QueryDecomposer:
    def __call__(self, query: str) -> List[Dict[str, str]]:
        # Placeholder decomposition: split by ' and ' into sub-queries
        parts = [p.strip() for p in query.split(' and ') if p.strip()]
        subqs = []
        for i, p in enumerate(parts, start=1):
            subqs.append({"id": f"q{i}", "text": p, "depends_on": "[]", "rationale": "heuristic split"})
        if not subqs:
            subqs = [{"id": "q1", "text": query, "depends_on": "[]", "rationale": "single"}]
        return subqs
