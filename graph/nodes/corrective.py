from __future__ import annotations
from typing import Dict, Any
import re


def rewrite_query(original: str) -> str:
    # Simple corrective rewrite: add synonyms and relax phrases
    q = original
    q = re.sub(r"\bvs\b", "versus", q, flags=re.I)
    q = re.sub(r"\bcompare\b", "compare differences similarities", q, flags=re.I)
    q = q + " summary overview key points"
    return q


class CorrectiveRAG:
    def __init__(self, max_iters: int = 2):
        self.max_iters = max_iters

    def __call__(self, run_once_fn, query: str, top_k: int, threshold: float) -> Dict[str, Any]:
        cur_q = query
        last = None
        for _ in range(self.max_iters):
            out = run_once_fn(cur_q, top_k)
            last = out
            conf = out.get("selfrag", {}).get("confidence", 0.0)
            if conf >= threshold:
                break
            cur_q = rewrite_query(cur_q)
        return last or {}
