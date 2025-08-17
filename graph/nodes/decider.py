from __future__ import annotations
import re
from typing import Dict


CUES = [
    "compare", "versus", "vs", "trade-offs", "pros and cons", "timeline",
    "steps", "step-by-step", "how to", "pipeline", "architecture", "advantages",
    "disadvantages", "between", "difference", "differences",
]


class ComplexityDecider:
    def __call__(self, query: str) -> Dict[str, str]:
        q = query.strip().lower()
        score = 0
        reasons = []
        for cue in CUES:
            if cue in q:
                score += 1
                reasons.append(f"cue:{cue}")
        # crude entity count: capitalized tokens not at start
        caps = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+\b", query)
        if len(caps) >= 3:
            score += 1
            reasons.append(f"entities:{len(caps)}")
        # conjunctions
        if q.count(" and ") >= 2 or " then " in q:
            score += 1
            reasons.append("multi-step")
        label = "Complex" if score >= 1 else "Simple"
        return {"label": label, "reason": ", ".join(reasons) or "heuristic: none"}
