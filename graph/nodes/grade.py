from __future__ import annotations
from typing import Dict, Any
from .selfrag import selfrag_checks


def grade_hallucination(answer: str, hits: list[dict[str, Any]], question: str, threshold: float = 0.6) -> Dict[str, Any]:
    sr = selfrag_checks(answer, hits, question)
    # Use fraction of bullets that are faithful + cited as confidence proxy
    faithful_cited = 0
    total = max(1, len(sr.get("results", [])))
    for r in sr.get("results", []):
        if r.get("faithful") and r.get("has_citation"):
            faithful_cited += 1
    conf = faithful_cited / total
    return {"score": conf, "pass": conf >= threshold, "selfrag": sr}


def grade_relevance(answer: str, question: str, threshold: float = 0.2) -> Dict[str, Any]:
    # Lightweight relevance via keyword overlap between question and answer
    import re

    def kw(s: str) -> set[str]:
        toks = re.findall(r"[A-Za-z0-9_\-]+", s.lower())
        stop = set(
            "a an and are as at be but by for from has have if in into is it its of on or over such than that the their then there these they this to was were will with your you about can how what when where which why who".split()
        )
        return {t for t in toks if t not in stop and len(t) >= 3}

    qk = kw(question)
    ak = kw(answer)
    # Treat summarize/overview/explain intents as relevant
    intent = any(w in question.lower() for w in ["summarize", "summary", "overview", "explain", "describe"])
    if intent:
        return {"score": 1.0, "pass": True, "overlap_terms": sorted(qk & ak)}
    inter = qk & ak
    score = len(inter) / max(1, len(qk)) if qk else 1.0
    return {"score": score, "pass": score >= threshold, "overlap_terms": sorted(inter)}


