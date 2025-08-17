from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re


def _extract_bullets(text: str) -> List[str]:
    bullets: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(('-', '•', '*')):
            s = s.lstrip('-•* ').strip()
            # skip section headings like "Compare:", "Usage:", "Summary:"
            if s.lower() in ("compare:", "usage:", "summary:"):
                continue
            bullets.append(s)
    return bullets


def _keywords(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    stop = set("a an and are as at be but by for from has have if in into is it its of on or over such than that the their then there these they this to was were will with your you about can how what when where which why who".split())
    return [t for t in toks if t not in stop and len(t) >= 3]


def _overlap(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    return len(inter) / max(1, len(sa))


def selfrag_checks(answer: str, hits: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    bullets = _extract_bullets(answer)
    q_kw = _keywords(question)
    passed = 0
    results: List[Dict[str, Any]] = []

    for idx, b in enumerate(bullets, start=1):
        # parse cited index if present
        m = re.search(r"\[#(\d+)\]", b)
        cited = int(m.group(1)) if m else None
        faithful = False
        support_i = cited if cited and 1 <= cited <= len(hits) else 1
        passage = hits[support_i - 1]["text"] if hits else ""
        b_kw = _keywords(b)
        p_kw = _keywords(passage)
        ov = _overlap(b_kw, p_kw)
        faithful = ov >= 0.2 or (('"' in b) and any(seg.strip('"') in passage for seg in re.findall(r'"([^"]{5,120})"', b)))
        has_citation = cited is not None
        covers_query = _overlap(q_kw, b_kw) >= 0.2
        ok = faithful and has_citation and covers_query
        if ok:
            passed += 1
        results.append({
            "bullet_index": idx,
            "cited": cited,
            "faithful": faithful,
            "has_citation": has_citation,
            "covers_query": covers_query,
            "overlap": ov,
        })

    confidence = passed / max(1, len(bullets))
    missing_citations = [r["bullet_index"] for r in results if not r["has_citation"]]
    return {
        "bullets": bullets,
        "results": results,
        "confidence": confidence,
        "missing_citations": missing_citations,
    }
