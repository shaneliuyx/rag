from __future__ import annotations
from typing import Dict, Any
from index.chroma_store import ChromaStore
from graph.nodes.retrieve import RetrieveNode
from graph.nodes.multi_retrieve import MultiRetrieveNode
from graph.nodes.rerank import RerankNode
from graph.nodes.synthesize import SynthesizeNode
from graph.nodes.decider import ComplexityDecider
from graph.nodes.decompose import QueryDecomposer
from graph.nodes.decompose_llm import LLMDecomposer
from graph.nodes.selfrag import selfrag_checks
from graph.nodes.grade import grade_hallucination, grade_relevance
from graph.nodes.llm_graders import grade_doc_relevance, grade_generation_grounded, grade_answer_addresses
from graph.nodes.corrective import CorrectiveRAG


class GraphPipeline:
    def __init__(self, store: ChromaStore) -> None:
        # Use multi-variant retrieval for better recall
        self.retrieve = MultiRetrieveNode(store)
        self.rerank = RerankNode()
        self.synthesize = SynthesizeNode()
        self.decider = ComplexityDecider()
        self.decomposer = QueryDecomposer()
        self.llm_decomposer = LLMDecomposer()
        self.corrective = CorrectiveRAG(max_iters=2)

    def run(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        decision = self.decider(query)
        subqs = self.decomposer(query) if decision["label"] == "Complex" else [{"id": "q1", "text": query}]
        # If LLM decomposition available, prefer it and merge with heuristic plan
        llm_plan = self.llm_decomposer(query)
        if llm_plan:
            subqs = llm_plan

        # Helper: topologically order sub-queries
        def topo_sort(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
            indeg: dict[str, int] = {}
            for n in nodes:
                nid = n.get("id") or n.get("text")
                indeg[nid] = 0
            for n in nodes:
                nid = n.get("id") or n.get("text")
                for d in (n.get("depends_on", []) or []):
                    indeg[nid] = indeg.get(nid, 0) + 1
                    indeg.setdefault(d, 0)
            out: list[dict[str, Any]] = []
            ready = [n for n in nodes if indeg.get(n.get("id") or n.get("text"), 0) == 0]
            seen: set[str] = set()
            while ready:
                cur = ready.pop(0)
                cid = cur.get("id") or cur.get("text")
                out.append(cur)
                seen.add(cid)
                for n in nodes:
                    nid = n.get("id") or n.get("text")
                    if nid in seen:
                        continue
                    if cid in (n.get("depends_on", []) or []):
                        indeg[nid] = max(0, indeg.get(nid, 0) - 1)
                        if indeg[nid] == 0:
                            ready.append(n)
            return out or nodes

        def execute_plan(q: str, nodes: list[dict[str, Any]], tk: int) -> dict[str, Any]:
            ordered = topo_sort(nodes)
            merged_hits: list[dict[str, Any]] = []
            id_to_global: dict[str, int] = {}
            bullets_out: list[str] = []
            sections: dict[str, list[str]] = {"Compare": [], "Usage": [], "Summary": []}
            import re as _re
            for sub in ordered:
                text = sub.get("text", q)
                hits = self.retrieve(text)
                rr = self.rerank(text, hits)[:tk]
                local_idx_to_id = {i + 1: rr[i]["id"] for i in range(len(rr))}
                for i, h in enumerate(rr):
                    cid = h["id"]
                    if cid not in id_to_global:
                        id_to_global[cid] = len(merged_hits) + 1
                        merged_hits.append(h)
                sy = self.synthesize(text, rr)
                lines = [ln.strip() for ln in sy.get("answer", "").splitlines() if ln.strip()]
                for ln in lines:
                    if not ln.startswith("-"):
                        ln = f"- {ln}"
                    def _repl(m):
                        n = int(m.group(1))
                        cid = local_idx_to_id.get(n)
                        if not cid:
                            return m.group(0)
                        gid = id_to_global.get(cid, n)
                        return f"[#{gid}]"
                    ln = _re.sub(r"\[#(\d+)\]", _repl, ln)
                    bullets_out.append(ln)
                    # assign to section based on sub-query intent
                    t = (sub.get("type") or "").lower()
                    txt_lower = text.lower()
                    if "compare" in txt_lower or t == "lookup" and ("vs" in txt_lower or "difference" in txt_lower):
                        sections["Compare"].append(ln)
                    elif "usage" in txt_lower or "example" in txt_lower:
                        sections["Usage"].append(ln)
                    elif t == "synthesis":
                        sections["Summary"].append(ln)
                    else:
                        sections["Summary"].append(ln)
            # build structured answer with headings
            def cap(xs: list[str]) -> list[str]:
                # cap each section to at most 3 bullets
                return xs[: min(3, tk)]
            parts: list[str] = []
            if sections["Compare"]:
                parts.append("- Compare:")
                parts.extend(cap(sections["Compare"]))
            if sections["Usage"]:
                parts.append("- Usage:")
                parts.extend(cap(sections["Usage"]))
            if sections["Summary"]:
                parts.append("- Summary:")
                parts.extend(cap(sections["Summary"]))
            answer = "\n".join(parts if parts else bullets_out[: tk]) if (sections["Compare"] or sections["Usage"] or sections["Summary"]) else ""
            return {"hits": merged_hits[: tk], "answer": answer, "sections": sections}

        def run_once(q: str, tk: int):
            # Simple path
            h = self.retrieve(q)
            # Optional LLM-based retrieval grading (pre-filter)
            from config.settings import settings
            if getattr(settings, "enable_retrieval_grader", False):
                filtered: list[dict[str, Any]] = []
                for hit in h:
                    gd = grade_doc_relevance(q, hit.get("text", ""))
                    if gd.get("binary_score") == "yes":
                        filtered.append(hit)
                h = filtered or h
            rr = self.rerank(q, h)[:tk]
            sy = self.synthesize(q, rr)
            sr = selfrag_checks(sy.get("answer", ""), rr, q)
            # Optional LLM-based post graders
            if getattr(settings, "enable_hallucination_grader", False):
                gg = grade_generation_grounded(q, [h.get("text", "") for h in rr], sy.get("answer", ""))
                gh = {"score": 1.0 if gg.get("binary_score") == "yes" else 0.0, "pass": gg.get("binary_score") == "yes", "raw": gg}
            else:
                gh = grade_hallucination(sy.get("answer", ""), rr, q)
            if getattr(settings, "enable_answer_grader", False):
                ga = grade_answer_addresses(q, sy.get("answer", ""))
                gr = {"score": 1.0 if ga.get("binary_score") == "yes" else 0.0, "pass": ga.get("binary_score") == "yes", "raw": ga}
            else:
                gr = grade_relevance(sy.get("answer", ""), q)
            return {"hits": rr, "answer": sy.get("answer", ""), "selfrag": sr, "grade_hallucination": gh, "grade_relevance": gr}

        # Choose execution mode
        if decision["label"] == "Complex" or len(subqs) > 1:
            initial = execute_plan(query, subqs, top_k)
            sr = selfrag_checks(initial.get("answer", ""), initial.get("hits", []), query)
            gh = grade_hallucination(initial.get("answer", ""), initial.get("hits", []), query)
            gr = grade_relevance(initial.get("answer", ""), query)
            out = {**initial, "selfrag": sr, "grade_hallucination": gh, "grade_relevance": gr}
        else:
            out = run_once(query, top_k)
        # Two-phase corrective loop with thresholds from settings
        from config.settings import settings
        regen_left = settings.max_regen_attempts
        while regen_left > 0 and not out.get("grade_hallucination", {}).get("pass", False):
            if decision["label"] == "Complex" or len(subqs) > 1:
                initial = execute_plan(query, subqs, top_k)
                sr = selfrag_checks(initial.get("answer", ""), initial.get("hits", []), query)
                gh = grade_hallucination(initial.get("answer", ""), initial.get("hits", []), query)
                gr = grade_relevance(initial.get("answer", ""), query)
                out = {**initial, "selfrag": sr, "grade_hallucination": gh, "grade_relevance": gr}
            else:
                out = run_once(query, top_k)
            regen_left -= 1

        if not out.get("grade_relevance", {}).get("pass", False):
            out = self.corrective(run_once, query, top_k, threshold=settings.selfrag_conf_threshold)
            # If still not relevant, suggest host-level web search via the existing web-search MCP
            if not out.get("grade_relevance", {}).get("pass", False):
                out["next_action"] = {
                    "type": "web_search",
                    "query": query,
                    "limit": 5,
                    "instructions": "Use the 'web-search' MCP server tool 'search' with {query, limit}. Then re-run summarize with cites."
                }

        return {"query": query, "decision": decision, "sub_queries": subqs, **out}
