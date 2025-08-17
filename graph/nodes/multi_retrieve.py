from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
from index.chroma_store import ChromaStore
from typing import Optional


STOPWORDS = set(
    """
    a an and are as at be but by for from has have if in into is it its of on or over such than that the their then there these they this to was were will with your you about can how what when where which why who
    """.split()
)


def _keywords(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]


def _variants(query: str) -> List[str]:
    kws = _keywords(query)
    keyword_query = " ".join(kws)
    return [query, keyword_query]


def _rrf_merge(results: List[Tuple[str, Dict[str, Any]]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for variant_id, res in results:
        ids_lists = res.get("ids", [[]])
        if not ids_lists or not ids_lists[0]:
            continue
        ids = ids_lists[0]
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


class MultiRetrieveNode:
    def __init__(self, store: ChromaStore, n_results: int = 50) -> None:
        self.store = store
        self.n_results = n_results
        self.kb_store = None
        try:
            from config.settings import settings
            if getattr(settings, "use_bedrock_kb", False) and getattr(settings, "bedrock_kb_id", ""):
                from index.bedrock_kb_store import BedrockKBStore
                self.kb_store = BedrockKBStore(
                    knowledge_base_id=settings.bedrock_kb_id,
                    region=getattr(settings, "bedrock_region", None),
                    profile=getattr(settings, "aws_profile", None),
                )
        except Exception:
            # Optional dependency missing or misconfigured; ignore to keep local path working
            self.kb_store = None

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        vars = _variants(query)
        res_pairs: List[Tuple[str, Dict[str, Any]]] = []
        res_cache: Dict[str, Dict[str, Any]] = {}
        for i, vq in enumerate(vars):
            res = self.store.query(query_texts=[vq], n_results=self.n_results)
            key = f"v{i}"
            res_pairs.append((key, res))
            res_cache[key] = res
        # Optional: add Bedrock KB results for original query only
        if self.kb_store is not None:
            try:
                kb_res = self.kb_store.query(query_texts=[vars[0]], n_results=self.n_results)
                res_pairs.append(("kb", kb_res))
                res_cache["kb"] = kb_res
            except Exception:
                pass
        fused = _rrf_merge(res_pairs)
        # build hit objects using the first variant's docs/metas fallback
        # We'll map id to its text/meta by scanning variant results until found
        id_to_doc: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for _, res in res_pairs:
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            for i, cid in enumerate(ids):
                if cid not in id_to_doc:
                    id_to_doc[cid] = (docs[i], metas[i])
        hits = [
            {"id": cid, "text": id_to_doc[cid][0], "metadata": id_to_doc[cid][1], "rrf_score": score}
            for cid, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ]
        return hits
