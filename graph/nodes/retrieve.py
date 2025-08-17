from __future__ import annotations
from typing import List, Dict, Any
from index.chroma_store import ChromaStore


class RetrieveNode:
    def __init__(self, store: ChromaStore, n_results: int = 50) -> None:
        self.store = store
        self.n_results = n_results

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        res = self.store.query(query_texts=[query], n_results=self.n_results)
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        hits: List[Dict[str, Any]] = []
        for i, d in enumerate(docs):
            hits.append({"id": ids[i], "text": d, "metadata": metas[i]})
        return hits
