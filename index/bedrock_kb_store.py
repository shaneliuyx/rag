from __future__ import annotations
from typing import List, Dict, Any, Tuple


class BedrockKBStore:
    def __init__(self, knowledge_base_id: str, region: str | None = None) -> None:
        import boto3  # type: ignore  # lazy import
        self.kb_id = knowledge_base_id
        self.client = boto3.client("bedrock-agent-runtime", region_name=region) if region else boto3.client("bedrock-agent-runtime")

    def query(self, query_texts: List[str], n_results: int = 50) -> Dict[str, List[List[str]]]:
        # Align to Chroma-like return shape: {ids: [[...]], documents: [[...]], metadatas: [[...]]}
        if not query_texts:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        q = query_texts[0]
        resp = self.client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": q},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": n_results}
            },
        )
        results = resp.get("retrievalResults", [])
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for i, it in enumerate(results):
            content = it.get("content", {})
            text = content.get("text", "")
            ids.append(f"kb:{i}")
            docs.append(text)
            metas.append(it.get("metadata", {}))
        return {"ids": [ids], "documents": [docs], "metadatas": [metas]}


