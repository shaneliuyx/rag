from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys

# Ensure project root is on sys.path for absolute imports when launched by an MCP host
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from index.chroma_store import ChromaStore
from index.ingest import load_and_chunk, ingest_to_bedrock_kb
from models.rerank_bge import rerank
from graph.builder import GraphPipeline
from config.settings import settings

mcp = FastMCP("RAG MCP Server", dependencies=["chromadb", "transformers", "sentence-transformers", "beautifulsoup4", "pypdf", "python-docx"])
store = ChromaStore()
pipeline = GraphPipeline(store)

# Keep Pydantic models for validation but don't use them as tool parameters
class IngestRequest(BaseModel):
    paths: Optional[List[str]] = Field(default=None)
    texts: Optional[List[str]] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)

class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=settings.k_top)
    allow_web_fallback: bool = Field(default=True)

class ClearRequest(BaseModel):
    scope: str = Field(default="all")
    tag_contains: Optional[str] = None

@mcp.tool()
def rag_status() -> Dict[str, Any]:
    return {
        "chroma_collection_size": store.count(),
        "collection": settings.chroma_collection,
        "persist_dir": settings.chroma_persist_dir,
    }

@mcp.tool()
def rag_ingest(req: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest documents into the RAG system"""
    # Extract parameters from the request dict
    paths = req.get('paths')
    texts = req.get('texts')
    tags = req.get('tags')
    
    # Validate input using Pydantic model
    validated_req = IngestRequest(paths=paths, texts=texts, tags=tags)
    
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    if validated_req.texts:
        for i, t in enumerate(validated_req.texts):
            ids.append(f"text:{i}")
            docs.append(t)
            tag_csv = ",".join(validated_req.tags) if validated_req.tags else ""
            metas.append({"tags": tag_csv})

    if validated_req.paths:
        file_ids, file_docs, file_metas = load_and_chunk(validated_req.paths, tags=validated_req.tags)
        ids.extend(file_ids)
        docs.extend(file_docs)
        metas.extend(file_metas)

    added = store.add_chunks(ids=ids, texts=docs, metadatas=metas)
    kb = ingest_to_bedrock_kb(ids, docs, metas)
    return {"added": added, "collection_size": store.count(), "bedrock_kb": kb}

@mcp.tool()
def rag_query(req: Dict[str, Any]) -> Dict[str, Any]:
    """Query the RAG system with a question"""
    # Extract parameters from the request dict
    query = req.get('query')
    k = req.get('k', settings.k_top)
    allow_web_fallback = req.get('allow_web_fallback', True)
    
    if not query:
        raise ValueError("Query parameter is required")
    
    # Validate input using Pydantic model
    validated_req = QueryRequest(query=query, k=k, allow_web_fallback=allow_web_fallback)
    
    out = pipeline.run(validated_req.query, top_k=validated_req.k)
    result: Dict[str, Any] = {"query": validated_req.query, "k": validated_req.k, "hits": out.get("hits", []), "answer": out.get("answer")}
    if validated_req.allow_web_fallback and "next_action" in out:
        result["next_action"] = out["next_action"]
    return result

@mcp.tool()
def rag_clear(req: Dict[str, Any]) -> Dict[str, Any]:
    """Clear documents from the RAG system"""
    # Extract parameters from the request dict
    scope = req.get('scope', 'all')
    tag_contains = req.get('tag_contains')
    
    # Validate input using Pydantic model
    validated_req = ClearRequest(scope=scope, tag_contains=tag_contains)
    
    if validated_req.scope == "all":
        store.clear_all()
    elif validated_req.scope == "by_tag" and validated_req.tag_contains:
        store.delete_where_tag_contains(validated_req.tag_contains)
    else:
        store.clear_all()
    return {"status": "ok", "chroma_collection_size": store.count()}

@mcp.tool()
def rag_export(path: str, limit: int = 1000) -> Dict[str, Any]:
    import json
    out_path = os.path.abspath(path)
    wrote = 0
    offset = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            batch = store.get_all(limit=limit, offset=offset)
            ids = batch.get("ids", [])
            docs = batch.get("documents", [])
            metas = batch.get("metadatas", [])
            if not ids:
                break
            for i in range(len(ids)):
                rec = {"id": ids[i], "text": docs[i], "metadata": metas[i]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            offset += len(ids)
    return {"status": "ok", "path": out_path, "rows": wrote}

if __name__ == "__main__":
    # Start the MCP server
    mcp.run()
