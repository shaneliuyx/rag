import os
from dataclasses import dataclass

@dataclass
class Settings:
    chroma_persist_dir: str = os.environ.get("RAG_MCP_CHROMA_DIR", os.path.expanduser("~/Documents/rag-mcp/.chroma"))
    chroma_collection: str = os.environ.get("RAG_MCP_CHROMA_COLLECTION", "rag_collection")
    embedding_model: str = os.environ.get("RAG_MCP_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.environ.get("RAG_MCP_RERANK_MODEL", "BAAI/bge-reranker-base")
    llm_model: str = os.environ.get("RAG_MCP_LLM_MODEL", "google/gemma-3-270m-instruct")
    k_top: int = int(os.environ.get("RAG_MCP_TOP_K", 10))
    enable_generation: bool = os.environ.get("RAG_MCP_ENABLE_GENERATION", "0") == "1"
    use_ollama: bool = os.environ.get("RAG_MCP_USE_OLLAMA", "1") == "1"
    ollama_model: str = os.environ.get("RAG_MCP_OLLAMA_MODEL", "gemma3:270m")
    use_extractive_summary: bool = os.environ.get("RAG_MCP_USE_EXTRACTIVE", "1") == "1"
    summary_bullets_min: int = int(os.environ.get("RAG_MCP_BULLETS_MIN", 3))
    summary_bullets_max: int = int(os.environ.get("RAG_MCP_BULLETS_MAX", 5))
    # Self-RAG / Corrective thresholds
    selfrag_conf_threshold: float = float(os.environ.get("RAG_MCP_SELRAG_CONF", 0.6))
    relevance_threshold: float = float(os.environ.get("RAG_MCP_RELEVANCE_T", 0.2))
    max_regen_attempts: int = int(os.environ.get("RAG_MCP_MAX_REGEN", 1))
    max_rewrite_attempts: int = int(os.environ.get("RAG_MCP_MAX_REWRITE", 2))

settings = Settings()
