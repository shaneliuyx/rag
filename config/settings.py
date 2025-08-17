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

    # v2: provider & bedrock hybrid
    llm_provider: str = os.environ.get("RAG_MCP_LLM_PROVIDER", "").strip().lower()
    bedrock_model_id: str = os.environ.get("RAG_MCP_BEDROCK_MODEL_ID", "")
    bedrock_region: str = os.environ.get("RAG_MCP_BEDROCK_REGION", "")
    aws_profile: str = os.environ.get("RAG_MCP_AWS_PROFILE", os.environ.get("AWS_PROFILE", ""))
    use_bedrock_kb: bool = os.environ.get("RAG_MCP_USE_BEDROCK_KB", "0") == "1"
    bedrock_kb_id: str = os.environ.get("RAG_MCP_BEDROCK_KB_ID", "")
    bedrock_kb_ds_id: str = os.environ.get("RAG_MCP_BEDROCK_KB_DS_ID", "")
    use_bedrock_rerank: bool = os.environ.get("RAG_MCP_USE_BEDROCK_RERANK", "0") == "1"
    bedrock_rerank_model_id: str = os.environ.get("RAG_MCP_BEDROCK_RERANK_MODEL_ID", "")
    # optional S3-backed ingest for Bedrock KB
    s3_bucket: str = os.environ.get("RAG_MCP_S3_BUCKET", "")
    s3_prefix: str = os.environ.get("RAG_MCP_S3_PREFIX", "rag/ingest")
    enable_bedrock_kb_ingest: bool = os.environ.get("RAG_MCP_ENABLE_BEDROCK_KB_INGEST", "1") == "1"
    # v2 graders & dynamics
    enable_retrieval_grader: bool = os.environ.get("RAG_MCP_ENABLE_RETRIEVAL_GRADER", "1") == "1"
    enable_hallucination_grader: bool = os.environ.get("RAG_MCP_ENABLE_HALLUCINATION_GRADER", "1") == "1"
    enable_answer_grader: bool = os.environ.get("RAG_MCP_ENABLE_ANSWER_GRADER", "1") == "1"

settings = Settings()
