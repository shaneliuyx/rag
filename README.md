# RAG MCP (Chroma + Gemma)

A self-contained RAG MCP server exposing tools for ingesting documents and answering questions with retrieval, reranking, optional generation, query complexity detection, and simple decomposition.

## Installation

Prereqs:
- Python 3.10+
- macOS/Linux
- Optional: [Ollama](https://ollama.com/) for local generation

Steps:
1) Clone or copy this folder
2) Install Python deps
```bash
python3 -m pip install -r /Users/yuxinliu/Documents/rag-mcp/requirements.txt
```
3) (Optional) Install Ollama and pull a generator
```bash
ollama pull gemma2:2b
# or use the provided local GGUF flow for gemma-3-270m in the docs
```

## Quickstart

1) Install deps
```bash
python3 -m pip install -r /Users/yuxinliu/Documents/rag-mcp/requirements.txt
```

2) Ingest some files
```bash
python3 /Users/yuxinliu/Documents/rag-mcp/cli/rag.py ingest /path/to/docs
```

3) Query (generation disabled by default)
```bash
RAG_MCP_ENABLE_GENERATION=0 python3 /Users/yuxinliu/Documents/rag-mcp/cli/rag.py query "Your question" -k 8
```

4) Start MCP server
```bash
python3 /Users/yuxinliu/Documents/rag-mcp/mcp_server/server.py
```

Add to your MCP host config:
```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "python3",
      "args": ["/Users/yuxinliu/Documents/rag-mcp/mcp_server/server.py"],
      "env": {
        "RAG_MCP_CHROMA_DIR": "/Users/yuxinliu/Documents/rag-mcp/.chroma",
        "RAG_MCP_CHROMA_COLLECTION": "rag_collection",
        "RAG_MCP_ENABLE_GENERATION": "0",
        "RAG_MCP_USE_OLLAMA": "1",
        "RAG_MCP_OLLAMA_MODEL": "gemma2:2b"
      }
    }
  }
}
```

## User Guide

### Ingestion
- Files and directories are accepted. Supported: md, txt, pdf, html, docx.
- Example:
```bash
python3 cli/rag.py ingest ~/Documents/notes ~/Documents/papers/*.pdf --tags notes papers
```

### Querying
- Local CLI:
```bash
# extractive fallback (no generation)
RAG_MCP_ENABLE_GENERATION=0 python3 cli/rag.py query "How to convert DOCX to Markdown?"

# with local generation via Ollama gemma2:2b
RAG_MCP_ENABLE_GENERATION=1 RAG_MCP_USE_OLLAMA=1 RAG_MCP_OLLAMA_MODEL=gemma2:2b \
  python3 cli/rag.py query "Summarize key features of the server with [#i] cites." -k 6
```

- From MCP host (e.g., Cursor/Claude): call tools `rag.ingest`, `rag.query`, `rag.status`, `rag.clear`, `rag.export`.

### Maintenance
- Clear the index:
```bash
# via MCP tool input {"scope":"all"}
```
- Export JSONL:
```bash
# via MCP tool: rag.export(path="/tmp/rag_export.jsonl", limit=1000)
```

### Configuration
- Environment variables:
  - `RAG_MCP_CHROMA_DIR` (default: `~/Documents/rag-mcp/.chroma`)
  - `RAG_MCP_CHROMA_COLLECTION` (default: `rag_collection`)
  - `RAG_MCP_ENABLE_GENERATION` (0/1)
  - `RAG_MCP_USE_OLLAMA` (0/1)
  - `RAG_MCP_OLLAMA_MODEL` (e.g., `gemma2:2b`, `gemma3:270m`)
  - `RAG_MCP_TOP_K` (final context size)

## Tools
- `rag.ingest(paths?: string[], texts?: string[], tags?: string[])`
- `rag.query(query: string, k?: number)` → returns hits and, if enabled, an answer
- `rag.status()`
 - `rag.clear(scope: 'all'|'by_tag', tag_contains?)`
 - `rag.export(path: string, limit?: number)`

## Notes
- Storage: ChromaDB persistent collection
- Embeddings: `BAAI/bge-small-en-v1.5`
- Reranker: `BAAI/bge-reranker-base`
- LLM (local): prefer `gemma2:2b` via Ollama for quality; optionally use gemma-3-270m GGUF as documented

See `DESIGN.md` for full architecture and parameters.
