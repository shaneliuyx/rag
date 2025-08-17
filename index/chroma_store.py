from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.utils import embedding_functions
from config.settings import settings


class ChromaStore:
    def __init__(self, persist_dir: Optional[str] = None, collection: Optional[str] = None, embedding_model: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection or settings.chroma_collection
        self.embedding_model_name = embedding_model or settings.embedding_model
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, ids: List[str], texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> int:
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas or [{} for _ in texts])
        return len(ids)

    def query(self, query_texts: List[str], n_results: int) -> Dict[str, Any]:
        return self.collection.query(query_texts=query_texts, n_results=n_results)

    def count(self) -> int:
        return self.collection.count()

    def get_all(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        return self.collection.get(limit=limit, offset=offset)

    def clear_all(self) -> None:
        # Drop and recreate the collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def delete_where_tag_contains(self, tag_substring: str) -> int:
        # Rely on $contains over our CSV string of tags
        self.collection.delete(where={"tags": {"$contains": tag_substring}})
        return self.count()
