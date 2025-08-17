from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=False).tolist()
