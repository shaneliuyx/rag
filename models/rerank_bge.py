from __future__ import annotations
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from functools import lru_cache
from config.settings import settings


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, docs: List[str], model_name: str | None = None, top_k: int | None = None) -> List[Tuple[int, float]]:
    model = _get_model(model_name or settings.reranker_model)
    pairs = [[query, d] for d in docs]
    scores = model.predict(pairs, convert_to_numpy=True)
    ranked = sorted(list(enumerate(scores.tolist())), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked
