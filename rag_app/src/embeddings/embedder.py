from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_CONFIG


class Embedder:
    def __init__(self, model_name: str | None = None):
        self._model = SentenceTransformer(model_name or EMBEDDING_CONFIG.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        return vecs

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]
