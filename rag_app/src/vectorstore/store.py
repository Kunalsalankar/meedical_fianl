from __future__ import annotations

from typing import Dict, List, Optional

import chromadb

from config import CHROMA_CONFIG
from src.embeddings.embedder import Embedder


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ):
        self._persist_directory = persist_directory or CHROMA_CONFIG.persist_directory
        self._collection_name = collection_name or CHROMA_CONFIG.collection_name

        self._client = chromadb.PersistentClient(path=self._persist_directory)
        self._collection = self._client.get_or_create_collection(name=self._collection_name)
        self._embedder = Embedder()

    def count(self) -> int:
        return int(self._collection.count())

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        if not texts:
            return

        embeddings = self._embedder.embed_texts(texts)
        ids = [f"chunk_{i}" for i in range(self.count(), self.count() + len(texts))]

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query: str, k: int = 5) -> Dict:
        q_emb = self._embedder.embed_query(query)
        res = self._collection.query(query_embeddings=[q_emb], n_results=int(k))
        return res
