from __future__ import annotations

from typing import Dict, List, Tuple

from src.vectorstore.store import ChromaVectorStore


class Retriever:
    def __init__(self, store: ChromaVectorStore):
        self._store = store

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[str]]:
        res = self._store.query(query, k=k)

        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]

        sources: List[str] = []
        for md in metadatas:
            if isinstance(md, dict) and md.get("source"):
                sources.append(str(md.get("source")))

        return [str(d) for d in documents], sources
