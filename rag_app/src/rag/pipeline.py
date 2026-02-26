from __future__ import annotations

from typing import Dict, List

from src.rag.prompts import build_prompt
from src.retriever.retriever import Retriever
from src.vectorstore.store import ChromaVectorStore


class RagPipeline:
    """Minimal RAG pipeline.

    Currently uses retrieval + rule-based answer formatting.
    Hook point for a real LLM client later.
    """

    def __init__(self):
        self._store = ChromaVectorStore()
        self._retriever = Retriever(self._store)

    def ensure_ready(self) -> None:
        # Touch store to initialize persistent collection
        _ = self._store.count()

    def answer(self, question: str, k: int = 5) -> Dict[str, object]:
        contexts, sources = self._retriever.retrieve(question, k=k)

        if not contexts:
            return {
                "answer": "I don't have any indexed documents yet. Add files to data/raw and run ingestion (python scripts\\ingest.py).",
                "sources": [],
            }

        # Rule-based response: return the most relevant snippets.
        # Replace with LLM call later.
        joined = "\n\n".join([f"- {c.strip()}" for c in contexts[:3] if c.strip()])
        answer = (
            "Based on the indexed documents, here are the most relevant excerpts:\n\n"
            f"{joined}\n\n"
            "If you want a synthesized answer, connect an LLM in src/llm/client.py and call it from this pipeline."
        )

        return {"answer": answer, "sources": sources}
