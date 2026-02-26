from __future__ import annotations

from pathlib import Path

from config import PROCESSED_DIR, RAW_DIR
from src.preprocessing.chunker import chunk_text
from src.loaders.dispatch import load_documents_from_path
from src.vectorstore.store import ChromaVectorStore


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents_from_path(RAW_DIR)
    if not docs:
        print(f"No documents found in: {RAW_DIR}")
        return

    chunks = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size=900, overlap=150):
            chunks.append(
                {
                    "text": ch,
                    "metadata": {
                        "source": d.get("source", "unknown"),
                        "doc_type": d.get("doc_type", "unknown"),
                    },
                }
            )

    store = ChromaVectorStore()
    store.add_texts([c["text"] for c in chunks], [c["metadata"] for c in chunks])

    print(f"Loaded {len(docs)} docs")
    print(f"Indexed {len(chunks)} chunks into Chroma")


if __name__ == "__main__":
    main()
