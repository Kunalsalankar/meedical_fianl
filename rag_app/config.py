from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ChromaConfig:
    collection_name: str = "medicinal_rag"
    persist_directory: str = str(VECTORSTORE_DIR)


EMBEDDING_CONFIG = EmbeddingConfig()
CHROMA_CONFIG = ChromaConfig()

# For LLM integration later (OpenAI/Azure/local). Keep empty for now.
DEFAULT_LLM_PROVIDER = "rule_based"
