from __future__ import annotations

from typing import Iterable, List

from .cleaner import clean_text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Simple character-based chunker with overlap.

    chunk_size/overlap are in characters to keep dependencies minimal.
    """

    text = clean_text(text)
    if not text:
        return []

    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks
