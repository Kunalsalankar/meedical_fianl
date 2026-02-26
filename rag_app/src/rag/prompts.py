from __future__ import annotations


def build_prompt(question: str, contexts: list[str]) -> str:
    ctx = "\n\n---\n\n".join(contexts)
    return (
        "You are a clinical assistant. Use ONLY the provided context to answer. "
        "If the context is insufficient, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
