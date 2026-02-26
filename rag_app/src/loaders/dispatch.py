from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .pdf_loader import load_pdf
from .docx_loader import load_docx
from .text_loader import load_text
from .csv_loader import load_csv


def load_documents_from_path(path: Path) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not path.exists():
        return docs

    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()

        if suffix == ".pdf":
            text = load_pdf(p)
            docs.append({"text": text, "source": str(p.name), "doc_type": "pdf"})
        elif suffix == ".docx":
            text = load_docx(p)
            docs.append({"text": text, "source": str(p.name), "doc_type": "docx"})
        elif suffix in {".txt", ".md"}:
            text = load_text(p)
            docs.append({"text": text, "source": str(p.name), "doc_type": "text"})
        elif suffix == ".csv":
            text = load_csv(p)
            docs.append({"text": text, "source": str(p.name), "doc_type": "csv"})

    return [d for d in docs if d.get("text")]
