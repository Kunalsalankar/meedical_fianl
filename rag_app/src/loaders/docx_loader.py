from __future__ import annotations

from pathlib import Path

import docx


def load_docx(path: Path) -> str:
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs if p.text).strip()
