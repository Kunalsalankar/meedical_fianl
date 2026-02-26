# RAG App (Chroma)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add documents
Put PDFs/DOCX/TXT/CSV into:
- `data/raw/`

## Ingest
```powershell
python scripts\ingest.py
```

## Run UI
```powershell
streamlit run app.py
```
