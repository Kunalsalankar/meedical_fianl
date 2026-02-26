from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

from config import RAW_DIR
from src.rag.pipeline import RagPipeline


def _init_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RagPipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title="Medicinal RAG (Chroma)", layout="wide")
    st.title("Medicinal RAG (Chroma)")

    _init_state()

    with st.sidebar:
        st.header("Data")
        st.write(f"Add files here:")
        st.code(str(RAW_DIR))
        st.write("Then run ingestion:")
        st.code("python scripts\\ingest.py")

        if st.button("Warm-up vector DB"):
            st.session_state.pipeline.ensure_ready()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    for s in m["sources"]:
                        st.write(f"- {s}")

    prompt = st.chat_input("Ask a clinical / medicinal question...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            answer = st.session_state.pipeline.answer(prompt)
        st.markdown(answer["answer"])
        if answer.get("sources"):
            with st.expander("Sources"):
                for s in answer["sources"]:
                    st.write(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer["answer"], "sources": answer.get("sources", [])})


if __name__ == "__main__":
    main()
