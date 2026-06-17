
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

_splitter: RecursiveCharacterTextSplitter | None = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        settings = get_settings()
        _splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=settings.chunk_encoding,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    return _splitter


def chunk_unit(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return _get_splitter().split_text(text)
