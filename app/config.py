
from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: SecretStr

    pinecone_api_key: SecretStr

    pinecone_index_name: str

    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    openrouter_http_referer: str | None = None

    openrouter_title: str | None = None

    gen_model: str = "openai/gpt-5.5"

    naming_model: str = "openai/gpt-5.4-nano"

    embedding_model: str = "openai/text-embedding-3-large"

    embedding_dimension: int = 3072

    db_path: str = "app.db"

    db_journal_mode: str = "WAL"

    chunk_size: int = 512

    chunk_overlap: int = 64

    chunk_encoding: str = "cl100k_base"

    embed_batch_inputs: int = 2048

    embed_batch_tokens: int = 300_000

    upsert_batch_size: int = 100

    max_files: int = 5

    max_file_bytes: int = 20 * 1024 * 1024

    api_base_url: str = "http://localhost:8000"

    retrieval_top_k: int = 5

    context_per_chunk_chars: int = 2000

    retrieval_alpha: float = 0.75

    retrieval_candidate_k: int = 40

    retrieval_top_n: int = 5

    rerank_model: str = "bge-reranker-v2-m3"

    chat_history_turns: int = 6


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
