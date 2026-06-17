
from __future__ import annotations

from functools import lru_cache

import openai

from app.config import get_settings
from app.contracts import ModelAnswer


@lru_cache(maxsize=1)
def get_openai_client() -> openai.OpenAI:
    settings = get_settings()
    default_headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        default_headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_title:
        default_headers["X-OpenRouter-Title"] = settings.openrouter_title

    return openai.OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key.get_secret_value(),
        default_headers=default_headers or None,
    )


def verify_models() -> dict[str, bool]:
    settings = get_settings()
    client = get_openai_client()

    required_models = [
        settings.gen_model,
        settings.naming_model,
        settings.embedding_model,
    ]

    available_ids: set[str] = {model.id for model in client.models.list()}

    return {model_id: (model_id in available_ids) for model_id in required_models}


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    client = get_openai_client()
    settings = get_settings()

    max_inputs = settings.embed_batch_inputs
    max_tokens = settings.embed_batch_tokens
    model = settings.embedding_model

    all_vectors: list[list[float]] = []

    i = 0
    while i < len(texts):
        batch: list[str] = []
        batch_token_estimate = 0

        while i < len(texts) and len(batch) < max_inputs:
            estimated_tokens = max(1, len(texts[i]) // 4)
            if batch and batch_token_estimate + estimated_tokens > max_tokens:
                break
            batch.append(texts[i])
            batch_token_estimate += estimated_tokens
            i += 1

        resp = client.embeddings.create(
            model=model,
            input=batch,
        )

        for datum in resp.data:
            all_vectors.append(datum.embedding)

    return all_vectors


def generate_answer(system_prompt: str, user_content: str) -> ModelAnswer | None:
    client = get_openai_client()
    settings = get_settings()

    try:
        completion = client.chat.completions.parse(
            model=settings.gen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=ModelAnswer,
        )
    except Exception:
        return None

    choice = completion.choices[0]

    if choice.message.refusal is not None:
        return None

    if choice.finish_reason == "length":
        return None

    return choice.message.parsed
