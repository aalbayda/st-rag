
from __future__ import annotations

from app.clients.openai_client import embed_texts
from app.clients.pinecone_client import hybrid_query, rerank_matches, sparse_encode_query
from app.config import get_settings


def assemble_context(
    matches: list[dict],
    per_chunk_chars: int,
) -> tuple[str, dict]:
    if not matches:
        return ("", {})

    id_to_meta: dict[str, dict] = {}
    blocks: list[str] = []

    for i, match in enumerate(matches, start=1):
        local_id = f"[{i}]"
        meta = match["metadata"]

        id_to_meta[local_id] = meta

        truncated_text = meta["chunk_text"][:per_chunk_chars]
        blocks.append(f"{local_id} {truncated_text}")

    context_block = "\n\n".join(blocks)
    return (context_block, id_to_meta)


def retrieve(question: str) -> tuple[str, dict]:
    settings = get_settings()

    dense_vec = embed_texts([question])[0]

    sparse_vec = sparse_encode_query(question)

    matches = hybrid_query(dense_vec, sparse_vec, settings.retrieval_alpha, settings.retrieval_candidate_k)

    top_matches = rerank_matches(question, matches, settings.retrieval_top_n)

    return assemble_context(top_matches, settings.context_per_chunk_chars)
