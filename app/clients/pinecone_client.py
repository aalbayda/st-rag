
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pinecone

from app.config import get_settings


@lru_cache(maxsize=1)
def get_pinecone() -> pinecone.Pinecone:
    settings = get_settings()
    return pinecone.Pinecone(api_key=settings.pinecone_api_key.get_secret_value())


def get_index() -> Any:
    settings = get_settings()
    pc = get_pinecone()
    return pc.Index(settings.pinecone_index_name)


def ensure_index() -> Any:
    settings = get_settings()
    pc = get_pinecone()

    index_name = settings.pinecone_index_name

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dimension,
            metric="dotproduct",
            spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.describe_index(index_name)


def ping_index() -> dict[str, Any]:
    settings = get_settings()
    pc = get_pinecone()

    try:
        desc = pc.describe_index(settings.pinecone_index_name)
        return {
            "reachable": True,
            "dimension": desc.dimension,
            "metric": str(desc.metric),
        }
    except Exception:
        return {
            "reachable": False,
            "dimension": 0,
            "metric": "",
        }


def upsert_chunks(vectors: list[dict]) -> None:
    index = get_index()
    settings = get_settings()
    index.upsert(vectors=vectors, batch_size=settings.upsert_batch_size)


def delete_by_file(file_id: str) -> None:
    index = get_index()
    ids = [
        item.id
        for page in index.list(prefix=f"{file_id}#")
        for item in page.vectors
        if item.id is not None
    ]
    if ids:
        index.delete(ids=ids)


def query(vector: list[float], top_k: int) -> list[dict]:
    index = get_index()
    resp = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {"id": m.id, "score": m.score, "metadata": dict(m.metadata or {})}
        for m in resp.matches
    ]


def sparse_encode_query(question: str) -> dict:
    pc = get_pinecone()
    result = pc.inference.embed(
        "pinecone-sparse-english-v0",
        inputs=[question],
        parameters={"input_type": "query", "truncate": "END"},
    )
    return {"indices": result[0].sparse_indices, "values": result[0].sparse_values}


def sparse_encode_passages(texts: list[str]) -> list[dict]:
    pc = get_pinecone()
    result = pc.inference.embed(
        "pinecone-sparse-english-v0",
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"},
    )
    return [{"indices": item.sparse_indices, "values": item.sparse_values} for item in result]


def hybrid_query(
    dense_vec: list[float],
    sparse_vec: dict,
    alpha: float,
    top_k: int,
) -> list[dict]:
    index = get_index()

    scaled_dense = [v * alpha for v in dense_vec]
    sparse_weight = 1.0 - alpha
    scaled_sparse = {
        "indices": sparse_vec["indices"],
        "values": [v * sparse_weight for v in sparse_vec["values"]],
    }

    resp = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )
    return [
        {"id": m.id, "score": m.score, "metadata": dict(m.metadata or {})}
        for m in resp.matches
    ]


def rerank_matches(question: str, matches: list[dict], top_n: int) -> list[dict]:
    if not matches:
        return []

    pc = get_pinecone()
    settings = get_settings()
    id_to_match = {m["id"]: m for m in matches}

    try:
        rerank_resp = pc.inference.rerank(
            model=settings.rerank_model,
            query=question,
            documents=[
                {"id": m["id"], "chunk_text": m["metadata"].get("chunk_text", "")}
                for m in matches
            ],
            top_n=top_n,
            rank_fields=["chunk_text"],
            return_documents=True,
            parameters={"truncate": "END"},
        )

        reranked: list[dict] = []
        for item in rerank_resp.results:
            doc_id = (
                item.document["id"]
                if isinstance(item.document, dict)
                else item.document.id
            )
            original = id_to_match.get(doc_id)
            if original is not None:
                reranked.append(original)

        return reranked

    except Exception:
        return matches[:top_n]
