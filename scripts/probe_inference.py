
from __future__ import annotations

import sys

from app.clients.pinecone_client import get_pinecone


def main() -> None:
    pc = get_pinecone()

    try:
        result = pc.inference.embed(
            "pinecone-sparse-english-v0",
            inputs=["hello world"],
            parameters={"input_type": "query", "truncate": "END"},
        )
        indices = result[0].sparse_indices
        values = result[0].sparse_values
        if indices and values:
            print("SPARSE_OK")
        else:
            print("SPARSE_FAIL: sparse_indices or sparse_values empty in response")
    except Exception as exc:
        print(f"SPARSE_FAIL: {exc}")

    primary_model = "bge-reranker-v2-m3"
    fallback_model = "pinecone-rerank-v0"

    try:
        result = pc.inference.rerank(
            model=primary_model,
            query="test",
            documents=[{"id": "t1", "chunk_text": "hello world"}],
            top_n=1,
            rank_fields=["chunk_text"],
            return_documents=True,
            parameters={"truncate": "END"},
        )
        print(f"RERANK_OK model={primary_model}")
    except Exception as exc:
        err_str = str(exc)
        print(f"RERANK_FAIL: {err_str}")

        if "not found" in err_str.lower() or "invalid model" in err_str.lower() or "model" in err_str.lower():
            try:
                result = pc.inference.rerank(
                    model=fallback_model,
                    query="test",
                    documents=[{"id": "t1", "chunk_text": "hello world"}],
                    top_n=1,
                    rank_fields=["chunk_text"],
                    return_documents=True,
                    parameters={"truncate": "END"},
                )
                print(f"RERANK_FALLBACK_OK model={fallback_model}")
            except Exception as exc2:
                print(f"RERANK_FALLBACK_FAIL: {exc2}")


if __name__ == "__main__":
    main
