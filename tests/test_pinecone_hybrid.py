
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_caches():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    try:
        from app.clients.pinecone_client import get_pinecone

        get_pinecone.cache_clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def mock_pc():
    return MagicMock()


@pytest.fixture
def fake_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pin-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    get_settings.cache_clear()
    return get_settings()


def _sparse_embed_result(indices: list[int], values: list[float]):
    item = SimpleNamespace(sparse_indices=indices, sparse_values=values)
    return [item]


def _rerank_result(id_order: list[str]):
    items = []
    for rank, doc_id in enumerate(id_order):
        item = SimpleNamespace(
            score=1.0 - rank * 0.1,
            document={"id": doc_id, "chunk_text": "text"},
        )
        items.append(item)
    return SimpleNamespace(results=items)


def _make_matches(ids: list[str]) -> list[dict]:
    return [
        {"id": id_, "score": float(i), "metadata": {"chunk_text": f"text {id_}"}}
        for i, id_ in enumerate(ids)
    ]


def test_settings_has_retrieval_alpha(fake_settings):
    assert hasattr(fake_settings, "retrieval_alpha")
    assert fake_settings.retrieval_alpha == 0.75


def test_settings_has_retrieval_candidate_k(fake_settings):
    assert hasattr(fake_settings, "retrieval_candidate_k")
    assert fake_settings.retrieval_candidate_k == 40


def test_settings_has_retrieval_top_n(fake_settings):
    assert hasattr(fake_settings, "retrieval_top_n")
    assert fake_settings.retrieval_top_n == 5


def test_settings_has_rerank_model(fake_settings):
    assert hasattr(fake_settings, "rerank_model")
    assert fake_settings.rerank_model == "bge-reranker-v2-m3"


def test_sparse_encode_query_returns_indices_and_values(monkeypatch, fake_settings, mock_pc):
    mock_pc.inference.embed.return_value = _sparse_embed_result([1, 5, 10], [0.1, 0.5, 0.4])

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        pinecone_client.get_pinecone.cache_clear = lambda: None
        result = pinecone_client.sparse_encode_query("hello world")

    assert "indices" in result
    assert "values" in result
    assert result["indices"] == [1, 5, 10]
    assert result["values"] == [0.1, 0.5, 0.4]


def test_sparse_encode_query_calls_inference_with_query_input_type(monkeypatch, fake_settings, mock_pc):
    mock_pc.inference.embed.return_value = _sparse_embed_result([1], [0.9])

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        pinecone_client.sparse_encode_query("test question")

    call_kwargs = mock_pc.inference.embed.call_args
    assert call_kwargs[0][0] == "pinecone-sparse-english-v0" or call_kwargs[1].get("model") == "pinecone-sparse-english-v0"
    params = call_kwargs[1].get("parameters") or (call_kwargs[0][2] if len(call_kwargs[0]) > 2 else {})
    assert params.get("input_type") == "query"


def test_sparse_encode_passages_returns_list_of_dicts(monkeypatch, fake_settings, mock_pc):
    mock_pc.inference.embed.return_value = _sparse_embed_result([2, 4], [0.3, 0.7]) + _sparse_embed_result([1], [1.0])

    mock_pc.inference.embed.return_value = [
        SimpleNamespace(sparse_indices=[2, 4], sparse_values=[0.3, 0.7]),
        SimpleNamespace(sparse_indices=[1], sparse_values=[1.0]),
    ]

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        results = pinecone_client.sparse_encode_passages(["chunk a", "chunk b"])

    assert len(results) == 2
    assert results[0]["indices"] == [2, 4]
    assert results[1]["indices"] == [1]


def test_sparse_encode_passages_calls_inference_with_passage_input_type(monkeypatch, fake_settings, mock_pc):
    mock_pc.inference.embed.return_value = [
        SimpleNamespace(sparse_indices=[1], sparse_values=[0.9]),
    ]

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        pinecone_client.sparse_encode_passages(["text"])

    call_kwargs = mock_pc.inference.embed.call_args
    params = call_kwargs[1].get("parameters") or {}
    assert params.get("input_type") == "passage"


def test_hybrid_query_alpha_1_zeros_sparse(monkeypatch, fake_settings, mock_pc):
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_index.query.return_value = SimpleNamespace(matches=[])

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        dense_vec = [0.1, 0.2, 0.3]
        sparse_vec = {"indices": [0, 1], "values": [0.5, 0.5]}
        pinecone_client.hybrid_query(dense_vec, sparse_vec, alpha=1.0, top_k=5)

    call_kwargs = mock_index.query.call_args[1]
    sparse_passed = call_kwargs.get("sparse_vector", {})
    assert all(v == 0.0 for v in sparse_passed.get("values", [1])), "sparse values should be 0 at alpha=1.0"


def test_hybrid_query_alpha_0_zeros_dense(monkeypatch, fake_settings, mock_pc):
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_index.query.return_value = SimpleNamespace(matches=[])

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        dense_vec = [0.1, 0.2, 0.3]
        sparse_vec = {"indices": [0, 1], "values": [0.5, 0.5]}
        pinecone_client.hybrid_query(dense_vec, sparse_vec, alpha=0.0, top_k=5)

    call_kwargs = mock_index.query.call_args[1]
    vector_passed = call_kwargs.get("vector", [1])
    assert all(v == 0.0 for v in vector_passed), "dense values should be 0 at alpha=0.0"


def test_hybrid_query_returns_list_of_dicts(monkeypatch, fake_settings, mock_pc):
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_match = SimpleNamespace(id="abc#0", score=0.9, metadata={"chunk_text": "hello"})
    mock_index.query.return_value = SimpleNamespace(matches=[mock_match])

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        result = pinecone_client.hybrid_query([0.1], {"indices": [0], "values": [0.5]}, alpha=0.5, top_k=5)

    assert len(result) == 1
    assert result[0]["id"] == "abc#0"
    assert result[0]["score"] == 0.9
    assert result[0]["metadata"] == {"chunk_text": "hello"}


def test_rerank_matches_empty_input_returns_empty(monkeypatch, fake_settings, mock_pc):
    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        result = pinecone_client.rerank_matches("question", [], top_n=5)

    assert result == []
    mock_pc.inference.rerank.assert_not_called()


def test_rerank_matches_reorders_by_score(monkeypatch, fake_settings, mock_pc):
    matches = _make_matches(["a", "b", "c"])

    mock_pc.inference.rerank.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(score=0.9, document={"id": "c", "chunk_text": "text c"}),
            SimpleNamespace(score=0.7, document={"id": "a", "chunk_text": "text a"}),
        ]
    )

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        result = pinecone_client.rerank_matches("question", matches, top_n=2)

    assert len(result) == 2
    assert result[0]["id"] == "c"
    assert result[1]["id"] == "a"


def test_rerank_matches_result_has_id_score_metadata_keys(monkeypatch, fake_settings, mock_pc):
    matches = _make_matches(["x"])
    mock_pc.inference.rerank.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(score=0.8, document={"id": "x", "chunk_text": "text x"}),
        ]
    )

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        result = pinecone_client.rerank_matches("q", matches, top_n=1)

    assert set(result[0].keys()) == {"id", "score", "metadata"}


def test_rerank_matches_graceful_fallback_on_exception(monkeypatch, fake_settings, mock_pc):
    matches = _make_matches(["a", "b", "c", "d"])
    mock_pc.inference.rerank.side_effect = RuntimeError("reranker unavailable")

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        result = pinecone_client.rerank_matches("question", matches, top_n=2)

    assert len(result) == 2
    assert result[0]["id"] == "a"
    assert result[1]["id"] == "b"


def test_rerank_matches_handles_namespace_document(monkeypatch, fake_settings, mock_pc):
    matches = _make_matches(["p", "q"])
    doc_ns = SimpleNamespace(id="p", chunk_text="text p")
    mock_pc.inference.rerank.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(score=0.95, document=doc_ns),
        ]
    )

    with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
        from app.clients import pinecone_client

        result = pinecone_client.rerank_matches("q", matches, top_n=1)

    assert result[0]["id"] == "p"
