
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "secret-or")
os.environ.setdefault("PINECONE_API_KEY", "secret-p")
os.environ.setdefault("PINECONE_INDEX_NAME", "rag-dense")

from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_caches():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_match(chunk_text: str, file_id: str = "f1", kind: str = "pdf", page: int = 1) -> dict:
    meta = {
        "file_id": file_id,
        "file_name": "doc.pdf",
        "chunk_text": chunk_text,
        "kind": kind,
        "page": page,
    }
    return {"id": f"{file_id}#0", "score": 0.9, "metadata": meta}


def _make_docx_match(chunk_text: str, file_id: str = "f2") -> dict:
    meta = {
        "file_id": file_id,
        "file_name": "doc.docx",
        "chunk_text": chunk_text,
        "kind": "docx",
        "section": "Introduction",
        "paragraph_index": 5,
    }
    return {"id": f"{file_id}#0", "score": 0.8, "metadata": meta}


class TestAssembleContext:

    def test_import_cleanly(self):
        from app.services.retrieval import assemble_context  # noqa: F401

    def test_single_match_returns_id_1(self):
        from app.services.retrieval import assemble_context

        match = _make_match("Hello world")
        context_block, id_to_meta = assemble_context([match], per_chunk_chars=2000)

        assert "[1]" in context_block
        assert "Hello world" in context_block
        assert "[1]" in id_to_meta

    def test_two_matches_produce_sequential_ids(self):
        from app.services.retrieval import assemble_context

        m1 = _make_match("First chunk", file_id="f1")
        m2 = _make_match("Second chunk", file_id="f2")
        context_block, id_to_meta = assemble_context([m1, m2], per_chunk_chars=2000)

        assert "[1]" in context_block
        assert "[2]" in context_block
        assert "[1]" in id_to_meta
        assert "[2]" in id_to_meta
        assert "First chunk" in context_block
        assert "Second chunk" in context_block

    def test_id_to_meta_keys_are_bracket_n(self):
        from app.services.retrieval import assemble_context

        m1 = _make_match("A", file_id="f1")
        m2 = _make_match("B", file_id="f2")
        _, id_to_meta = assemble_context([m1, m2], per_chunk_chars=2000)

        assert set(id_to_meta.keys()) == {"[1]", "[2]"}
        for key in id_to_meta:
            assert "#" not in key

    def test_id_to_meta_values_are_full_metadata(self):
        from app.services.retrieval import assemble_context

        match = _make_match("Full text here")
        _, id_to_meta = assemble_context([match], per_chunk_chars=2000)

        meta = id_to_meta["[1]"]
        assert meta["file_id"] == "f1"
        assert meta["chunk_text"] == "Full text here"
        assert meta["kind"] == "pdf"

    def test_no_drift_between_context_and_id_to_meta(self):
        from app.services.retrieval import assemble_context

        m1 = _make_match("Alpha")
        m2 = _make_match("Beta", file_id="f2")
        context_block, id_to_meta = assemble_context([m1, m2], per_chunk_chars=2000)

        for key in id_to_meta:
            assert key in context_block

        assert len(id_to_meta) == 2

    def test_chunks_separated_by_double_newline(self):
        from app.services.retrieval import assemble_context

        m1 = _make_match("Alpha")
        m2 = _make_match("Beta", file_id="f2")
        context_block, _ = assemble_context([m1, m2], per_chunk_chars=2000)

        assert "\n\n" in context_block


class TestAssembleContextTruncation:

    def test_long_chunk_truncated_in_context_block(self):
        from app.services.retrieval import assemble_context

        long_text = "x" * 500
        match = _make_match(long_text)
        context_block, _ = assemble_context([match], per_chunk_chars=100)

        assert "x" * 500 not in context_block
        assert "x" * 100 in context_block

    def test_full_chunk_text_in_id_to_meta(self):
        from app.services.retrieval import assemble_context

        long_text = "y" * 500
        match = _make_match(long_text)
        _, id_to_meta = assemble_context([match], per_chunk_chars=100)

        assert id_to_meta["[1]"]["chunk_text"] == long_text

    def test_short_chunk_not_truncated(self):
        from app.services.retrieval import assemble_context

        short_text = "Short text"
        match = _make_match(short_text)
        context_block, _ = assemble_context([match], per_chunk_chars=2000)

        assert short_text in context_block


class TestAssembleContextEmpty:

    def test_empty_matches_returns_empty_tuple(self):
        from app.services.retrieval import assemble_context

        context_block, id_to_meta = assemble_context([], per_chunk_chars=2000)

        assert context_block == ""
        assert id_to_meta == {}


class TestRetrieve:

    def test_import_cleanly(self):
        from app.services.retrieval import retrieve  # noqa: F401

    def test_retrieve_calls_embed_texts_once(self, monkeypatch):
        from app.services.retrieval import retrieve

        mock_embed = MagicMock(return_value=[[0.1] * 3072])
        mock_sparse = MagicMock(return_value={"indices": [1], "values": [0.5]})
        mock_hybrid = MagicMock(return_value=[])
        mock_rerank = MagicMock(return_value=[])

        with (
            patch("app.services.retrieval.embed_texts", mock_embed),
            patch("app.services.retrieval.sparse_encode_query", mock_sparse),
            patch("app.services.retrieval.hybrid_query", mock_hybrid),
            patch("app.services.retrieval.rerank_matches", mock_rerank),
        ):
            retrieve("What is X?")

        mock_embed.assert_called_once_with(["What is X?"])

    def test_retrieve_passes_first_embedding_to_hybrid_query(self, monkeypatch):
        from app.services.retrieval import retrieve

        vector = [0.5] * 3072
        mock_embed = MagicMock(return_value=[vector])
        mock_sparse = MagicMock(return_value={"indices": [1], "values": [0.5]})
        mock_hybrid = MagicMock(return_value=[])
        mock_rerank = MagicMock(return_value=[])

        with (
            patch("app.services.retrieval.embed_texts", mock_embed),
            patch("app.services.retrieval.sparse_encode_query", mock_sparse),
            patch("app.services.retrieval.hybrid_query", mock_hybrid),
            patch("app.services.retrieval.rerank_matches", mock_rerank),
        ):
            retrieve("Any question")

        call_args = mock_hybrid.call_args
        assert call_args[0][0] == vector

    def test_retrieve_passes_retrieval_candidate_k(self, monkeypatch):
        from app.services.retrieval import retrieve

        mock_embed = MagicMock(return_value=[[0.1] * 3072])
        mock_sparse = MagicMock(return_value={"indices": [1], "values": [0.5]})
        mock_hybrid = MagicMock(return_value=[])
        mock_rerank = MagicMock(return_value=[])

        with (
            patch("app.services.retrieval.embed_texts", mock_embed),
            patch("app.services.retrieval.sparse_encode_query", mock_sparse),
            patch("app.services.retrieval.hybrid_query", mock_hybrid),
            patch("app.services.retrieval.rerank_matches", mock_rerank),
        ):
            retrieve("question")

        call_args = mock_hybrid.call_args
        settings = get_settings()
        assert call_args[0][3] == settings.retrieval_candidate_k

    def test_retrieve_returns_context_and_meta(self, monkeypatch):
        from app.services.retrieval import retrieve

        match = _make_match("Important text")
        mock_embed = MagicMock(return_value=[[0.1] * 3072])
        mock_sparse = MagicMock(return_value={"indices": [1], "values": [0.5]})
        mock_hybrid = MagicMock(return_value=[match])
        mock_rerank = MagicMock(return_value=[match])

        with (
            patch("app.services.retrieval.embed_texts", mock_embed),
            patch("app.services.retrieval.sparse_encode_query", mock_sparse),
            patch("app.services.retrieval.hybrid_query", mock_hybrid),
            patch("app.services.retrieval.rerank_matches", mock_rerank),
        ):
            context_block, id_to_meta = retrieve("question")

        assert "[1]" in context_block
        assert "[1]" in id_to_meta
        assert id_to_meta["[1]"]["chunk_text"] == "Important text"

    def test_retrieve_zero_matches_returns_empty_tuple(self, monkeypatch):
        from app.services.retrieval import retrieve

        mock_embed = MagicMock(return_value=[[0.1] * 3072])
        mock_sparse = MagicMock(return_value={"indices": [], "values": []})
        mock_hybrid = MagicMock(return_value=[])
        mock_rerank = MagicMock(return_value=[])

        with (
            patch("app.services.retrieval.embed_texts", mock_embed),
            patch("app.services.retrieval.sparse_encode_query", mock_sparse),
            patch("app.services.retrieval.hybrid_query", mock_hybrid),
            patch("app.services.retrieval.rerank_matches", mock_rerank),
        ):
            context_block, id_to_meta = retrieve("question")

        assert context_block == ""
        assert id_to_meta == {}


class TestSourceGuard:

    def test_no_import_pinecone_in_source(self):
        import os

        module_dir = os.path.dirname(os.path.abspath(__file__))
        retrieval_path = os.path.join(
            os.path.dirname(module_dir), "app", "services", "retrieval.py"
        )
        with open(retrieval_path) as f:
            lines = f.readlines()

        code_lines = [
            l.rstrip()
            for l in lines
            if not l.lstrip().startswith("#") and not l.lstrip().startswith('"""')
            and not l.lstrip().startswith("'''")
        ]
        source = "\n".join(code_lines)
        assert "import pinecone" not in source

    def test_no_import_openai_in_source(self):
        import os

        module_dir = os.path.dirname(os.path.abspath(__file__))
        retrieval_path = os.path.join(
            os.path.dirname(module_dir), "app", "services", "retrieval.py"
        )
        with open(retrieval_path) as f:
            lines = f.readlines()

        code_lines = [
            l.rstrip()
            for l in lines
            if not l.lstrip().startswith("#") and not l.lstrip().startswith('"""')
            and not l.lstrip().startswith("'''")
        ]
        source = "\n".join(code_lines)
        assert "import openai" not in source
