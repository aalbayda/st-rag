
from __future__ import annotations

import inspect
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


def _make_mock_index(*, list_ids: list[str] | None = None) -> MagicMock:
    if list_ids is None:
        list_ids = []

    mock_index = MagicMock()
    page = SimpleNamespace(vectors=[SimpleNamespace(id=v) for v in list_ids])
    mock_index.list.return_value = iter([page]) if list_ids else iter([])
    mock_index.upsert.return_value = None
    mock_index.delete.return_value = None
    return mock_index


def _env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
    monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
    get_settings.cache_clear()


class TestUpsertChunks:

    def test_import_cleanly(self):
        from app.clients.pinecone_client import upsert_chunks  # noqa: F401

    def test_upsert_called_with_vectors_and_batch_size(self, monkeypatch):
        _env(monkeypatch)
        vectors = [
            {"id": "file1#0", "values": [0.1, 0.2], "metadata": {"text": "chunk0"}},
            {"id": "file1#1", "values": [0.3, 0.4], "metadata": {"text": "chunk1"}},
        ]
        mock_index = _make_mock_index()

        from app.clients.pinecone_client import upsert_chunks

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            upsert_chunks(vectors)

        mock_index.upsert.assert_called_once()
        call_kwargs = mock_index.upsert.call_args.kwargs
        assert "vectors" in call_kwargs, "'vectors' kwarg not passed to index.upsert"
        assert "batch_size" in call_kwargs, "'batch_size' kwarg not passed to index.upsert"

    def test_upsert_uses_settings_batch_size(self, monkeypatch):
        _env(monkeypatch)
        settings = get_settings()

        vectors = [{"id": "f#0", "values": [0.1], "metadata": {}}]
        mock_index = _make_mock_index()

        from app.clients.pinecone_client import upsert_chunks

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            upsert_chunks(vectors)

        call_kwargs = mock_index.upsert.call_args.kwargs
        assert call_kwargs["batch_size"] == settings.upsert_batch_size

    def test_upsert_passes_all_vectors(self, monkeypatch):
        _env(monkeypatch)
        vectors = [
            {"id": f"file1#{n}", "values": [float(n)], "metadata": {}}
            for n in range(5)
        ]
        mock_index = _make_mock_index()

        from app.clients.pinecone_client import upsert_chunks

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            upsert_chunks(vectors)

        call_kwargs = mock_index.upsert.call_args.kwargs
        assert call_kwargs["vectors"] == vectors

    def test_empty_vectors_calls_upsert_noop(self, monkeypatch):
        _env(monkeypatch)
        mock_index = _make_mock_index()

        from app.clients.pinecone_client import upsert_chunks

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            upsert_chunks([])


class TestDeleteByFile:

    def test_import_cleanly(self):
        from app.clients.pinecone_client import delete_by_file  # noqa: F401

    def test_list_called_with_prefix(self, monkeypatch):
        _env(monkeypatch)
        file_id = "doc123"
        mock_index = _make_mock_index(list_ids=["doc123#0", "doc123#1"])

        from app.clients.pinecone_client import delete_by_file

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            delete_by_file(file_id)

        mock_index.list.assert_called_once_with(prefix=f"{file_id}#")

    def test_delete_called_with_ids_not_filter(self, monkeypatch):
        _env(monkeypatch)
        file_id = "doc123"
        vector_ids = ["doc123#0", "doc123#1", "doc123#2"]
        mock_index = _make_mock_index(list_ids=vector_ids)

        from app.clients.pinecone_client import delete_by_file

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            delete_by_file(file_id)

        mock_index.delete.assert_called_once()
        call_kwargs = mock_index.delete.call_args.kwargs
        assert "ids" in call_kwargs, "'ids' kwarg not passed to index.delete()"
        assert "filter" not in call_kwargs, (
            "'filter' kwarg found in index.delete(). Serverless rejects filter delete!"
        )
        assert set(call_kwargs["ids"]) == set(vector_ids)

    def test_delete_noop_when_no_matching_ids(self, monkeypatch):
        _env(monkeypatch)
        mock_index = _make_mock_index(list_ids=[])

        from app.clients.pinecone_client import delete_by_file

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            delete_by_file("nonexistent-file")

        mock_index.delete.assert_not_called()

    def test_never_calls_delete_filter(self, monkeypatch):
        _env(monkeypatch)
        vector_ids = ["myfile#0", "myfile#1"]
        mock_index = _make_mock_index(list_ids=vector_ids)

        from app.clients.pinecone_client import delete_by_file

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            delete_by_file("myfile")

        for c in mock_index.delete.call_args_list:
            assert "filter" not in c.kwargs, (
                f"delete called with filter= kwarg: {c}"
            )


class TestSourceGuard:

    def test_prefix_in_delete_by_file_source(self):
        from app.clients import pinecone_client

        source = inspect.getsource(pinecone_client.delete_by_file)
        assert "prefix=" in source, (
            "'prefix=' not found in delete_by_file source. ID-prefix delete pattern missing"
        )

    def test_filter_not_in_delete_by_file_source(self):
        from app.clients import pinecone_client

        source = inspect.getsource(pinecone_client.delete_by_file)
        assert "delete(filter=" not in source, (
            "'delete(filter=' found in delete_by_file source. Serverless rejects filter delete!"
        )

    def test_module_level_no_delete_filter_grep(self):
        import os

        module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app", "clients", "pinecone_client.py",
        )
        with open(module_path) as f:
            lines = f.readlines()

        offending = [
            (i + 1, line.rstrip())
            for i, line in enumerate(lines)
            if not line.lstrip().startswith("#") and "delete(filter=" in line
        ]
        assert not offending, (
            f"'delete(filter=' found on lines: {offending}"
        )


def _make_query_response(matches: list) -> SimpleNamespace:
    return SimpleNamespace(matches=matches)


class TestQueryFunction:

    def test_import_cleanly(self):
        from app.clients.pinecone_client import query  # noqa: F401

    def test_returns_ordered_dicts_with_two_matches(self, monkeypatch):
        _env(monkeypatch)
        meta0 = {"file_id": "f1", "chunk_text": "chunk zero"}
        meta1 = {"file_id": "f1", "chunk_text": "chunk one"}
        matches = [
            SimpleNamespace(id="f1#0", score=0.95, metadata=meta0),
            SimpleNamespace(id="f1#1", score=0.80, metadata=meta1),
        ]
        mock_index = _make_mock_index()
        mock_index.query.return_value = _make_query_response(matches)

        from app.clients.pinecone_client import query

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            result = query([0.1, 0.2], top_k=2)

        assert len(result) == 2
        assert result[0]["id"] == "f1#0"
        assert result[0]["score"] == 0.95
        assert result[0]["metadata"] == meta0
        assert result[1]["id"] == "f1#1"
        assert result[1]["score"] == 0.80
        assert result[1]["metadata"] == meta1

    def test_returns_plain_dicts_not_sdk_objects(self, monkeypatch):
        _env(monkeypatch)
        matches = [
            SimpleNamespace(id="f1#0", score=0.9, metadata={"k": "v"}),
        ]
        mock_index = _make_mock_index()
        mock_index.query.return_value = _make_query_response(matches)

        from app.clients.pinecone_client import query

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            result = query([0.1], top_k=1)

        assert isinstance(result[0], dict), "query must return plain dicts, not SDK objects"
        assert isinstance(result[0]["metadata"], dict), "metadata must be a plain dict"

    def test_empty_matches_returns_empty_list(self, monkeypatch):
        _env(monkeypatch)
        mock_index = _make_mock_index()
        mock_index.query.return_value = _make_query_response([])

        from app.clients.pinecone_client import query

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            result = query([0.1, 0.2], top_k=5)

        assert result == [], f"Expected [] on empty matches, got {result!r}"

    def test_no_exception_on_empty_matches(self, monkeypatch):
        _env(monkeypatch)
        mock_index = _make_mock_index()
        mock_index.query.return_value = _make_query_response([])

        from app.clients.pinecone_client import query

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            query([0.0] * 3072, top_k=10)

    def test_query_called_with_include_metadata_true(self, monkeypatch):
        _env(monkeypatch)
        mock_index = _make_mock_index()
        mock_index.query.return_value = _make_query_response([])

        from app.clients.pinecone_client import query

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            query([0.1], top_k=3)

        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs.get("include_metadata") is True, (
            f"include_metadata=True not passed; call kwargs: {call_kwargs}"
        )

    def test_query_does_not_pass_namespace(self, monkeypatch):
        _env(monkeypatch)
        mock_index = _make_mock_index()
        mock_index.query.return_value = _make_query_response([])

        from app.clients.pinecone_client import query

        with patch("app.clients.pinecone_client.get_index", return_value=mock_index):
            query([0.1], top_k=3)

        call_kwargs = mock_index.query.call_args.kwargs
        assert "namespace" not in call_kwargs, (
            f"namespace= kwarg found in index.query call. Omit it to read default namespace; "
            f"call kwargs: {call_kwargs}"
        )


class TestQuerySourceGuard:

    def test_include_metadata_in_query_source(self):
        from app.clients import pinecone_client

        source = inspect.getsource(pinecone_client.query)
        assert "include_metadata" in source, (
            "'include_metadata' not found in query source"
        )

    def test_namespace_kwarg_not_in_query_source(self):
        from app.clients import pinecone_client

        source = inspect.getsource(pinecone_client.query)
        lines = source.splitlines()
        in_docstring = False
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith("#"):
                continue
            code_lines.append(line)
        code_only = "\n".join(code_lines)
        assert "namespace=" not in code_only, (
            "'namespace=' kwarg found in query code (not comments/docstring). "
            "omit it so retrieval reads the default namespace where upsert_chunks wrote."
        )
