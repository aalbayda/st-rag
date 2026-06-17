
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
        from app.clients.openai_client import get_openai_client

        get_openai_client.cache_clear()
    except (ImportError, AttributeError):
        pass


def _make_embed_response(texts: list[str], dim: int = 8) -> MagicMock:
    resp = MagicMock()
    resp.data = [_make_embedding(i, dim) for i in range(len(texts))]
    return resp


def _make_embedding(index: int, dim: int = 8) -> MagicMock:
    emb = MagicMock()
    emb.embedding = [float(index)] + [0.0] * (dim - 1)
    return emb


class TestEmbedTexts:


    def test_import_cleanly(self):
        from app.clients.openai_client import embed_texts  # noqa: F401


    def test_returns_one_vector_per_input(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        texts = ["a", "b", "c"]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embed_response(texts)

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = embed_texts(texts)

        assert len(result) == 3

    def test_output_order_matches_input_order(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        texts = ["first", "second", "third"]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embed_response(texts)

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = embed_texts(texts)

        assert result[0][0] == 0.0
        assert result[1][0] == 1.0
        assert result[2][0] == 2.0


    def test_batching_issues_multiple_calls_for_large_input(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        monkeypatch.setenv("EMBED_BATCH_INPUTS", "2")
        get_settings.cache_clear()

        texts = [f"text_{i}" for i in range(5)]

        def _side_effect(model, input, **kwargs):
            return _make_embed_response(input)

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = _side_effect

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = embed_texts(texts)

        assert mock_client.embeddings.create.call_count >= 3, (
            f"Expected >=3 create calls for 5 inputs with batch_size=2, "
            f"got {mock_client.embeddings.create.call_count}"
        )

    def test_batching_preserves_global_order(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        monkeypatch.setenv("EMBED_BATCH_INPUTS", "2")
        get_settings.cache_clear()

        n = 5
        texts = [f"text_{i}" for i in range(n)]

        call_counter = {"n": 0}

        def _side_effect(model, input, **kwargs):
            start = call_counter["n"]
            call_counter["n"] += len(input)
            resp = MagicMock()
            resp.data = [
                _make_embedding(start + j)
                for j in range(len(input))
            ]
            return resp

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = _side_effect

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = embed_texts(texts)

        assert len(result) == n
        for i in range(n):
            assert result[i][0] == float(i), (
                f"Position {i}: expected embedding index {i}, got {result[i][0]}"
            )


    def test_uses_configured_embedding_model(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        settings = get_settings()
        texts = ["hello"]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embed_response(texts)

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            embed_texts(texts)

        calls = mock_client.embeddings.create.call_args_list
        assert len(calls) >= 1
        for c in calls:
            kwargs = c.kwargs
            assert kwargs.get("model") == settings.embedding_model, (
                f"Expected model={settings.embedding_model!r}, got {kwargs.get('model')!r}"
            )

    def test_no_dimensions_kwarg_passed(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        texts = ["hello"]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embed_response(texts)

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            embed_texts(texts)

        for c in mock_client.embeddings.create.call_args_list:
            assert "dimensions" not in c.kwargs, (
                f"'dimensions' kwarg found in embeddings.create call: {c}"
            )


    def test_returns_list_of_lists(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        texts = ["a", "b"]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embed_response(texts)

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = embed_texts(texts)

        assert isinstance(result, list)
        for vec in result:
            assert isinstance(vec, list)

    def test_empty_input_returns_empty_list(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        mock_client = MagicMock()

        from app.clients.openai_client import embed_texts, get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = embed_texts([])

        assert result == []
        mock_client.embeddings.create.assert_not_called()


def _env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
    monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
    get_settings.cache_clear()


def _make_mock_client_for_generate(
    *,
    refusal: str | None = None,
    finish_reason: str = "stop",
    parsed=None,
    side_effect=None,
) -> MagicMock:
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.chat.completions.parse.side_effect = side_effect
    else:
        choice = SimpleNamespace(
            message=SimpleNamespace(refusal=refusal, parsed=parsed),
            finish_reason=finish_reason,
        )
        completion = SimpleNamespace(choices=[choice])
        mock_client.chat.completions.parse.return_value = completion
    return mock_client


class TestGenerateAnswer:

    def test_import_cleanly(self):
        from app.clients.openai_client import generate_answer  # noqa: F401

    def test_returns_model_answer_on_clean_stop(self, monkeypatch):
        _env(monkeypatch)
        from app.contracts import ModelAnswer

        expected = ModelAnswer(answer="42", reasoning="because", cited_ids=["[1]"])
        mock_client = _make_mock_client_for_generate(
            refusal=None, finish_reason="stop", parsed=expected
        )

        from app.clients.openai_client import generate_answer

        with patch(
            "app.clients.openai_client.get_openai_client", return_value=mock_client
        ):
            result = generate_answer("system prompt", "user question")

        assert result is expected, f"Expected the parsed ModelAnswer, got {result!r}"

    def test_returns_none_on_refusal(self, monkeypatch):
        _env(monkeypatch)
        mock_client = _make_mock_client_for_generate(
            refusal="I cannot answer that.", finish_reason="stop", parsed=None
        )

        from app.clients.openai_client import generate_answer

        with patch(
            "app.clients.openai_client.get_openai_client", return_value=mock_client
        ):
            result = generate_answer("system", "user")

        assert result is None, f"Expected None on refusal, got {result!r}"

    def test_returns_none_on_finish_reason_length(self, monkeypatch):
        _env(monkeypatch)
        mock_client = _make_mock_client_for_generate(
            refusal=None, finish_reason="length", parsed=None
        )

        from app.clients.openai_client import generate_answer

        with patch(
            "app.clients.openai_client.get_openai_client", return_value=mock_client
        ):
            result = generate_answer("system", "user")

        assert result is None, f"Expected None on finish_reason=='length', got {result!r}"

    def test_returns_none_on_parse_exception(self, monkeypatch):
        _env(monkeypatch)
        mock_client = _make_mock_client_for_generate(
            side_effect=RuntimeError("network error")
        )

        from app.clients.openai_client import generate_answer

        with patch(
            "app.clients.openai_client.get_openai_client", return_value=mock_client
        ):
            result = generate_answer("system", "user")

        assert result is None, f"Expected None on parse exception, got {result!r}"

    def test_no_exception_raised_on_parse_failure(self, monkeypatch):
        _env(monkeypatch)
        mock_client = _make_mock_client_for_generate(
            side_effect=Exception("unexpected SDK error")
        )

        from app.clients.openai_client import generate_answer

        with patch(
            "app.clients.openai_client.get_openai_client", return_value=mock_client
        ):
            generate_answer("sys", "usr")

    def test_uses_gen_model_from_settings(self, monkeypatch):
        _env(monkeypatch)
        settings = get_settings()
        from app.contracts import ModelAnswer

        expected = ModelAnswer(answer="x", reasoning="y")
        mock_client = _make_mock_client_for_generate(
            refusal=None, finish_reason="stop", parsed=expected
        )

        from app.clients.openai_client import generate_answer

        with patch(
            "app.clients.openai_client.get_openai_client", return_value=mock_client
        ):
            generate_answer("sys", "usr")

        call_kwargs = mock_client.chat.completions.parse.call_args.kwargs
        assert call_kwargs.get("model") == settings.gen_model, (
            f"Expected model={settings.gen_model!r}, got {call_kwargs.get('model')!r}"
        )

    def test_passes_response_format_model_answer(self, monkeypatch):
        _env(monkeypatch)
        from app.contracts import ModelAnswer

        expected = ModelAnswer(answer="x", reasoning="y")
        mock_client = _make_mock_client_for_generate(
            refusal=None, finish_reason="stop", parsed=expected
        )

        from app.clients.openai_client import generate_answer

        with patch(
            "app.clients.openai_client.get_openai_client", return_value=mock_client
        ):
            generate_answer("sys", "usr")

        call_kwargs = mock_client.chat.completions.parse.call_args.kwargs
        assert call_kwargs.get("response_format") is ModelAnswer, (
            f"Expected response_format=ModelAnswer, got {call_kwargs.get('response_format')!r}"
        )


class TestGenerateAnswerSourceGuard:

    def test_response_format_model_answer_in_source(self):
        from app.clients import openai_client

        source = inspect.getsource(openai_client.generate_answer)
        assert "response_format=ModelAnswer" in source, (
            "'response_format=ModelAnswer' not found in generate_answer source"
        )

    def test_no_retry_loop_in_source(self):
        from app.clients import openai_client

        source = inspect.getsource(openai_client.generate_answer)
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
        parse_count = code_only.count("chat.completions.parse")
        assert parse_count == 1, (
            f"chat.completions.parse appears {parse_count} times in generate_answer code lines "
            "expected exactly 1 (no retry loop)"
        )
