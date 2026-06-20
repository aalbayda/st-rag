
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clear_client_caches():
    yield
    try:
        from app.clients import openai_client
        openai_client.get_openai_client.cache_clear()
    except (ImportError, AttributeError):
        pass
    try:
        from app.clients import pinecone_client
        pinecone_client.get_pinecone.cache_clear()
    except (ImportError, AttributeError):
        pass


class TestOpenAIClient:

    def test_import_cleanly(self):
        from app.clients.openai_client import get_openai_client, verify_models  # noqa: F401

    def test_verify_models_returns_all_three_keys(self, monkeypatch):
        fake_model_ids = [
            "openai/gpt-oss-120b:free",
            "openai/text-embedding-3-large",
        ]

        mock_model = MagicMock()
        mock_model.id = None

        mock_models_page = MagicMock()
        mock_models_page.__iter__ = lambda self: iter(
            [_make_model(mid) for mid in fake_model_ids]
        )

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_models_page

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        from app.clients.openai_client import get_openai_client, verify_models

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = verify_models()

        settings = get_settings()
        assert settings.gen_model in result, "gen_model key missing"
        assert settings.naming_model in result, "naming_model key missing"
        assert settings.embedding_model in result, "embedding_model key missing"

    def test_verify_models_present_flag_true_when_in_list(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        settings = get_settings()
        all_three = [settings.gen_model, settings.naming_model, settings.embedding_model]

        mock_page = MagicMock()
        mock_page.__iter__ = lambda self: iter([_make_model(mid) for mid in all_three])

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_page

        from app.clients.openai_client import get_openai_client, verify_models

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = verify_models()

        assert result[settings.gen_model] is True
        assert result[settings.naming_model] is True
        assert result[settings.embedding_model] is True

    def test_verify_models_present_flag_false_when_absent(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        settings = get_settings()
        partial_list = [settings.gen_model]

        mock_page = MagicMock()
        mock_page.__iter__ = lambda self: iter([_make_model(mid) for mid in partial_list])

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_page

        from app.clients.openai_client import get_openai_client, verify_models

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.get_openai_client", return_value=mock_client):
            result = verify_models()

        assert result[settings.gen_model] is True
        assert result[settings.naming_model] is True
        assert result[settings.embedding_model] is False

    def test_get_openai_client_uses_openrouter_base_url_and_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test")
        monkeypatch.setenv("OPENROUTER_TITLE", "RAG Agent Test")
        get_settings.cache_clear()

        from app.clients.openai_client import get_openai_client

        get_openai_client.cache_clear()

        with patch("app.clients.openai_client.openai.OpenAI") as mock_openai:
            get_openai_client()

        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key-openrouter",
            default_headers={
                "HTTP-Referer": "https://example.test",
                "X-OpenRouter-Title": "RAG Agent Test",
            },
        )

    def test_no_legacy_api_usage(self):
        with open("/home/boberoo/Desktop/rag/app/clients/openai_client.py") as f:
            source = f.read()
        assert "ChatCompletion" not in source, "Legacy ChatCompletion API found"

    def test_key_read_via_secret_str(self):
        with open("/home/boberoo/Desktop/rag/app/clients/openai_client.py") as f:
            source = f.read()
        assert "get_secret_value" in source, "API key not read via SecretStr.get_secret_value()"


class TestPineconeClient:

    def test_import_cleanly(self):
        from app.clients.pinecone_client import (  # noqa: F401
            ensure_index,
            get_index,
            get_pinecone,
            ping_index,
        )

    def test_ensure_index_creates_with_correct_args_when_absent(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        mock_pc = _build_mock_pinecone(index_exists=False)

        from app.clients.pinecone_client import ensure_index, get_pinecone

        get_pinecone.cache_clear()

        with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
            ensure_index()

        mock_pc.create_index.assert_called_once()
        call_kwargs = mock_pc.create_index.call_args

        all_args = {**call_kwargs.kwargs}
        if call_kwargs.args:
            pass

        call_str = str(call_kwargs)
        assert "3072" in call_str, "dimension=3072 not passed to create_index"
        assert "dotproduct" in call_str, "metric=dotproduct not passed to create_index"

    def test_ensure_index_noop_when_exists(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        mock_pc = _build_mock_pinecone(index_exists=True)

        from app.clients.pinecone_client import ensure_index, get_pinecone

        get_pinecone.cache_clear()

        with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
            ensure_index()

        mock_pc.create_index.assert_not_called()

    def test_ping_index_returns_expected_shape(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-openrouter")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key-pinecone")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")
        get_settings.cache_clear()

        mock_pc = _build_mock_pinecone(index_exists=True)

        from app.clients.pinecone_client import get_pinecone, ping_index

        get_pinecone.cache_clear()

        with patch("app.clients.pinecone_client.get_pinecone", return_value=mock_pc):
            result = ping_index()

        assert "reachable" in result
        assert "dimension" in result
        assert "metric" in result
        assert result["reachable"] is True
        assert result["dimension"] == 3072
        assert result["metric"] == "dotproduct"

    def test_dotproduct_in_source(self):
        with open("/home/boberoo/Desktop/rag/app/clients/pinecone_client.py") as f:
            source = f.read()
        assert "dotproduct" in source, "dotproduct metric not found in pinecone_client.py"

    def test_no_pinecone_client_package(self):
        with open("/home/boberoo/Desktop/rag/app/clients/pinecone_client.py") as f:
            source = f.read()
        assert "pinecone-client" not in source, "Deprecated pinecone-client reference found"

    def test_dimension_lock_in_source(self):
        with open("/home/boberoo/Desktop/rag/app/clients/pinecone_client.py") as f:
            source = f.read()
        assert "3072" in source or "embedding_dimension" in source, \
            "dimension lock (3072 or embedding_dimension) not found in pinecone_client.py"


def _make_model(model_id: str) -> MagicMock:
    m = MagicMock()
    m.id = model_id
    return m


def _build_mock_pinecone(*, index_exists: bool) -> MagicMock:
    mock_pc = MagicMock()

    mock_pc.has_index.return_value = index_exists

    desc = MagicMock()
    desc.dimension = 3072
    desc.metric = "dotproduct"
    mock_pc.describe_index.return_value = desc

    mock_pc.create_index.return_value = None

    return mock_pc
