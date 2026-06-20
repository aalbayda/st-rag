
import importlib


def _reload_settings():
    import app.config as cfg_module

    cfg_module.get_settings.cache_clear()
    importlib.reload(cfg_module)
    cfg_module.get_settings.cache_clear()
    return cfg_module


def test_imports_clean():
    from app.config import Settings, get_settings  # noqa: F401


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "my-test-index")

    import app.config as cfg_module
    cfg_module.get_settings.cache_clear()

    settings = cfg_module.Settings()
    assert settings.pinecone_index_name == "my-test-index"


def test_model_id_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or")
    monkeypatch.setenv("PINECONE_API_KEY", "test-p")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")

    import app.config as cfg_module
    cfg_module.get_settings.cache_clear()

    settings = cfg_module.Settings()
    assert settings.gen_model == "openai/gpt-oss-120b:free", (
        f"Expected gen_model='openai/gpt-oss-120b:free', got {settings.gen_model!r}"
    )
    assert settings.naming_model == "openai/gpt-oss-120b:free", (
        f"Expected naming_model='openai/gpt-oss-120b:free', got {settings.naming_model!r}"
    )
    assert settings.embedding_model == "openai/text-embedding-3-large", (
        "Expected embedding_model='openai/text-embedding-3-large', "
        f"got {settings.embedding_model!r}"
    )


def test_embedding_dimension_is_3072(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or")
    monkeypatch.setenv("PINECONE_API_KEY", "test-p")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")

    import app.config as cfg_module
    cfg_module.get_settings.cache_clear()

    settings = cfg_module.Settings()
    assert settings.embedding_dimension == 3072, (
        f"Expected embedding_dimension=3072, got {settings.embedding_dimension!r}"
    )


def test_secret_keys_masked_in_repr(monkeypatch):
    raw_openrouter = "test-openrouter-secret-key"
    raw_pinecone = "test-pinecone-secret-key"
    monkeypatch.setenv("OPENROUTER_API_KEY", raw_openrouter)
    monkeypatch.setenv("PINECONE_API_KEY", raw_pinecone)
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")

    import app.config as cfg_module
    cfg_module.get_settings.cache_clear()

    settings = cfg_module.Settings()
    settings_repr = repr(settings)
    settings_str = str(settings)

    assert raw_openrouter not in settings_repr, (
        f"Raw OpenRouter key must not appear in repr(settings); repr={settings_repr!r}"
    )
    assert raw_pinecone not in settings_repr, (
        f"Raw Pinecone key must not appear in repr(settings); repr={settings_repr!r}"
    )
    assert raw_openrouter not in settings_str, (
        "Raw OpenRouter key must not appear in str(settings)"
    )
    assert raw_pinecone not in settings_str, (
        "Raw Pinecone key must not appear in str(settings)"
    )


def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or")
    monkeypatch.setenv("PINECONE_API_KEY", "test-p")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "idx")

    import app.config as cfg_module
    cfg_module.get_settings.cache_clear()

    s1 = cfg_module.get_settings
    s2 = cfg_module.get_settings
    assert s1 is s2, "get_settings() must return the same cached instance"


def test_secret_str_typed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or")
    monkeypatch.setenv("PINECONE_API_KEY", "test-p")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag")

    from pydantic import SecretStr

    import app.config as cfg_module
    cfg_module.get_settings.cache_clear()

    settings = cfg_module.Settings()
    assert isinstance(settings.openrouter_api_key, SecretStr), (
        "openrouter_api_key must be SecretStr"
    )
    assert isinstance(settings.pinecone_api_key, SecretStr), (
        "pinecone_api_key must be SecretStr"
    )


def test_phase2_chunking_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or")
    monkeypatch.setenv("PINECONE_API_KEY", "test-p")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "rag-dense")

    import app.config as cfg_module
    cfg_module.get_settings.cache_clear()

    settings = cfg_module.Settings()

    assert settings.chunk_size == 512, f"Expected chunk_size=512, got {settings.chunk_size}"
    assert settings.chunk_overlap == 64, f"Expected chunk_overlap=64, got {settings.chunk_overlap}"
    assert settings.chunk_encoding == "cl100k_base", (
        f"Expected chunk_encoding='cl100k_base', got {settings.chunk_encoding!r}"
    )

    assert settings.embed_batch_inputs == 2048, (
        f"Expected embed_batch_inputs=2048, got {settings.embed_batch_inputs}"
    )
    assert settings.embed_batch_tokens == 300_000, (
        f"Expected embed_batch_tokens=300000, got {settings.embed_batch_tokens}"
    )

    assert settings.upsert_batch_size == 100, (
        f"Expected upsert_batch_size=100, got {settings.upsert_batch_size}"
    )

    assert settings.max_files == 5, f"Expected max_files=5, got {settings.max_files}"
    assert settings.max_file_bytes == 20971520, (
        f"Expected max_file_bytes=20971520 (20MB), got {settings.max_file_bytes}"
    )

    assert settings.embedding_model == "openai/text-embedding-3-large", (
        f"embedding_model must not change, got {settings.embedding_model!r}"
    )
    assert settings.embedding_dimension == 3072, (
        f"embedding_dimension must not change, got {settings.embedding_dimension}"
    )
