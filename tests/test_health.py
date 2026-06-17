
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


os.environ.setdefault("OPENROUTER_API_KEY", "secret-or")
os.environ.setdefault("PINECONE_API_KEY", "secret-p")
os.environ.setdefault("PINECONE_INDEX_NAME", "rag-dense")

from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


HEALTHY_PING = {"reachable": True, "dimension": 3072, "metric": "dotproduct"}
HEALTHY_MODELS = {
    "openai/gpt-5.5": True,
    "openai/gpt-5.4-nano": True,
    "openai/text-embedding-3-large": True,
}


def _mock_engine_ok() -> MagicMock:
    conn = MagicMock()
    conn.execute = MagicMock(return_value=None)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    eng = MagicMock()
    eng.connect = MagicMock(return_value=ctx)
    return eng


def test_health_all_healthy_returns_200():
    with (
        patch("app.routers.health.ping_index", return_value=HEALTHY_PING),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    assert response.status_code == 200


def test_health_all_healthy_status_ok():
    with (
        patch("app.routers.health.ping_index", return_value=HEALTHY_PING),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"


def test_health_all_healthy_openrouter_has_all_three_model_keys():
    with (
        patch("app.routers.health.ping_index", return_value=HEALTHY_PING),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    openrouter_sub = response.json()["openrouter"]
    assert "openai/gpt-5.5" in openrouter_sub
    assert "openai/gpt-5.4-nano" in openrouter_sub
    assert "openai/text-embedding-3-large" in openrouter_sub


def test_health_all_healthy_pinecone_dimension_and_metric():
    with (
        patch("app.routers.health.ping_index", return_value=HEALTHY_PING),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    pinecone_sub = response.json()["pinecone"]
    assert pinecone_sub["dimension"] == 3072
    assert pinecone_sub["metric"] == "dotproduct"


def test_health_all_healthy_database_ok():
    with (
        patch("app.routers.health.ping_index", return_value=HEALTHY_PING),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    db_sub = response.json()["database"]
    assert db_sub["ok"] is True


def test_health_pinecone_raises_no_500():
    with (
        patch("app.routers.health.ping_index", side_effect=RuntimeError("boom")),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    assert response.status_code != 500


def test_health_pinecone_raises_overall_status_not_ok():
    with (
        patch("app.routers.health.ping_index", side_effect=RuntimeError("boom")),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    assert response.json()["status"] != "ok"


def test_health_pinecone_raises_no_openrouter_key_in_body():
    openrouter_key = os.environ["OPENROUTER_API_KEY"]
    with (
        patch("app.routers.health.ping_index", side_effect=RuntimeError("boom")),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    assert openrouter_key not in response.text


def test_health_pinecone_raises_no_pinecone_key_in_body():
    pinecone_key = os.environ["PINECONE_API_KEY"]
    with (
        patch("app.routers.health.ping_index", side_effect=RuntimeError("boom")),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    assert pinecone_key not in response.text


def test_health_pinecone_raises_no_traceback_in_body():
    with (
        patch("app.routers.health.ping_index", side_effect=RuntimeError("boom")),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    assert "Traceback" not in response.text


def test_health_openrouter_missing_model_status_not_ok():
    partial_models = {
        "openai/gpt-5.5": False,
        "openai/gpt-5.4-nano": True,
        "openai/text-embedding-3-large": True,
    }
    with (
        patch("app.routers.health.ping_index", return_value=HEALTHY_PING),
        patch("app.routers.health.verify_models", return_value=partial_models),
        patch("app.routers.health.get_engine", return_value=_mock_engine_ok()),
    ):
        response = client.get("/health")
    body = response.json()
    assert body["status"] != "ok"
    assert body["openrouter"]["openai/gpt-5.5"] is False


def test_health_db_raises_status_not_ok():
    bad_engine = MagicMock()
    bad_engine.connect.side_effect = RuntimeError("no db")
    with (
        patch("app.routers.health.ping_index", return_value=HEALTHY_PING),
        patch("app.routers.health.verify_models", return_value=HEALTHY_MODELS),
        patch("app.routers.health.get_engine", return_value=bad_engine),
    ):
        response = client.get("/health")
    assert response.json()["status"] != "ok"
    assert response.json()["database"]["ok"] is False
