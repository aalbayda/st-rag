
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY", "secret-or")
os.environ.setdefault("PINECONE_API_KEY", "secret-p")
os.environ.setdefault("PINECONE_INDEX_NAME", "rag-dense")


@pytest.fixture
def db_client(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_sessions.db")
    monkeypatch.setenv("DB_PATH", db_file)

    from app.config import get_settings
    from app.db.engine import get_engine, init_db

    get_settings.cache_clear()
    get_engine.cache_clear()
    init_db()

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    get_engine.cache_clear()
    get_settings.cache_clear()


def test_post_sessions_returns_201(db_client):
    r = db_client.post("/sessions")
    assert r.status_code == 201
    body = r.json()
    assert "id" in body
    assert "name" in body
    assert "created_at" in body


def test_post_sessions_creates_unique_ids(db_client):
    r1 = db_client.post("/sessions")
    r2 = db_client.post("/sessions")
    assert r1.json()["id"] != r2.json()["id"]


def test_post_sessions_name_is_null(db_client):
    r = db_client.post("/sessions")
    assert r.json()["name"] is None


def test_get_sessions_empty_returns_list(db_client):
    r = db_client.get("/sessions")
    assert r.status_code == 200
    assert r.json() == []


def test_get_sessions_lists_created_sessions(db_client):
    db_client.post("/sessions")
    db_client.post("/sessions")
    r = db_client.get("/sessions")
    assert r.status_code == 200
    sessions = r.json()
    assert len(sessions) == 2
    for s in sessions:
        assert "id" in s
        assert "name" in s
        assert "updated_at" in s


def test_get_sessions_ordered_by_updated_at_desc(db_client):
    r1 = db_client.post("/sessions")
    r2 = db_client.post("/sessions")
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]

    sessions = db_client.get("/sessions").json()
    ids_returned = [s["id"] for s in sessions]
    assert set(ids_returned) == {id1, id2}


def test_get_messages_empty_returns_list(db_client):
    sess = db_client.post("/sessions").json()
    r = db_client.get(f"/sessions/{sess['id']}/messages")
    assert r.status_code == 200
    assert r.json() == []


def test_get_messages_unknown_session_returns_empty_list(db_client):
    r = db_client.get("/sessions/nonexistent-id/messages")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_patch_session_name_endpoint_removed(db_client):
    sess = db_client.post("/sessions").json()
    r = db_client.patch(f"/sessions/{sess['id']}/name", json={"name": "X"})
    assert r.status_code == 404


def test_delete_session_returns_ok_true(db_client):
    sess = db_client.post("/sessions").json()
    r = db_client.delete(f"/sessions/{sess['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["deleted"] is True


def test_delete_session_removes_session_from_list(db_client):
    sess = db_client.post("/sessions").json()
    db_client.delete(f"/sessions/{sess['id']}")
    sessions = db_client.get("/sessions").json()
    ids = [s["id"] for s in sessions]
    assert sess["id"] not in ids


def test_delete_session_cascades_to_messages(db_client):
    from app.db.engine import get_engine
    from app.db.models import Message
    from sqlmodel import Session

    sess = db_client.post("/sessions").json()
    session_id = sess["id"]

    import uuid
    engine = get_engine()
    with Session(engine) as db:
        msg = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content="hello",
        )
        db.add(msg)
        db.commit()

    msgs_before = db_client.get(f"/sessions/{session_id}/messages").json()
    assert len(msgs_before) == 1

    db_client.delete(f"/sessions/{session_id}")

    msgs_after = db_client.get(f"/sessions/{session_id}/messages").json()
    assert msgs_after == []


def test_delete_nonexistent_session_returns_ok_false(db_client):
    r = db_client.delete("/sessions/does-not-exist")
    assert r.status_code == 200
    assert r.json()["ok"] is False
