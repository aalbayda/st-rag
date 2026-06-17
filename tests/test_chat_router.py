
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY", "secret-or")
os.environ.setdefault("PINECONE_API_KEY", "secret-p")
os.environ.setdefault("PINECONE_INDEX_NAME", "rag-dense")


@pytest.fixture
def chat_client():
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def chat_client_with_mock():
    from app.main import app

    mock_fn = MagicMock()
    with patch("app.routers.chat.answer_question", mock_fn):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_fn


def _make_answer(abstained: bool = False, citations: list | None = None):
    from app.contracts import Answer

    if abstained:
        return Answer(
            answer="",
            reasoning="No relevant documents found.",
            citations=[],
            abstained=True,
        )
    else:
        from app.contracts import Citation, PdfLocator

        cit = Citation(
            id="[1]",
            file_id="file-abc",
            file_name="report.pdf",
            locator=PdfLocator(page=3),
            chunk_text="This is the source chunk text.",
        )
        return Answer(
            answer="The answer is X [1].",
            reasoning="Based on [1] which says X.",
            citations=[cit],
            abstained=False,
        )


def test_post_chat_happy_path_200(chat_client_with_mock):
    client, mock_fn = chat_client_with_mock
    mock_fn.return_value = _make_answer(abstained=False)

    response = client.post(
        "/chat", json={"question": "What is X?", "session_id": "sess-1"}
    )

    assert response.status_code == 200
    body = response.json()

    assert "answer" in body
    assert "reasoning" in body
    assert "citations" in body
    assert "abstained" in body

    assert isinstance(body["citations"], list)
    assert body["abstained"] is False
    assert len(body["citations"]) == 1


def test_post_chat_abstained_answer(chat_client_with_mock):
    client, mock_fn = chat_client_with_mock
    mock_fn.return_value = _make_answer(abstained=True)

    response = client.post(
        "/chat",
        json={"question": "What is the meaning of life?", "session_id": "sess-1"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["abstained"] is True
    assert body["citations"] == []


def test_post_chat_empty_question_422(chat_client):
    response = chat_client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_post_chat_missing_question_422(chat_client):
    response = chat_client.post("/chat", json={})
    assert response.status_code == 422


def test_post_chat_no_leak_on_exception(chat_client_with_mock):
    client, mock_fn = chat_client_with_mock
    mock_fn.side_effect = RuntimeError("internal boom")

    response = client.post(
        "/chat", json={"question": "What is X?", "session_id": "sess-1"}
    )

    assert response.status_code == 200
    body = response.json()

    assert "answer" in body
    assert "reasoning" in body
    assert "citations" in body
    assert "abstained" in body
    assert body["abstained"] is True
    assert body["citations"] == []

    assert "Traceback" not in response.text
    assert "secret-or" not in response.text
    assert "secret-p" not in response.text


@pytest.fixture
def db_chat_client(tmp_path, monkeypatch):
    import uuid
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch

    db_file = str(tmp_path / "test_chat_history.db")
    monkeypatch.setenv("DB_PATH", db_file)

    from app.config import get_settings
    from app.db.engine import get_engine, init_db

    get_settings.cache_clear()
    get_engine.cache_clear()
    init_db()

    mock_fn = MagicMock()
    with patch("app.routers.chat.answer_question", mock_fn):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_fn

    get_engine.cache_clear()
    get_settings.cache_clear()


def test_post_chat_passes_history_to_answer_question(db_chat_client):
    import uuid
    from datetime import datetime, timezone

    from sqlmodel import Session

    from app.contracts import Answer
    from app.db.engine import get_engine
    from app.db.models import ChatSession, Message

    client, mock_fn = db_chat_client
    mock_fn.return_value = _make_answer(abstained=False)

    session_id = str(uuid.uuid4())
    engine = get_engine()
    with Session(engine) as db:
        db.add(ChatSession(
            id=session_id,
            name="Test Session",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        db.add(Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content="prior user question",
            created_at="2026-01-01T00:00:00Z",
        ))
        db.add(Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content="prior assistant answer",
            created_at="2026-01-01T00:00:01Z",
        ))
        db.commit()

    response = client.post(
        "/chat", json={"question": "follow up question", "session_id": session_id}
    )

    assert response.status_code == 200
    call_kwargs = mock_fn.call_args[1]
    assert "history" in call_kwargs
    history = call_kwargs["history"]
    assert isinstance(history, list)
    roles = [h["role"] for h in history]
    contents = [h["content"] for h in history]
    assert "user" in roles
    assert "assistant" in roles
    assert "prior user question" in contents
    assert "prior assistant answer" in contents


def test_post_chat_empty_session_id_rejected(db_chat_client):
    client, mock_fn = db_chat_client
    mock_fn.return_value = _make_answer(abstained=False)

    response = client.post(
        "/chat", json={"question": "What is X?", "session_id": ""}
    )

    assert response.status_code == 422
    mock_fn.assert_not_called()
