
from __future__ import annotations


def test_models_imports():
    from app.db.models import ChatSession, File, Message  # noqa: F401


def test_utcnow_is_fixed_width_and_chronologically_sortable():
    from datetime import datetime, timezone
    from unittest.mock import patch

    import app.db.models as models

    base = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)

    with patch.object(models, "datetime") as mock_dt:
        mock_dt.now.return_value = base
        ts_zero = models._utcnow()
        mock_dt.now.return_value = later
        ts_micros = models._utcnow()

    assert len(ts_zero) == len(ts_micros)
    assert ts_zero < ts_micros


def test_file_has_expected_columns():
    from app.db.models import File

    fields = File.model_fields
    assert "id" in fields
    assert "name" in fields
    assert "byte_size" in fields
    assert "page_count" in fields
    assert "chunk_count" in fields
    assert "status" in fields
    assert "error" in fields
    assert "created_at" in fields


def test_file_has_stage_column():
    from app.db.models import File

    fields = File.model_fields
    assert "stage" in fields, "File must have a 'stage' column for granular lifecycle tracking"


def test_file_stage_defaults_to_none():
    from app.db.models import File

    f = File(id="f-stage-test", name="test.pdf", status="pending")
    assert f.stage is None, f"Expected stage=None by default, got {f.stage!r}"


def test_file_status_values_documented():
    from app.db.models import File

    fields = File.model_fields
    assert "status" in fields, "Stable status column must remain on File model"

    for status_val in ("pending", "ready", "failed", "deleting"):
        f = File(id=f"f-{status_val}", name="test.pdf", status=status_val)
        assert f.status == status_val, f"status='{status_val}' must round-trip"


def test_file_stage_and_status_round_trip_in_db(tmp_path):
    import os

    from sqlmodel import Session

    from app.db.engine import get_engine, init_db
    from app.db.models import File

    db_file = tmp_path / "stage_test.db"
    os.environ["DB_PATH"] = str(db_file)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-openrouter")
    os.environ.setdefault("PINECONE_API_KEY", "test-key-pinecone")
    os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")

    from app.config import get_settings
    from app.db.engine import get_engine as _ge
    get_settings.cache_clear()
    _ge.cache_clear()

    init_db()
    engine = get_engine()

    with Session(engine) as session:
        f = File(
            id="file-stage-001",
            name="doc.pdf",
            byte_size=512,
            status="pending",
            stage="queued",
            created_at="2026-01-01T00:00:00",
        )
        session.add(f)
        session.commit()

    with Session(engine) as session:
        read_f = session.get(File, "file-stage-001")

    assert read_f is not None
    assert read_f.status == "pending", f"Expected status='pending', got {read_f.status!r}"
    assert read_f.stage == "queued", f"Expected stage='queued', got {read_f.stage!r}"

    get_settings.cache_clear()
    _ge.cache_clear()
    del os.environ["DB_PATH"]


def test_fresh_db_has_stage_column_via_pragma(tmp_path):
    import os

    from sqlmodel import text

    from app.db.engine import get_engine, init_db

    db_file = tmp_path / "pragma_test.db"
    os.environ["DB_PATH"] = str(db_file)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-openrouter")
    os.environ.setdefault("PINECONE_API_KEY", "test-key-pinecone")
    os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")

    from app.config import get_settings
    from app.db.engine import get_engine as _ge
    get_settings.cache_clear()
    _ge.cache_clear()

    init_db()
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info('files')")).fetchall()
        column_names = {row[1] for row in rows}

    assert "stage" in column_names, (
        f"'stage' column missing from files table after init_db; got columns: {column_names}"
    )
    assert "status" in column_names, (
        f"'status' column must remain in files table; got columns: {column_names}"
    )

    get_settings.cache_clear()
    _ge.cache_clear()
    del os.environ["DB_PATH"]


def test_chat_session_name_is_optional():
    from app.db.models import ChatSession

    session = ChatSession(id="s1", created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
    assert session.name is None


def test_message_has_citations_and_reasoning():
    from app.db.models import Message

    fields = Message.model_fields
    assert "citations" in fields, "Message must have a citations column"
    assert "reasoning" in fields, "Message must have a reasoning column"


def test_message_citations_accepts_none():
    from app.db.models import Message

    msg = Message(
        id="m1",
        session_id="s1",
        role="user",
        content="Hello",
        created_at="2026-01-01T00:00:00",
    )
    assert msg.citations is None


def test_message_citations_accepts_list():
    import json

    from app.db.models import Message

    citations_payload = [
        {
            "id": "[1]",
            "file_id": "f1",
            "file_name": "report.pdf",
            "locator": {"kind": "pdf", "page": 5},
            "chunk_text": "Some relevant chunk text.",
        }
    ]
    msg = Message(
        id="m2",
        session_id="s1",
        role="assistant",
        content="The answer is ...",
        reasoning="Because ...",
        citations=json.dumps(citations_payload),
        created_at="2026-01-01T00:00:01",
    )
    assert msg.citations is not None
    loaded = json.loads(msg.citations)
    assert loaded[0]["id"] == "[1]"
    assert loaded[0]["chunk_text"] == "Some relevant chunk text."


def test_engine_imports():
    from app.db.engine import get_engine, get_session, init_db  # noqa: F401


def _set_db_env(db_file: str) -> None:
    import os

    os.environ["DB_PATH"] = str(db_file)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-openrouter")
    os.environ.setdefault("PINECONE_API_KEY", "test-key-pinecone")
    os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")


def _clear_caches() -> None:
    from app.config import get_settings
    from app.db.engine import get_engine

    get_settings.cache_clear()
    get_engine.cache_clear()


def test_init_db_creates_tables_and_sets_wal(tmp_path):
    import os

    from sqlmodel import text

    from app.db.engine import get_engine, init_db

    db_file = tmp_path / "test.db"
    _set_db_env(str(db_file))
    _clear_caches()

    init_db()

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode")).fetchone()
        assert result[0].lower() == "wal", f"Expected WAL, got {result[0]}"

    _clear_caches()
    del os.environ["DB_PATH"]


def test_init_db_creates_all_three_tables(tmp_path):
    import os

    from sqlmodel import text

    from app.db.engine import get_engine, init_db

    db_file = tmp_path / "tables_test.db"
    _set_db_env(str(db_file))
    _clear_caches()

    init_db()

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).fetchall()
        table_names = {r[0] for r in rows}

    assert "files" in table_names, f"Expected 'files' table, got {table_names}"
    assert "chat_sessions" in table_names, f"Expected 'chat_sessions' table, got {table_names}"
    assert "messages" in table_names, f"Expected 'messages' table, got {table_names}"

    _clear_caches()
    del os.environ["DB_PATH"]


def test_engine_round_trip_write_read(tmp_path):
    import json
    import os

    from sqlmodel import Session, select

    from app.db.engine import get_engine, init_db
    from app.db.models import ChatSession, File, Message

    db_file = tmp_path / "roundtrip.db"
    _set_db_env(str(db_file))
    _clear_caches()

    init_db()
    engine = get_engine()

    citations_payload = json.dumps(
        [
            {
                "id": "[1]",
                "file_id": "file-001",
                "file_name": "report.pdf",
                "locator": {"kind": "pdf", "page": 12},
                "chunk_text": "The quarterly revenue was $4.2M.",
            }
        ]
    )

    with Session(engine) as session:
        file_row = File(
            id="file-001",
            name="report.pdf",
            byte_size=1024,
            page_count=20,
            chunk_count=40,
            status="ready",
            created_at="2026-01-01T00:00:00",
        )
        chat_session = ChatSession(
            id="session-001",
            name=None,
            created_at="2026-01-01T00:01:00",
            updated_at="2026-01-01T00:01:00",
        )
        message = Message(
            id="msg-001",
            session_id="session-001",
            role="assistant",
            content="The quarterly revenue was $4.2M [1].",
            reasoning="Found evidence in report.pdf page 12.",
            citations=citations_payload,
            created_at="2026-01-01T00:01:01",
        )
        session.add(file_row)
        session.add(chat_session)
        session.add(message)
        session.commit()

    with Session(engine) as session:
        read_file = session.get(File, "file-001")
        read_session = session.get(ChatSession, "session-001")
        read_msg = session.exec(select(Message).where(Message.id == "msg-001")).first()

    assert read_file is not None
    assert read_file.name == "report.pdf"
    assert read_file.status == "ready"

    assert read_session is not None
    assert read_session.name is None

    assert read_msg is not None
    assert read_msg.role == "assistant"
    assert read_msg.reasoning is not None and "report.pdf" in read_msg.reasoning

    loaded_citations = json.loads(read_msg.citations)
    assert len(loaded_citations) == 1
    assert loaded_citations[0]["id"] == "[1]"
    assert loaded_citations[0]["chunk_text"] == "The quarterly revenue was $4.2M."
    assert loaded_citations[0]["locator"]["page"] == 12

    _clear_caches()
    del os.environ["DB_PATH"]
