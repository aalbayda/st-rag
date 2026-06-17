
from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY", "secret-or")
os.environ.setdefault("PINECONE_API_KEY", "secret-p")
os.environ.setdefault("PINECONE_INDEX_NAME", "rag-dense")

PDF_MAGIC = b"%PDF-" + b"fake pdf content" * 10
ZIP_MAGIC = b"PK\x03\x04" + b"fake docx content" * 10


@pytest.fixture
def db_client(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)

    from app.config import get_settings
    from app.db.engine import get_engine, init_db

    get_settings.cache_clear()
    get_engine.cache_clear()
    init_db()

    from app.main import app

    with (
        patch("app.routers.files.ingest_file"),
        patch("app.routers.files.delete_by_file"),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture
def db_client_with_spy(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_spy.db")
    monkeypatch.setenv("DB_PATH", db_file)

    from app.config import get_settings
    from app.db.engine import get_engine, init_db

    get_settings.cache_clear()
    get_engine.cache_clear()
    init_db()

    from app.main import app

    ingest_mock = MagicMock()
    delete_mock = MagicMock()

    with (
        patch("app.routers.files.ingest_file", ingest_mock),
        patch("app.routers.files.delete_by_file", delete_mock),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, ingest_mock, delete_mock


def _make_pdf_upload(name: str = "doc.pdf", size: int = 0) -> tuple:
    content = PDF_MAGIC if size == 0 else PDF_MAGIC[:size]
    return (name, io.BytesIO(content), "application/pdf")


def _make_docx_upload(name: str = "doc.docx") -> tuple:
    content = ZIP_MAGIC
    return (name, io.BytesIO(content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


def _make_txt_upload(name: str = "doc.txt") -> tuple:
    content = b"This is a plain text file"
    return (name, io.BytesIO(content), "text/plain")


def _post_files(client, file_tuples):
    files = [("files", ft) for ft in file_tuples]
    return client.post("/files", files=files)


def test_post_files_too_many_rejected(db_client):
    uploads = [_make_pdf_upload(f"doc{i}.pdf") for i in range(6)]
    response = _post_files(db_client, uploads)

    assert response.status_code == 202
    body = response.json()
    results = body["results"]
    assert len(results) == 6
    for r in results:
        assert r["accepted"] is False
        assert "5" in r["reason"] or "limit" in r["reason"].lower()


def test_post_files_spoofed_type_rejected(db_client):
    bad_file = ("bad.pdf", io.BytesIO(b"This is not a PDF"), "application/pdf")
    response = _post_files(db_client, [bad_file])

    assert response.status_code == 202
    body = response.json()
    results = body["results"]
    assert len(results) == 1
    assert results[0]["accepted"] is False
    reason = results[0]["reason"]
    assert reason and len(reason) > 0
    assert "Traceback" not in response.text
    assert "secret-or" not in response.text
    assert "secret-p" not in response.text


def test_post_files_oversized_rejected(tmp_path, monkeypatch):
    db_file = str(tmp_path / "oversized_test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("MAX_FILE_BYTES", "10")

    from app.config import get_settings
    from app.db.engine import get_engine, init_db

    get_settings.cache_clear()
    get_engine.cache_clear()
    init_db()

    from app.main import app

    with (
        patch("app.routers.files.ingest_file"),
        patch("app.routers.files.delete_by_file"),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            big = PDF_MAGIC + b"x" * 100
            response = client.post(
                "/files",
                files=[("files", ("big.pdf", io.BytesIO(big), "application/pdf"))],
            )

    assert response.status_code == 202
    body = response.json()
    results = body["results"]
    assert results[0]["accepted"] is False
    assert "Traceback" not in response.text


def test_post_valid_pdf_accepted_202(db_client_with_spy):
    client, ingest_mock, delete_mock = db_client_with_spy
    response = _post_files(client, [_make_pdf_upload("report.pdf")])

    assert response.status_code == 202
    body = response.json()
    results = body["results"]
    assert len(results) == 1
    assert results[0]["accepted"] is True
    assert results[0]["name"] == "report.pdf"
    assert "file_id" in results[0]


def test_post_valid_pdf_creates_file_row(db_client_with_spy, tmp_path, monkeypatch):
    client, ingest_mock, delete_mock = db_client_with_spy

    _post_files(client, [_make_pdf_upload("report.pdf")])

    from sqlmodel import Session, select

    from app.db.engine import get_engine
    from app.db.models import File

    with Session(get_engine()) as session:
        rows = session.exec(select(File)).all()

    assert len(rows) == 1
    assert rows[0].name == "report.pdf"
    assert rows[0].status == "pending"
    assert rows[0].stage == "queued"


def test_post_reupload_calls_delete_by_file(db_client_with_spy):
    client, ingest_mock, delete_mock = db_client_with_spy

    _post_files(client, [_make_pdf_upload("report.pdf")])
    _post_files(client, [_make_pdf_upload("report.pdf")])

    assert delete_mock.call_count >= 1


def test_get_files_empty_returns_empty_list(db_client):
    response = db_client.get("/files")
    assert response.status_code == 200
    body = response.json()
    assert body == []


def test_get_files_returns_uploaded_files(db_client_with_spy):
    client, ingest_mock, delete_mock = db_client_with_spy

    _post_files(client, [_make_pdf_upload("report.pdf")])

    response = client.get("/files")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    file_info = body[0]
    assert file_info["name"] == "report.pdf"
    assert "status" in file_info
    assert "stage" in file_info


def test_no_secret_leakage_on_rejection(db_client):
    bad_file = ("bad.pdf", io.BytesIO(b"not a pdf"), "application/pdf")
    response = _post_files(db_client, [bad_file])

    assert "Traceback" not in response.text
    assert "secret-or" not in response.text
    assert "secret-p" not in response.text


def test_no_secret_leakage_on_get(db_client):
    response = db_client.get("/files")
    assert "Traceback" not in response.text
    assert "secret-or" not in response.text
    assert "secret-p" not in response.text


def test_delete_file_removes_row_and_clears_vectors(db_client_with_spy):
    client, _ingest, delete_mock = db_client_with_spy

    resp = _post_files(client, [_make_pdf_upload("doc.pdf")])
    file_id = resp.json()["results"][0]["file_id"]
    delete_mock.reset_mock()

    r = client.delete(f"/files/{file_id}")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "deleted": True}
    delete_mock.assert_called_once_with(file_id)

    listing = client.get("/files").json()
    assert all(row["id"] != file_id for row in listing)


def test_delete_nonexistent_file_returns_ok_false(db_client):
    r = db_client.delete("/files/does-not-exist")
    assert r.status_code == 200
    assert r.json() == {"ok": False}


def test_delete_file_keeps_row_when_vector_delete_fails(db_client_with_spy):
    client, _ingest, delete_mock = db_client_with_spy

    resp = _post_files(client, [_make_pdf_upload("doc.pdf")])
    file_id = resp.json()["results"][0]["file_id"]

    delete_mock.reset_mock()
    delete_mock.side_effect = RuntimeError("pinecone down")

    r = client.delete(f"/files/{file_id}")
    assert r.status_code == 200
    assert r.json() == {"ok": False}

    listing = client.get("/files").json()
    assert any(row["id"] == file_id for row in listing)
