
from __future__ import annotations

import os
import re
import shutil
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENROUTER_API_KEY", "secret-or")
os.environ.setdefault("PINECONE_API_KEY", "secret-p")
os.environ.setdefault("PINECONE_INDEX_NAME", "rag-dense")


FAKE_EMBEDDING = [0.1, 0.2, 0.3]


def _copy_fixture(src: str, dest_dir: str, dest_name: str) -> str:
    dest = os.path.join(dest_dir, dest_name)
    shutil.copy2(src, dest)
    return dest


def _make_embed_mock(n_texts: int | None = None):

    def _embed(texts: list[str]) -> list[list[float]]:
        return [FAKE_EMBEDDING[:] for _ in texts]

    mock = MagicMock(side_effect=_embed)
    return mock


def _setup_db(tmp_path, monkeypatch):
    from app.config import get_settings
    from app.db.engine import get_engine, init_db

    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    get_settings.cache_clear()
    get_engine.cache_clear()
    init_db()
    return db_file


def test_pipeline_e2e(tmp_path, monkeypatch, sample_pdf_path, sample_docx_path):
    _setup_db(tmp_path, monkeypatch)

    upserted: list[dict] = []

    def _upsert(vectors):
        upserted.extend(vectors)

    from app.services.ids import make_file_id

    pdf_file_id = make_file_id("sample.pdf")
    docx_file_id = make_file_id("sample.docx")

    from sqlmodel import Session

    from app.db.engine import get_engine
    from app.db.models import File

    with Session(get_engine()) as session:
        session.add(File(id=pdf_file_id, name="sample.pdf", status="pending", stage="queued"))
        session.add(File(id=docx_file_id, name="sample.docx", status="pending", stage="queued"))
        session.commit()

    pdf_copy = _copy_fixture(sample_pdf_path, str(tmp_path), "sample_copy.pdf")
    docx_copy = _copy_fixture(sample_docx_path, str(tmp_path), "sample_copy.docx")

    def _fake_sparse(texts):
        return [{"indices": [1], "values": [0.5]} for _ in texts]

    with (
        patch("app.services.ingestion.embed_texts", side_effect=_make_embed_mock()),
        patch("app.services.ingestion.sparse_encode_passages", side_effect=_fake_sparse),
        patch("app.services.ingestion.upsert_chunks", side_effect=_upsert),
        patch("app.services.ingestion.delete_by_file"),
    ):
        from app.services.ingestion import ingest_file

        ingest_file(pdf_file_id, pdf_copy, "pdf", "sample.pdf")
        ingest_file(docx_file_id, docx_copy, "docx", "sample.docx")

    assert len(upserted) > 0, "No vectors upserted"
    for v in upserted:
        meta = v["metadata"]
        assert "file_id" in meta and meta["file_id"] is not None
        assert "file_name" in meta and meta["file_name"] is not None
        assert "chunk_text" in meta and meta["chunk_text"]
        assert "kind" in meta, f"Vector {v['id']} missing 'kind'"
        vid = v["id"]
        assert re.match(r".+#\d+$", vid), f"Vector id '{vid}' does not match {{file_id}}#<int>"

    pdf_vectors = [v for v in upserted if v["metadata"]["file_id"] == pdf_file_id]
    assert len(pdf_vectors) > 0, "No PDF vectors upserted"
    for v in pdf_vectors:
        meta = v["metadata"]
        assert meta["kind"] == "pdf"
        assert "page" in meta, "PDF vector missing 'page'"
        assert isinstance(meta["page"], int) and meta["page"] >= 1, (
            f"PDF page must be int >= 1, got {meta['page']}"
        )

    docx_vectors = [v for v in upserted if v["metadata"]["file_id"] == docx_file_id]
    assert len(docx_vectors) > 0, "No DOCX vectors upserted"
    for v in docx_vectors:
        meta = v["metadata"]
        assert meta["kind"] == "docx"
        assert "paragraph_index" in meta, "DOCX vector missing 'paragraph_index'"
        assert "section" in meta, "DOCX vector missing 'section'"
        assert "page" not in meta or meta.get("page") is None, (
            f"DOCX vector must not carry a real page number, got {meta.get('page')}"
        )


def test_stage_progression(tmp_path, monkeypatch, sample_pdf_path):
    _setup_db(tmp_path, monkeypatch)

    upserted: list[dict] = []

    def _upsert(vectors):
        upserted.extend(vectors)

    from app.services.ids import make_file_id

    file_id = make_file_id("sample.pdf")

    from sqlmodel import Session

    from app.db.engine import get_engine
    from app.db.models import File

    with Session(get_engine()) as session:
        session.add(File(id=file_id, name="sample.pdf", status="pending", stage="queued"))
        session.commit()

    pdf_copy = _copy_fixture(sample_pdf_path, str(tmp_path), "stage_prog.pdf")

    def _fake_sparse(texts):
        return [{"indices": [1], "values": [0.5]} for _ in texts]

    with (
        patch("app.services.ingestion.embed_texts", side_effect=_make_embed_mock()),
        patch("app.services.ingestion.sparse_encode_passages", side_effect=_fake_sparse),
        patch("app.services.ingestion.upsert_chunks", side_effect=_upsert),
        patch("app.services.ingestion.delete_by_file"),
    ):
        from app.services.ingestion import ingest_file

        ingest_file(file_id, pdf_copy, "pdf", "sample.pdf")

    with Session(get_engine()) as session:
        from sqlmodel import select

        row = session.exec(select(File).where(File.id == file_id)).one()

    assert row.status == "ready", f"Expected status=ready, got {row.status}"
    assert row.stage == "indexed", f"Expected stage=indexed, got {row.stage}"
    assert row.chunk_count is not None and row.chunk_count > 0, (
        f"chunk_count should be > 0, got {row.chunk_count}"
    )
    assert row.page_count is not None and row.page_count >= 1, (
        f"page_count should be >= 1 for PDF, got {row.page_count}"
    )
    assert row.chunk_count == len(upserted), (
        f"chunk_count {row.chunk_count} != actual upserted {len(upserted)}"
    )


def test_partial_failure(tmp_path, monkeypatch, sample_pdf_path):
    _setup_db(tmp_path, monkeypatch)

    upserted: list[dict] = []
    deleted_file_ids: list[str] = []

    def _upsert(vectors):
        upserted.extend(vectors)

    def _delete(file_id: str):
        deleted_file_ids.append(file_id)

    from app.services.ids import make_file_id

    good_file_id = make_file_id("good.pdf")
    bad_file_id = make_file_id("bad.pdf")

    from sqlmodel import Session

    from app.db.engine import get_engine
    from app.db.models import File

    with Session(get_engine()) as session:
        session.add(File(id=good_file_id, name="good.pdf", status="pending", stage="queued"))
        session.add(File(id=bad_file_id, name="bad.pdf", status="pending", stage="queued"))
        session.commit()

    call_count = {"n": 0}

    def _selective_embed(texts):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("Simulated embedding failure")
        return [FAKE_EMBEDDING[:] for _ in texts]

    good_copy = _copy_fixture(sample_pdf_path, str(tmp_path), "good_copy.pdf")
    bad_copy = _copy_fixture(sample_pdf_path, str(tmp_path), "bad_copy.pdf")

    def _fake_sparse(texts):
        return [{"indices": [1], "values": [0.5]} for _ in texts]

    with (
        patch("app.services.ingestion.embed_texts", side_effect=_selective_embed),
        patch("app.services.ingestion.sparse_encode_passages", side_effect=_fake_sparse),
        patch("app.services.ingestion.upsert_chunks", side_effect=_upsert),
        patch("app.services.ingestion.delete_by_file", side_effect=_delete),
    ):
        from app.services import ingestion

        ingestion.ingest_file(good_file_id, good_copy, "pdf", "good.pdf")
        ingestion.ingest_file(bad_file_id, bad_copy, "pdf", "bad.pdf")

    from sqlmodel import select

    with Session(get_engine()) as session:
        good_row = session.exec(select(File).where(File.id == good_file_id)).one()
        bad_row = session.exec(select(File).where(File.id == bad_file_id)).one()

    assert good_row.stage == "indexed", f"Good file stage: {good_row.stage}"
    assert good_row.status == "ready", f"Good file status: {good_row.status}"

    assert bad_row.stage == "failed", f"Bad file stage: {bad_row.stage}"
    assert bad_row.status == "failed", f"Bad file status: {bad_row.status}"

    assert bad_row.error is not None, "Bad file should have an error message"
    assert "Traceback" not in (bad_row.error or ""), "Error must not contain traceback"
    assert "secret" not in (bad_row.error or "").lower(), "Error must not contain secrets"

    assert bad_file_id in deleted_file_ids, (
        f"delete_by_file should be called for bad file_id; called for: {deleted_file_ids}"
    )
    assert good_file_id not in deleted_file_ids, (
        "delete_by_file must not be called for good file_id"
    )
