
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, UploadFile
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from app.clients.pinecone_client import delete_by_file
from app.config import get_settings
from app.db.engine import get_engine
from app.db.models import File
from app.services.ids import make_file_id
from app.services.ingestion import ingest_file
from app.services.validation import validate_upload

router = APIRouter()


@router.post("/files", status_code=202)
async def upload_files(
    files: list[UploadFile],
    bg: BackgroundTasks,
) -> JSONResponse:
    raw_files: list[tuple[bytes, str]] = []
    for f in files:
        data = await f.read()
        raw_files.append((data, f.filename or ""))

    validation_results = validate_upload(raw_files)

    results: list[dict] = []

    for (data, filename), vr in zip(raw_files, validation_results):
        if not vr["accepted"]:
            results.append(
                {
                    "name": filename,
                    "accepted": False,
                    "reason": vr["reason"],
                }
            )
            continue

        kind: str = vr["kind"]
        file_id = make_file_id(filename)

        try:
            delete_by_file(file_id)
        except Exception:
            pass

        tmp_dir = tempfile.gettempdir()
        temp_path = os.path.join(tmp_dir, f"rag_ingest_{file_id}.tmp")
        with open(temp_path, "wb") as tmp_f:
            tmp_f.write(data)

        engine = get_engine()
        with Session(engine) as session:
            existing = session.exec(
                select(File).where(File.id == file_id)
            ).first()
            if existing is not None:
                session.delete(existing)
                session.flush()

            row = File(
                id=file_id,
                name=filename,
                byte_size=len(data),
                status="pending",
                stage="queued",
            )
            session.add(row)
            session.commit()

        bg.add_task(ingest_file, file_id, temp_path, kind, filename)

        results.append(
            {
                "file_id": file_id,
                "name": filename,
                "accepted": True,
            }
        )

    return JSONResponse(content={"results": results}, status_code=202)


@router.get("/files")
def list_files() -> JSONResponse:
    try:
        engine = get_engine()
        with Session(engine) as session:
            rows = session.exec(select(File).order_by(File.created_at)).all()  # type: ignore[arg-type]

        file_list = [
            {
                "id": r.id,
                "name": r.name,
                "byte_size": r.byte_size,
                "page_count": r.page_count,
                "chunk_count": r.chunk_count,
                "status": r.status,
                "stage": r.stage,
                "error": r.error,
                "created_at": r.created_at,
            }
            for r in rows
        ]
        return JSONResponse(content=file_list, status_code=200)
    except Exception:
        return JSONResponse(content=[], status_code=200)


@router.delete("/files/{file_id}")
def delete_file(file_id: str) -> JSONResponse:
    try:
        delete_by_file(file_id)

        engine = get_engine()
        with Session(engine) as session:
            row = session.exec(select(File).where(File.id == file_id)).first()
            if row is None:
                return JSONResponse(content={"ok": False}, status_code=200)
            session.delete(row)
            session.commit()
        return JSONResponse(content={"ok": True, "deleted": True}, status_code=200)
    except Exception:
        return JSONResponse(content={"ok": False}, status_code=200)
