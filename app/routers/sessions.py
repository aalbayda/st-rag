
from __future__ import annotations

import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import delete
from sqlmodel import Session, select

from app.db.engine import get_engine
from app.db.models import ChatSession, Message

router = APIRouter()


@router.post("/sessions", status_code=201)
def create_session() -> JSONResponse:
    session_id = str(uuid.uuid4())
    engine = get_engine()
    with Session(engine) as db:
        row = ChatSession(id=session_id)
        db.add(row)
        db.commit()
        db.refresh(row)
    return JSONResponse(
        content={"id": row.id, "name": row.name, "created_at": row.created_at},
        status_code=201,
    )


@router.get("/sessions")
def list_sessions() -> JSONResponse:
    try:
        engine = get_engine()
        with Session(engine) as db:
            rows = db.exec(
                select(ChatSession)
                .order_by(ChatSession.updated_at.desc())  # type: ignore[arg-type]
                .limit(100)
            ).all()
        return JSONResponse(
            content=[{"id": r.id, "name": r.name, "updated_at": r.updated_at} for r in rows],
            status_code=200,
        )
    except Exception:
        return JSONResponse(content=[], status_code=200)


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str) -> JSONResponse:
    try:
        engine = get_engine()
        with Session(engine) as db:
            rows = db.exec(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at)  # type: ignore[arg-type]
            ).all()
        return JSONResponse(
            content=[
                {
                    "role": r.role,
                    "content": r.content,
                    "reasoning": r.reasoning,
                    "citations": r.citations,
                    "created_at": r.created_at,
                }
                for r in rows
            ],
            status_code=200,
        )
    except Exception:
        return JSONResponse(content=[], status_code=200)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> JSONResponse:
    try:
        engine = get_engine()
        with Session(engine) as db:
            db.exec(delete(Message).where(Message.session_id == session_id))
            row = db.exec(select(ChatSession).where(ChatSession.id == session_id)).first()
            if row is not None:
                db.delete(row)
            else:
                return JSONResponse(content={"ok": False}, status_code=200)
            db.commit()
        return JSONResponse(content={"ok": True, "deleted": True}, status_code=200)
    except Exception:
        return JSONResponse(content={"ok": False}, status_code=200)
