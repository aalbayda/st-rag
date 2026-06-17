
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.clients.openai_client import verify_models
from app.clients.pinecone_client import ping_index
from app.db.engine import get_engine

router = APIRouter()


def _probe_pinecone() -> dict[str, Any]:
    try:
        result = ping_index()
        if result.get("reachable"):
            return {
                "reachable": True,
                "dimension": result.get("dimension", 0),
                "metric": result.get("metric", ""),
            }
        return {"reachable": False, "reason": "unreachable"}
    except Exception:
        return {"reachable": False, "reason": "unreachable"}


def _probe_openrouter() -> dict[str, Any]:
    try:
        flags = verify_models()
        all_present = all(flags.values())
        return {"ok": all_present, **flags}
    except Exception:
        return {"ok": False, "reason": "unreachable"}


def _probe_database() -> dict[str, Any]:
    try:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception:
        return {"ok": False, "reason": "unreachable"}


@router.get("/health")
def health_check() -> JSONResponse:
    pinecone_status = _probe_pinecone()
    openrouter_status = _probe_openrouter()
    database_status = _probe_database()

    all_ok = (
        pinecone_status.get("reachable", False)
        and openrouter_status.get("ok", False)
        and database_status.get("ok", False)
    )

    body = {
        "status": "ok" if all_ok else "degraded",
        "pinecone": pinecone_status,
        "openrouter": openrouter_status,
        "database": database_status,
    }

    status_code = 200 if all_ok else 503
    return JSONResponse(content=body, status_code=status_code)
