from __future__ import annotations

from functools import lru_cache

import httpx

from app.config import get_settings


@lru_cache(maxsize=1)
def get_client() -> httpx.Client:
    settings = get_settings()
    return httpx.Client(
        base_url=settings.api_base_url,
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0),
    )


def post_chat(question: str, session_id: str) -> dict:
    r = get_client().post("/chat", json={"question": question, "session_id": session_id})
    r.raise_for_status()
    return r.json()


def post_session() -> dict:
    r = get_client().post("/sessions")
    r.raise_for_status()
    return r.json()


def get_sessions() -> list:
    r = get_client().get("/sessions")
    r.raise_for_status()
    return r.json()


def get_session_messages(session_id: str) -> list:
    r = get_client().get(f"/sessions/{session_id}/messages")
    r.raise_for_status()
    return r.json()


def delete_session(session_id: str) -> dict:
    r = get_client().delete(f"/sessions/{session_id}")
    r.raise_for_status()
    return r.json()


def post_files(uploaded: list) -> dict:
    files = [
        (
            "files",
            (
                f.name,
                f.getvalue(),
                getattr(f, "type", None) or "application/octet-stream",
            ),
        )
        for f in uploaded
    ]
    r = get_client().post("/files", files=files)
    return r.json()


def get_files() -> list:
    r = get_client().get("/files")
    return r.json()


def delete_file(file_id: str) -> dict:
    r = get_client().delete(f"/files/{file_id}")
    r.raise_for_status()
    return r.json()
