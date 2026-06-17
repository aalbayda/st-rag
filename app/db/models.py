
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


class File(SQLModel, table=True):

    __tablename__ = "files"

    id: str = Field(primary_key=True)

    name: str

    byte_size: Optional[int] = None

    page_count: Optional[int] = None

    chunk_count: Optional[int] = None

    status: str

    stage: Optional[str] = None

    error: Optional[str] = None

    created_at: str = Field(default_factory=_utcnow)


class ChatSession(SQLModel, table=True):

    __tablename__ = "chat_sessions"

    id: str = Field(primary_key=True)

    name: Optional[str] = None

    created_at: str = Field(default_factory=_utcnow)

    updated_at: str = Field(default_factory=_utcnow)


class Message(SQLModel, table=True):

    __tablename__ = "messages"

    id: str = Field(primary_key=True)

    session_id: str = Field(foreign_key="chat_sessions.id")

    role: str

    content: str

    reasoning: Optional[str] = None

    citations: Optional[str] = None

    created_at: str = Field(default_factory=_utcnow)
