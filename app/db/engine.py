
from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    db_url = f"sqlite:///{settings.db_path}"

    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    journal_mode = settings.db_journal_mode

    @event.listens_for(engine, "connect")
    def _set_journal_mode(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute(f"PRAGMA journal_mode={journal_mode}")
        cursor.close()

    return engine


def init_db() -> None:
    import app.db.models as _models  # noqa: F401

    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    engine = get_engine()
    with Session(engine) as session:
        yield session
