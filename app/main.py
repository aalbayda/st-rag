
from __future__ import annotations

from fastapi import FastAPI

from app.routers import chat, files, health, sessions


def create_app() -> FastAPI:
    application = FastAPI(
        title="Ask Bob API",
        description="Retrieval-Augmented Generation API for citation-grounded answers.",
        version="0.1.0",
    )

    application.include_router(health.router)

    application.include_router(files.router)

    application.include_router(chat.router)

    application.include_router(sessions.router)

    return application


app = create_app()
