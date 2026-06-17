
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.config import get_settings
from app.contracts import Answer
from app.db.engine import get_engine
from app.db.models import ChatSession, Message
from app.services.generation import answer_question

router = APIRouter()


class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The question to answer from the uploaded documents.",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Active chat session UUID.",
    )


def name_session_bg(session_id: str, first_question: str) -> None:
    from app.services.naming import generate_session_name  # noqa: PLC0415

    name = generate_session_name(first_question)
    fallback = f"Chat {datetime.now(timezone.utc).date().isoformat()}"
    final_name = name or fallback
    try:
        engine = get_engine()
        with Session(engine) as db:
            row = db.exec(select(ChatSession).where(ChatSession.id == session_id)).first()
            if row:
                row.name = final_name
                db.add(row)
                db.commit()
    except Exception:
        pass


@router.post("/chat", response_model=Answer)
def chat(req: ChatRequest, bg: BackgroundTasks) -> Answer:
    try:
        prior_turns: list[dict] = []
        if req.session_id:
            try:
                engine = get_engine()
                with Session(engine) as db:
                    all_msgs = db.exec(
                        select(Message)
                        .where(Message.session_id == req.session_id)
                        .order_by(Message.created_at)  # type: ignore[arg-type]
                    ).all()
                    n = get_settings().chat_history_turns
                    prior_turns = [
                        {"role": r.role, "content": r.content}
                        for r in all_msgs[-n:]
                    ]
            except Exception:
                prior_turns = []

        answer = answer_question(req.question, history=prior_turns)

        engine = get_engine()
        with Session(engine) as db:
            db.add(Message(
                id=str(uuid.uuid4()),
                session_id=req.session_id,
                role="user",
                content=req.question,
            ))
            db.add(Message(
                id=str(uuid.uuid4()),
                session_id=req.session_id,
                role="assistant",
                content=answer.answer,
                reasoning=answer.reasoning,
                citations=json.dumps([c.model_dump() for c in answer.citations]),
            ))
            sess_row = db.exec(select(ChatSession).where(ChatSession.id == req.session_id)).first()
            if sess_row:
                sess_row.updated_at = datetime.now(timezone.utc).isoformat()
                db.add(sess_row)
            db.commit()

            msg_count = db.exec(
                select(Message).where(Message.session_id == req.session_id)
            ).all()

        if len(msg_count) == 2:
            bg.add_task(name_session_bg, req.session_id, req.question)

        return answer
    except Exception:
        return Answer(
            answer="",
            reasoning="The service was unable to process this request.",
            citations=[],
            abstained=True,
        )
