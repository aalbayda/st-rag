
from __future__ import annotations

import re

from app.clients.openai_client import generate_answer
from app.contracts import Answer, Citation, DocxLocator, PdfLocator
from app.services.retrieval import retrieve


SYSTEM_PROMPT = """You are Bob, a precise document-grounded assistant. You answer questions
only from the numbered passages provided below. Each passage is labelled [1], [2], etc.

Rules:
1. Cite every factual claim with the matching [n] inline in your answer text AND list
   each used passage number in cited_ids.
2. Your reasoning field must explain how the cited passages support your answer.
3. If the provided passages do not contain sufficient information to answer the question,
   set abstained=true, leave cited_ids empty, and explain why in the reasoning field.
4. Do not use any knowledge beyond the numbered passages. If unsure, abstain.
5. Do not fabricate passage numbers not present in the list below.
Prior conversation turns may be provided for context only; citations must still come
exclusively from the numbered passages listed below."""


def _abstain(reasoning: str) -> Answer:
    return Answer(
        answer="",
        reasoning=reasoning,
        citations=[],
        abstained=True,
    )


def _enrich(local_id: str, meta: dict) -> Citation:
    if meta["kind"] == "pdf":
        locator = PdfLocator(page=meta["page"])
    else:
        locator = DocxLocator(
            section=meta.get("section"),
            paragraph_index=meta["paragraph_index"],
        )

    return Citation(
        id=local_id,
        file_id=meta["file_id"],
        file_name=meta["file_name"],
        locator=locator,
        chunk_text=meta["chunk_text"],
    )


def _format_history(turns: list[dict]) -> str:
    if not turns:
        return ""
    lines = ["Prior conversation:"]
    for turn in turns:
        label = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{label}: {turn.get('content', '')}")
    return "\n".join(lines)


def answer_question(question: str, history: list[dict] | None = None) -> Answer:
    context_block, id_to_meta = retrieve(question)

    if not id_to_meta:
        return _abstain("No relevant documents found.")

    try:
        effective_history = history or []
        history_block = _format_history(effective_history)
    except Exception:
        history_block = ""

    if history_block:
        user_message = f"{history_block}\n\n{context_block}\n\nQuestion: {question}"
    else:
        user_message = f"{context_block}\n\nQuestion: {question}"

    model_answer = generate_answer(
        SYSTEM_PROMPT,
        user_message,
    )

    if model_answer is None:
        return _abstain("Unable to produce a grounded answer.")

    if model_answer.abstained:
        return _abstain(model_answer.reasoning or "No relevant documents found.")

    def _canonical_id(raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        return f"[{digits}]" if digits else raw

    cited_keys = {_canonical_id(cid) for cid in model_answer.cited_ids}
    valid = [key for key in id_to_meta if key in cited_keys]
    valid_nums = {cid.strip("[]") for cid in valid}

    def strip_orphans(text: str) -> str:
        return re.sub(
            r"\[(\d+)\]",
            lambda m: m.group(0) if m.group(1) in valid_nums else "",
            text,
        )

    citations = [_enrich(cid, id_to_meta[cid]) for cid in valid]

    return Answer(
        answer=strip_orphans(model_answer.answer),
        reasoning=model_answer.reasoning,
        citations=citations,
        abstained=False,
    )
