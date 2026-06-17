
from __future__ import annotations

import os

from app.clients.openai_client import embed_texts
from app.clients.pinecone_client import delete_by_file, sparse_encode_passages, upsert_chunks
from app.db.engine import get_engine
from app.db.models import File
from app.services.chunking import chunk_unit
from app.services.parsing import docx_units, pdf_pages
from sqlmodel import Session, select


def _set_stage(file_id: str, stage: str) -> None:
    with Session(get_engine()) as session:
        row = session.exec(select(File).where(File.id == file_id)).one()
        row.stage = stage
        session.add(row)
        session.commit()


def _set_success(file_id: str, chunk_count: int, page_count: int | None) -> None:
    with Session(get_engine()) as session:
        row = session.exec(select(File).where(File.id == file_id)).one()
        row.status = "ready"
        row.stage = "indexed"
        row.chunk_count = chunk_count
        row.page_count = page_count
        session.add(row)
        session.commit()


def _set_failed(file_id: str, safe_message: str) -> None:
    with Session(get_engine()) as session:
        row = session.exec(select(File).where(File.id == file_id)).one()
        row.status = "failed"
        row.stage = "failed"
        row.error = safe_message
        session.add(row)
        session.commit()


def ingest_file(file_id: str, path: str, kind: str, file_name: str) -> None:
    try:
        _set_stage(file_id, "parsing")

        vectors: list[dict] = []
        chunk_texts: list[str] = []
        chunk_locators: list[dict] = []

        if kind == "pdf":
            pages = pdf_pages(path)
            for text, page_number in pages:
                for chunk_text in chunk_unit(text):
                    chunk_texts.append(chunk_text)
                    chunk_locators.append({"page": page_number})

            page_count: int | None = len(pages)

        else:
            units = docx_units(path)
            for text, section, paragraph_index in units:
                for chunk_text in chunk_unit(text):
                    chunk_texts.append(chunk_text)
                    chunk_locators.append(
                        {
                            "section": section,
                            "paragraph_index": paragraph_index,
                        }
                    )

            page_count = None

        _set_stage(file_id, "embedding")

        embeddings = embed_texts(chunk_texts)

        sparse_vecs = sparse_encode_passages(chunk_texts)

        for n, (chunk_text, locator_fields, embedding, sparse_vec) in enumerate(
            zip(chunk_texts, chunk_locators, embeddings, sparse_vecs)
        ):
            metadata: dict = {
                "file_id": file_id,
                "file_name": file_name,
                "chunk_text": chunk_text,
                "kind": kind,
                **locator_fields,
            }
            vectors.append(
                {
                    "id": f"{file_id}#{n}",
                    "values": embedding,
                    "sparse_values": sparse_vec,
                    "metadata": metadata,
                }
            )

        upsert_chunks(vectors)

        _set_success(file_id, chunk_count=len(vectors), page_count=page_count)

    except Exception:
        try:
            delete_by_file(file_id)
        except Exception:
            pass

        _set_failed(file_id, "ingestion failed")

    finally:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
