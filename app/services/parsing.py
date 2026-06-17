
from __future__ import annotations


def pdf_pages(path: str) -> list[tuple[str, int]]:
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(path, page_chunks=True)
    result: list[tuple[str, int]] = []
    for chunk in chunks:
        text: str = chunk["text"]
        page_number: int = chunk["metadata"]["page_number"]
        result.append((text, page_number))
    return result


def docx_units(path: str) -> list[tuple[str, str | None, int]]:
    from docx import Document

    doc = Document(path)
    result: list[tuple[str, str | None, int]] = []
    running_section: str | None = None

    for i, para in enumerate(doc.paragraphs):
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading") or style_name == "Title":
            running_section = para.text.strip() or running_section

        text = para.text.strip()
        if not text:
            continue

        result.append((text, running_section, i))

    return result
