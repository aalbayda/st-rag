
import pytest
from pydantic import ValidationError


def test_imports_are_clean():
    from app.contracts import Answer, Citation, ModelAnswer, PdfLocator, DocxLocator  # noqa: F401


def test_citation_has_required_fields():
    from app.contracts import Citation, PdfLocator

    loc = PdfLocator(page=5)
    cit = Citation(
        id="[1]",
        file_id="file-abc",
        file_name="report.pdf",
        locator=loc,
        chunk_text="Some text from the document.",
    )
    assert cit.id == "[1]"
    assert cit.file_id == "file-abc"
    assert cit.file_name == "report.pdf"
    assert cit.chunk_text == "Some text from the document."


def test_citation_has_no_quote_field():
    from app.contracts import Citation, PdfLocator

    loc = PdfLocator(page=1)
    cit = Citation(
        id="[1]",
        file_id="file-abc",
        file_name="doc.pdf",
        locator=loc,
        chunk_text="text",
    )
    assert not hasattr(cit, "quote"), "Citation must not have a 'quote' field"


def test_model_answer_lacks_file_name_locator_chunk_text():
    from app.contracts import ModelAnswer

    ma = ModelAnswer(
        answer="The answer is [1].",
        reasoning="Based on the document.",
        cited_ids=["[1]"],
        abstained=False,
    )
    assert not hasattr(ma, "chunk_text"), "ModelAnswer must not expose chunk_text"
    assert not hasattr(ma, "file_name"), "ModelAnswer must not expose file_name"
    assert not hasattr(ma, "locator"), "ModelAnswer must not expose locator"


def test_answer_abstained_true_requires_empty_citations():
    from app.contracts import Answer, Citation, PdfLocator

    loc = PdfLocator(page=3)
    cit = Citation(
        id="[1]",
        file_id="f1",
        file_name="doc.pdf",
        locator=loc,
        chunk_text="chunk text",
    )
    with pytest.raises(ValidationError):
        Answer(
            answer="I cannot answer.",
            reasoning="No relevant docs.",
            citations=[cit],
            abstained=True,
        )


def test_answer_abstained_false_allows_citations():
    from app.contracts import Answer, Citation, PdfLocator

    loc = PdfLocator(page=7)
    cit = Citation(
        id="[1]",
        file_id="f1",
        file_name="doc.pdf",
        locator=loc,
        chunk_text="chunk text",
    )
    ans = Answer(
        answer="The answer is [1].",
        reasoning="Based on doc.",
        citations=[cit],
        abstained=False,
    )
    assert ans.abstained is False
    assert len(ans.citations) == 1


def test_answer_abstained_true_allows_empty_citations():
    from app.contracts import Answer

    ans = Answer(
        answer="I don't know.",
        reasoning="No relevant documents found.",
        citations=[],
        abstained=True,
    )
    assert ans.abstained is True
    assert ans.citations == []


def test_pdf_locator_label_contains_page():
    from app.contracts import PdfLocator

    loc = PdfLocator(page=12)
    label = loc.label()
    assert "p. " in label, f"Expected 'p. ' in PDF label, got: {label!r}"
    assert "12" in label, f"Expected page 12 in PDF label, got: {label!r}"


def test_docx_locator_label_is_not_page_label():
    from app.contracts import DocxLocator

    loc = DocxLocator(paragraph_index=42)
    label = loc.label()
    assert "p. " not in label, f"DOCX label must not look like a page label, got: {label!r}"
    assert loc.page is None, f"DocxLocator.page must be None, got: {loc.page!r}"


def test_docx_locator_with_section():
    from app.contracts import DocxLocator

    loc = DocxLocator(section="Introduction", paragraph_index=0)
    label = loc.label()
    assert "Introduction" in label or "§" in label or "¶" in label, (
        f"Expected section label to reference 'Introduction' or use §/¶, got: {label!r}"
    )


def test_locator_discriminated_by_kind():
    from app.contracts import PdfLocator, DocxLocator

    pdf = PdfLocator(page=1)
    docx = DocxLocator(paragraph_index=5)
    assert pdf.kind == "pdf"
    assert docx.kind == "docx"


def test_model_answer_carries_cited_ids_and_abstained():
    from app.contracts import ModelAnswer

    ma = ModelAnswer(
        answer="No relevant info.",
        reasoning="Nothing found.",
        cited_ids=[],
        abstained=True,
    )
    assert ma.abstained is True
    assert ma.cited_ids == []
