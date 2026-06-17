
from __future__ import annotations

import pytest

from app.services.parsing import docx_units, pdf_pages

PAGE_ONE_MARKER = "ALPHA_PAGE_ONE_MARKER"
PAGE_TWO_MARKER = "BETA_PAGE_TWO_MARKER"


class TestPdfPages:
    def test_returns_two_tuples(self, sample_pdf_path):
        pages = pdf_pages(sample_pdf_path)
        assert len(pages) == 2

    def test_tuple_structure_is_text_and_int(self, sample_pdf_path):
        pages = pdf_pages(sample_pdf_path)
        for text, page_num in pages:
            assert isinstance(text, str)
            assert isinstance(page_num, int)

    def test_page_one_marker_is_on_page_1(self, sample_pdf_path):
        pages = pdf_pages(sample_pdf_path)
        marker_pages = [p for t, p in pages if PAGE_ONE_MARKER in t]
        assert len(marker_pages) == 1, f"Expected 1 page with marker, got {marker_pages}"
        assert marker_pages[0] == 1, (
            f"PAGE_ONE_MARKER must be on page 1 (1-based), got page {marker_pages[0]}"
        )

    def test_page_two_marker_is_on_page_2(self, sample_pdf_path):
        pages = pdf_pages(sample_pdf_path)
        marker_pages = [p for t, p in pages if PAGE_TWO_MARKER in t]
        assert len(marker_pages) == 1
        assert marker_pages[0] == 2

    def test_all_page_numbers_are_1_based(self, sample_pdf_path):
        pages = pdf_pages(sample_pdf_path)
        for _, page_num in pages:
            assert page_num >= 1, f"page_num must be >= 1, got {page_num}"

    def test_page_numbers_are_sequential(self, sample_pdf_path):
        pages = pdf_pages(sample_pdf_path)
        page_nums = [p for _, p in pages]
        assert page_nums == sorted(page_nums)

    def test_page_one_text_does_not_contain_page_two_marker(self, sample_pdf_path):
        pages = pdf_pages(sample_pdf_path)
        for text, page_num in pages:
            if page_num == 1:
                assert PAGE_TWO_MARKER not in text
            if page_num == 2:
                assert PAGE_ONE_MARKER not in text

    def test_source_contains_page_chunks_true(self):
        import inspect

        import app.services.parsing as p_mod

        src = inspect.getsource(p_mod)
        assert "page_chunks=True" in src, "parsing.py must use page_chunks=True"


class TestDocxUnits:
    def test_returns_non_empty_list(self, sample_docx_path):
        units = docx_units(sample_docx_path)
        assert len(units) >= 1

    def test_tuple_structure_is_text_section_int(self, sample_docx_path):
        units = docx_units(sample_docx_path)
        for tup in units:
            assert len(tup) == 3, f"Expected 3-tuple, got {len(tup)}-tuple: {tup}"
            text, section, para_idx = tup
            assert isinstance(text, str)
            assert section is None or isinstance(section, str)
            assert isinstance(para_idx, int)

    def test_body_paragraphs_carry_introduction_section(self, sample_docx_path):
        units = docx_units(sample_docx_path)
        body_units = [u for u in units if "paragraph" in u[0].lower() or "first" in u[0].lower() or "second" in u[0].lower()]
        assert len(body_units) >= 1, "Should find at least one body paragraph unit"
        for text, section, para_idx in body_units:
            assert section == "Introduction", (
                f"Body paragraph should have section='Introduction', got {section!r}: {text!r}"
            )

    def test_paragraph_index_is_non_negative_int(self, sample_docx_path):
        units = docx_units(sample_docx_path)
        for text, section, para_idx in units:
            assert para_idx >= 0, f"paragraph_index must be >= 0, got {para_idx}"

    def test_no_page_value_in_tuple(self, sample_docx_path):
        units = docx_units(sample_docx_path)
        for tup in units:
            assert len(tup) == 3, (
                f"DOCX units must be 3-tuples (text, section, para_idx). Got {len(tup)}: {tup}"
            )

    def test_paragraph_indices_are_absolute(self, sample_docx_path):
        units = docx_units(sample_docx_path)
        indices = [u[2] for u in units]
        assert len(set(indices)) == len(indices), f"Duplicate paragraph indices: {indices}"

    def test_units_contain_expected_text(self, sample_docx_path):
        units = docx_units(sample_docx_path)
        all_text = " ".join(u[0] for u in units)
        assert "First paragraph" in all_text or "Second paragraph" in all_text

    def test_source_contains_paragraph_index(self):
        import inspect

        import app.services.parsing as p_mod

        src = inspect.getsource(p_mod)
        assert "paragraph_index" in src or "paragraph" in src
