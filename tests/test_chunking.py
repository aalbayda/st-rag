
from __future__ import annotations

import pytest

from app.services.chunking import chunk_unit


PAGE_ONE_MARKER = "ALPHA_PAGE_ONE_MARKER"
PAGE_TWO_MARKER = "BETA_PAGE_TWO_MARKER"


def _count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


def _make_long_text(n_tokens: int = 1500) -> str:
    word = "word"
    return " ".join([word] * n_tokens)


class TestChunkUnit:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_unit(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_returns_multiple_chunks(self):
        text = _make_long_text(n_tokens=1500)
        chunks = chunk_unit(text)
        assert len(chunks) > 1

    def test_each_chunk_within_token_limit(self):
        text = _make_long_text(n_tokens=2000)
        chunks = chunk_unit(text)
        for i, chunk in enumerate(chunks):
            token_count = _count_tokens(chunk)
            assert token_count <= 512, (
                f"Chunk {i} exceeds 512 tokens: {token_count} tokens. Text: {chunk[:80]!r}..."
            )

    def test_returns_list_of_strings(self):
        chunks = chunk_unit("Hello, world.")
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, str)

    def test_empty_string_returns_list(self):
        chunks = chunk_unit("")
        assert isinstance(chunks, list)

    def test_token_counting_not_character_counting(self):
        text = "a " * 200
        chunks = chunk_unit(text.strip())
        assert len(chunks) == 1


    def test_d03_page_one_chunks_do_not_contain_page_two_marker(self, sample_pdf_path):
        from app.services.parsing import pdf_pages

        pages = pdf_pages(sample_pdf_path)
        page1_text = next((t for t, p in pages if p == 1), None)
        assert page1_text is not None, "Page 1 text not found"

        chunks = chunk_unit(page1_text)
        for chunk in chunks:
            assert PAGE_TWO_MARKER not in chunk, (
                f"violation: chunk from page 1 contains page 2 marker: {chunk!r}"
            )

    def test_d03_page_two_chunks_do_not_contain_page_one_marker(self, sample_pdf_path):
        from app.services.parsing import pdf_pages

        pages = pdf_pages(sample_pdf_path)
        page2_text = next((t for t, p in pages if p == 2), None)
        assert page2_text is not None, "Page 2 text not found"

        chunks = chunk_unit(page2_text)
        for chunk in chunks:
            assert PAGE_ONE_MARKER not in chunk, (
                f"violation: chunk from page 2 contains page 1 marker: {chunk!r}"
            )

    def test_d03_no_chunk_spans_both_page_markers(self, sample_pdf_path):
        from app.services.parsing import pdf_pages

        pages = pdf_pages(sample_pdf_path)
        all_chunks = []
        for text, page_num in pages:
            for chunk in chunk_unit(text):
                all_chunks.append((chunk, page_num))

        for chunk, page_num in all_chunks:
            assert not (PAGE_ONE_MARKER in chunk and PAGE_TWO_MARKER in chunk), (
                f"violation: chunk contains both page markers "
                f"(page {page_num}): {chunk[:120]!r}"
            )


    def test_source_uses_from_tiktoken_encoder(self):
        import inspect

        import app.services.chunking as c_mod

        src = inspect.getsource(c_mod)
        assert "from_tiktoken_encoder" in src, (
            "chunking.py must use RecursiveCharacterTextSplitter.from_tiktoken_encoder"
        )

    def test_source_reads_chunk_size_from_settings(self):
        import inspect

        import app.services.chunking as c_mod

        src = inspect.getsource(c_mod)
        assert "chunk_size" in src
        assert "get_settings()" in src

    def test_source_reads_chunk_overlap_from_settings(self):
        import inspect

        import app.services.chunking as c_mod

        src = inspect.getsource(c_mod)
        assert "chunk_overlap" in src

    def test_source_reads_chunk_encoding_from_settings(self):
        import inspect

        import app.services.chunking as c_mod

        src = inspect.getsource(c_mod)
        assert "chunk_encoding" in src
