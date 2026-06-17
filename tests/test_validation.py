
from __future__ import annotations

from app.services.validation import sniff_kind, validate_upload


PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
MZ_BYTES = b"MZ\x90\x00\x03\x00\x00\x00"


def _make_pdf_bytes(extra: int = 0) -> bytes:
    return PDF_MAGIC + b"\x00" * extra


def _make_docx_bytes(extra: int = 0) -> bytes:
    return ZIP_MAGIC + b"\x00" * extra


class TestSniffKind:
    def test_valid_pdf_bytes_and_extension(self):
        assert sniff_kind(_make_pdf_bytes(), "document.pdf") == "pdf"

    def test_valid_docx_bytes_and_extension(self):
        assert sniff_kind(_make_docx_bytes(), "report.docx") == "docx"

    def test_spoofed_extension_pdf_but_non_pdf_bytes(self):
        assert sniff_kind(MZ_BYTES, "evil.pdf") is None

    def test_spoofed_extension_docx_but_non_zip_bytes(self):
        assert sniff_kind(MZ_BYTES, "evil.docx") is None

    def test_pdf_bytes_with_wrong_extension(self):
        assert sniff_kind(_make_pdf_bytes(), "document.txt") is None

    def test_docx_bytes_with_pdf_extension(self):
        assert sniff_kind(_make_docx_bytes(), "sneaky.pdf") is None

    def test_pdf_bytes_with_pdf_uppercase_extension(self):
        assert sniff_kind(_make_pdf_bytes(), "DOCUMENT.PDF") == "pdf"

    def test_docx_bytes_with_docx_uppercase_extension(self):
        assert sniff_kind(_make_docx_bytes(), "REPORT.DOCX") == "docx"

    def test_empty_bytes_rejected(self):
        assert sniff_kind(b"", "empty.pdf") is None

    def test_non_document_type_rejected(self):
        assert sniff_kind(b"\x00\x01\x02\x03", "image.png") is None


class TestValidateUpload:
    def test_single_valid_pdf_accepted(self):
        files = [(_make_pdf_bytes(), "doc.pdf")]
        result = validate_upload(files)
        assert len(result) == 1
        assert result[0]["accepted"] is True
        assert result[0]["kind"] == "pdf"
        assert result[0]["name"] == "doc.pdf"

    def test_single_valid_docx_accepted(self):
        files = [(_make_docx_bytes(), "report.docx")]
        result = validate_upload(files)
        assert len(result) == 1
        assert result[0]["accepted"] is True
        assert result[0]["kind"] == "docx"

    def test_over_5_files_all_rejected(self):
        files = [(_make_pdf_bytes(), f"file{i}.pdf") for i in range(6)]
        result = validate_upload(files)
        assert len(result) == 6
        for r in result:
            assert r["accepted"] is False, f"Expected rejected: {r}"
            assert r["reason"] is not None
            reason_lower = r["reason"].lower()
            assert "5" in reason_lower or "limit" in reason_lower, (
                f"Reason should mention 5-file limit: {r['reason']!r}"
            )

    def test_exactly_5_files_all_accepted(self):
        files = [(_make_pdf_bytes(), f"file{i}.pdf") for i in range(5)]
        result = validate_upload(files)
        assert all(r["accepted"] for r in result)

    def test_oversized_file_rejected(self):
        big_data = _make_pdf_bytes(extra=20 * 1024 * 1024 + 1)
        small_data = _make_pdf_bytes()
        files = [(small_data, "small.pdf"), (big_data, "big.pdf")]
        result = validate_upload(files)
        by_name = {r["name"]: r for r in result}
        assert by_name["small.pdf"]["accepted"] is True
        assert by_name["big.pdf"]["accepted"] is False
        reason = by_name["big.pdf"]["reason"]
        assert reason is not None
        reason_lower = reason.lower()
        assert "size" in reason_lower or "large" in reason_lower or "20mb" in reason_lower or "mb" in reason_lower

    def test_unsupported_type_rejected(self):
        files = [(b"\x89PNG\r\n\x1a\n", "image.png")]
        result = validate_upload(files)
        assert result[0]["accepted"] is False
        reason = result[0]["reason"]
        assert reason is not None
        reason_lower = reason.lower()
        assert "type" in reason_lower or "pdf" in reason_lower or "unsupported" in reason_lower

    def test_mixed_batch_partial_accept(self):
        files = [
            (_make_pdf_bytes(), "good.pdf"),
            (b"\x89PNG\r\n\x1a\n", "bad.png"),
        ]
        result = validate_upload(files)
        by_name = {r["name"]: r for r in result}
        assert by_name["good.pdf"]["accepted"] is True
        assert by_name["bad.png"]["accepted"] is False

    def test_empty_batch_returns_empty(self):
        assert validate_upload([]) == []

    def test_result_has_required_keys(self):
        files = [(_make_pdf_bytes(), "x.pdf")]
        result = validate_upload(files)
        keys = set(result[0].keys())
        assert {"name", "accepted", "kind", "reason"} <= keys

    def test_accepted_file_kind_is_not_none(self):
        files = [(_make_pdf_bytes(), "x.pdf")]
        result = validate_upload(files)
        assert result[0]["kind"] is not None

    def test_rejected_file_reason_not_none(self):
        files = [(MZ_BYTES, "x.pdf")]
        result = validate_upload(files)
        assert result[0]["reason"] is not None


    def test_no_traceback_in_any_reason(self):
        files = [
            (MZ_BYTES, "evil.pdf"),
            (b"\x89PNG\r\n\x1a\n", "image.png"),
            (_make_pdf_bytes(extra=21 * 1024 * 1024), "big.pdf"),
        ]
        result = validate_upload(files)
        for r in result:
            if r["reason"]:
                assert "Traceback" not in r["reason"], (
                    f"Reason contains 'Traceback': {r['reason']!r}"
                )

    def test_no_api_key_in_any_reason(self):
        import os

        openai_key = os.environ.get("OPENROUTER_API_KEY", "secret-or")
        pinecone_key = os.environ.get("PINECONE_API_KEY", "secret-p")
        files = [(MZ_BYTES, "evil.pdf")]
        result = validate_upload(files)
        for r in result:
            if r["reason"]:
                assert openai_key not in r["reason"]
                assert pinecone_key not in r["reason"]

    def test_reads_max_files_from_settings(self):
        import inspect

        import app.services.validation as v_mod

        src = inspect.getsource(v_mod)
        assert "get_settings()" in src, "validation.py must call get_settings()"

    def test_reads_max_file_bytes_from_settings(self):
        import inspect

        import app.services.validation as v_mod

        src = inspect.getsource(v_mod)
        assert "max_file_bytes" in src, "validation.py must reference max_file_bytes"
