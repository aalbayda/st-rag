
from __future__ import annotations

import pytest

from app.services.ids import make_file_id


class TestMakeFileId:


    def test_deterministic_same_name(self):
        assert make_file_id("Report 2024.pdf") == make_file_id("Report 2024.pdf")

    def test_deterministic_across_multiple_calls(self):
        name = "my_document_v3.docx"
        ids = [make_file_id(name) for _ in range(3)]
        assert ids[0] == ids[1] == ids[2]


    def test_distinct_different_names(self):
        assert make_file_id("a.pdf") != make_file_id("b.pdf")

    def test_distinct_similar_names(self):
        assert make_file_id("file1.pdf") != make_file_id("file2.pdf")


    def test_no_hash_char_simple(self):
        assert "#" not in make_file_id("x.pdf")

    def test_no_hash_char_various(self):
        names = [
            "Report 2024.pdf",
            "data-export.docx",
            "Résumé François.pdf",
            "path/to/file.pdf",
            "file#weird.pdf",
        ]
        for name in names:
            result = make_file_id(name)
            assert "#" not in result, f"'#' found in make_file_id({name!r}) = {result!r}"


    def test_length_within_limit_normal(self):
        assert len(make_file_id("document.pdf")) <= 512

    def test_length_within_limit_very_long_name(self):
        long_name = "x" * 1000 + ".pdf"
        result = make_file_id(long_name)
        assert len(result) <= 512, (
            f"ID length {len(result)} exceeds 512 for long filename"
        )


    def test_no_path_separator_forward_slash(self):
        assert "/" not in make_file_id("dir/file.pdf")

    def test_no_path_separator_backslash(self):
        assert "\\" not in make_file_id("dir\\file.pdf")

    def test_no_dotdot(self):
        assert ".." not in make_file_id("../../etc/passwd.pdf")

    def test_no_spaces(self):
        assert " " not in make_file_id("my report 2024.pdf")


    def test_unicode_filename_safe(self):
        result = make_file_id("Résumé François.pdf")
        assert "#" not in result
        assert "/" not in result
        assert " " not in result
        assert len(result) <= 512

    def test_unicode_deterministic(self):
        assert make_file_id("文件名.pdf") == make_file_id("文件名.pdf")


    def test_returns_string(self):
        assert isinstance(make_file_id("report.pdf"), str)

    def test_nonempty(self):
        assert len(make_file_id("report.pdf")) > 0
