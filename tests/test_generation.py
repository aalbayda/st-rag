
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "secret-or")
os.environ.setdefault("PINECONE_API_KEY", "secret-p")
os.environ.setdefault("PINECONE_INDEX_NAME", "rag-dense")

from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_caches():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _pdf_meta(
    file_id: str = "f1",
    file_name: str = "report.pdf",
    chunk_text: str = "Chunk text here.",
    page: int = 3,
) -> dict:
    return {
        "file_id": file_id,
        "file_name": file_name,
        "chunk_text": chunk_text,
        "kind": "pdf",
        "page": page,
    }


def _docx_meta(
    file_id: str = "f2",
    file_name: str = "report.docx",
    chunk_text: str = "DOCX chunk text.",
    section: str | None = "Introduction",
    paragraph_index: int = 5,
) -> dict:
    return {
        "file_id": file_id,
        "file_name": file_name,
        "chunk_text": chunk_text,
        "kind": "docx",
        "section": section,
        "paragraph_index": paragraph_index,
    }


def _model_answer(
    answer: str = "The answer is X [1].",
    reasoning: str = "Based on passage [1].",
    cited_ids: list[str] | None = None,
    abstained: bool = False,
):
    from app.contracts import ModelAnswer

    if cited_ids is None:
        cited_ids = ["[1]"]
    return ModelAnswer(
        answer=answer,
        reasoning=reasoning,
        cited_ids=cited_ids,
        abstained=abstained,
    )


class TestD02ZeroResult:

    def test_import_cleanly(self):
        from app.services.generation import answer_question  # noqa: F401

    def test_zero_result_returns_abstained(self):
        from app.contracts import Answer
        from app.services.generation import answer_question

        with patch("app.services.generation.retrieve", return_value=("", {})), patch(
            "app.services.generation.generate_answer"
        ) as mock_gen:
            result = answer_question("What is X?")

        assert isinstance(result, Answer)
        assert result.abstained is True
        assert result.citations == []

    def test_zero_result_does_not_call_generate_answer(self):
        from app.services.generation import answer_question

        mock_gen = MagicMock()

        with (
            patch("app.services.generation.retrieve", return_value=("", {})),
            patch("app.services.generation.generate_answer", mock_gen),
        ):
            answer_question("What is X?")

        mock_gen.assert_not_called()

    def test_zero_result_has_reasoning(self):
        from app.services.generation import answer_question

        with (
            patch("app.services.generation.retrieve", return_value=("", {})),
            patch("app.services.generation.generate_answer"),
        ):
            result = answer_question("question")

        assert result.reasoning


class TestD08NoneAbstain:

    def test_none_returns_abstained(self):
        from app.contracts import Answer
        from app.services.generation import answer_question

        context = "[1] Some text"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=None),
        ):
            result = answer_question("What?")

        assert isinstance(result, Answer)
        assert result.abstained is True
        assert result.citations == []

    def test_none_returns_well_formed_answer(self):
        from app.services.generation import answer_question

        context = "[1] Text"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=None),
        ):
            result = answer_question("question")

        assert hasattr(result, "answer")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "citations")
        assert hasattr(result, "abstained")


class TestD01ModelJudgedAbstain:

    def test_model_abstained_returns_abstained_answer(self):
        from app.contracts import Answer
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="I cannot answer from the provided documents.",
            reasoning="No relevant information found.",
            cited_ids=[],
            abstained=True,
        )

        context = "[1] Text"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("What?")

        assert isinstance(result, Answer)
        assert result.abstained is True
        assert result.citations == []

    def test_model_abstained_uses_model_reasoning(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="",
            reasoning="Specific reasoning from model.",
            cited_ids=[],
            abstained=True,
        )

        context = "[1] Text"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert "Specific reasoning from model." in result.reasoning


class TestGEN04CitationValidation:

    def test_fabricated_id_dropped(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer [1] and also [99].",
            reasoning="Based on [1].",
            cited_ids=["[1]", "[99]"],
        )

        context = "[1] Real chunk"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert len(result.citations) == 1
        assert result.citations[0].id == "[1]"

    def test_orphaned_marker_stripped_from_answer_text(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer [1] and also [99].",
            reasoning="Based on [1].",
            cited_ids=["[1]", "[99]"],
        )

        context = "[1] Real chunk"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert "[99]" not in result.answer

    def test_surviving_marker_not_renumbered(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer [1] and also [99].",
            reasoning="Based on [1].",
            cited_ids=["[1]", "[99]"],
        )

        context = "[1] Real chunk"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert "[1]" in result.answer

    def test_all_fabricated_ids_returns_abstained_implicitly(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer [99] and [100].",
            reasoning="Based on context.",
            cited_ids=["[99]", "[100]"],
        )

        context = "[1] Real chunk"
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert len(result.citations) == 0
        assert "[99]" not in result.answer
        assert "[100]" not in result.answer


class TestEnrichment:

    def test_pdf_meta_enriches_to_pdf_locator(self):
        from app.contracts import PdfLocator
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer [1].",
            reasoning="Based on [1].",
            cited_ids=["[1]"],
        )

        context = "[1] PDF chunk"
        id_to_meta = {"[1]": _pdf_meta(page=7)}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert len(result.citations) == 1
        citation = result.citations[0]
        assert isinstance(citation.locator, PdfLocator)
        assert citation.locator.page == 7
        assert citation.locator.kind == "pdf"

    def test_docx_meta_enriches_to_docx_locator_no_keyerror(self):
        from app.contracts import DocxLocator
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer [1].",
            reasoning="Based on [1].",
            cited_ids=["[1]"],
        )

        context = "[1] DOCX chunk"
        id_to_meta = {"[1]": _docx_meta(section="Methods", paragraph_index=12)}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert len(result.citations) == 1
        citation = result.citations[0]
        assert isinstance(citation.locator, DocxLocator)
        assert citation.locator.section == "Methods"
        assert citation.locator.paragraph_index == 12
        assert citation.locator.page is None

    def test_docx_meta_none_section_enriches_correctly(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(cited_ids=["[1]"])
        id_to_meta = {"[1]": _docx_meta(section=None, paragraph_index=3)}

        with (
            patch("app.services.generation.retrieve", return_value=("[1] text", id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert len(result.citations) == 1
        assert result.citations[0].locator.section is None

    def test_citation_fields_from_metadata_not_model(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(cited_ids=["[1]"])
        meta = _pdf_meta(file_id="trusted-id", file_name="trusted.pdf", chunk_text="Trusted chunk.")
        id_to_meta = {"[1]": meta}

        with (
            patch("app.services.generation.retrieve", return_value=("[1] Trusted chunk.", id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        citation = result.citations[0]
        assert citation.file_id == "trusted-id"
        assert citation.file_name == "trusted.pdf"
        assert citation.chunk_text == "Trusted chunk."


class TestHappyPath:

    def test_happy_path_returns_non_abstained_answer(self):
        from app.contracts import Answer
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer is X [1].",
            reasoning="Based on passage [1] which says X.",
            cited_ids=["[1]"],
        )

        context = "[1] X is the answer."
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=(context, id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("What is X?")

        assert isinstance(result, Answer)
        assert result.abstained is False
        assert len(result.citations) == 1
        assert result.answer
        assert result.reasoning

    def test_happy_path_answer_contains_inline_marker(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(
            answer="The answer is X [1].",
            cited_ids=["[1]"],
        )

        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=("[1] X", id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert "[1]" in result.answer

    def test_happy_path_citation_id_matches_inline_marker(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(answer="Based on [1].", cited_ids=["[1]"])
        id_to_meta = {"[1]": _pdf_meta()}

        with (
            patch("app.services.generation.retrieve", return_value=("[1] text", id_to_meta)),
            patch("app.services.generation.generate_answer", return_value=model_ans),
        ):
            result = answer_question("question")

        assert result.citations[0].id == "[1]"


class TestFormatHistory:

    def test_empty_list_returns_empty_string(self):
        from app.services.generation import _format_history

        assert _format_history([]) == ""

    def test_user_role_labelled(self):
        from app.services.generation import _format_history

        result = _format_history([{"role": "user", "content": "hi"}])
        assert "User:" in result

    def test_assistant_role_labelled(self):
        from app.services.generation import _format_history

        result = _format_history([{"role": "assistant", "content": "hello"}])
        assert "Assistant:" in result

    def test_content_appears_in_output(self):
        from app.services.generation import _format_history

        result = _format_history([
            {"role": "user", "content": "what is X?"},
            {"role": "assistant", "content": "X is Y."},
        ])
        assert "what is X?" in result
        assert "X is Y." in result


class TestAnswerQuestionHistory:

    def _retrieve(self):
        context_block = "[1] Some chunk text."
        id_to_meta = {
            "[1]": {
                "kind": "pdf",
                "page": 1,
                "file_id": "f1",
                "file_name": "doc.pdf",
                "chunk_text": "Some chunk text.",
            }
        }
        return context_block, id_to_meta

    def test_no_history_backward_compat(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(cited_ids=["[1]"])
        with patch("app.services.generation.retrieve", return_value=self._retrieve()), patch(
            "app.services.generation.generate_answer", return_value=model_ans
        ) as mock_gen:
            result = answer_question("what is X?")

        assert result.abstained is False
        user_message_arg = mock_gen.call_args[0][1]
        assert "Prior conversation" not in user_message_arg

    def test_with_history_includes_prior_turns(self):
        from app.services.generation import answer_question

        history = [
            {"role": "user", "content": "first q"},
            {"role": "assistant", "content": "first a"},
        ]
        model_ans = _model_answer(cited_ids=["[1]"])
        with patch("app.services.generation.retrieve", return_value=self._retrieve()), patch(
            "app.services.generation.generate_answer", return_value=model_ans
        ) as mock_gen:
            answer_question("follow up", history=history)

        user_message_arg = mock_gen.call_args[0][1]
        assert "first q" in user_message_arg
        assert "first a" in user_message_arg

    def test_empty_history_no_prefix(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(cited_ids=["[1]"])
        with patch("app.services.generation.retrieve", return_value=self._retrieve()), patch(
            "app.services.generation.generate_answer", return_value=model_ans
        ) as mock_gen:
            answer_question("q", history=[])

        user_message_arg = mock_gen.call_args[0][1]
        assert "Prior conversation" not in user_message_arg

    def test_none_history_no_prefix(self):
        from app.services.generation import answer_question

        model_ans = _model_answer(cited_ids=["[1]"])
        with patch("app.services.generation.retrieve", return_value=self._retrieve()), patch(
            "app.services.generation.generate_answer", return_value=model_ans
        ) as mock_gen:
            answer_question("q", history=None)

        user_message_arg = mock_gen.call_args[0][1]
        assert "Prior conversation" not in user_message_arg
