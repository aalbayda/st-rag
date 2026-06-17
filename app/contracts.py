
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class PdfLocator(BaseModel):

    kind: Literal["pdf"] = "pdf"
    page: int

    def label(self) -> str:
        return f"p. {self.page}"


class DocxLocator(BaseModel):

    kind: Literal["docx"] = "docx"
    section: str | None = None
    paragraph_index: int

    @property
    def page(self) -> None:
        return None

    def label(self) -> str:
        if self.section:
            return f"§ {self.section}"
        return f"¶ {self.paragraph_index}"


Locator = Annotated[
    Union[PdfLocator, DocxLocator],
    Field(discriminator="kind"),
]


class Citation(BaseModel):

    id: str

    file_id: str

    file_name: str

    locator: Locator

    chunk_text: str


class Answer(BaseModel):

    answer: str

    reasoning: str

    citations: list[Citation] = Field(default_factory=list)

    abstained: bool = False

    @model_validator(mode="after")
    def _enforce_abstained_invariant(self) -> "Answer":
        if self.abstained and self.citations:
            raise ValueError(
                "abstained=True requires citations to be empty"
            )
        return self


class ModelAnswer(BaseModel):

    answer: str

    reasoning: str

    cited_ids: list[str] = Field(default_factory=list)

    abstained: bool = False
