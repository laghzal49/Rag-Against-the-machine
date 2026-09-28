"""Data models for RAG datasets and retrieval results."""


import uuid
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject input fields that are not declared in the model."""

    model_config = ConfigDict(extra="forbid")


class MinimalSource(StrictModel):
    """Locate a passage using a file path and character offsets.

    The start is included; the end is excluded:
    passage = file_text[first_character_index:last_character_index]
    """

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        """Require a nonempty range of at most 2,000 characters.

        Returns:
            The validated source instance.

        Raises:
            ValueError: If the span is empty, reversed, or too long.
        """

        span_length = self.last_character_index - self.first_character_index

        if span_length == 0:
            raise ValueError("The source range cannot be empty")

        if span_length < 0:
            raise ValueError(
                "last_character_index must be greater than "
                "first_character_index"
            )

        if span_length > 2000:
            raise ValueError(
                "A source cannot contain more than 2000 characters"
            )

        return self


class UnansweredQuestion(StrictModel):
    """Represent an input question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with an answer and supporting sources."""

    model_config = ConfigDict(extra="ignore")
    sources: list[MinimalSource]
    answer: str


class RagDataset(StrictModel):
    """Hold answered questions, unanswered questions, or both."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(StrictModel):
    """Hold ranked sources retrieved for one question.

    Sources are ordered from highest to lowest relevance.
    The question ID must be copied from the input question.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Hold retrieval results together with a generated answer."""

    answer: str


class StudentSearchResults(StrictModel):
    """Hold retrieval results for a collection of questions.

    k is the requested number of sources per question.
    A result may contain fewer than k sources.
    """

    search_results: list[MinimalSearchResults]
    k: int = Field(ge=0)


class StudentSearchResultsAndAnswer(StrictModel):
    """Hold generated answers for a collection of questions."""

    search_results: list[MinimalAnswer]
    k: int = Field(ge=0)


class Chunk(BaseModel):
    """Represent a source passage and its searchable text."""

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def validate_text_length(self) -> Self:
        """Ensure text length matches the recorded character range.

        Returns:
            The validated chunk instance.

        Raises:
            ValueError: If text length differs from the recorded span.
        """

        span_length = self.last_character_index - self.first_character_index

        if len(self.text) != span_length:
            raise ValueError("Text length must match the source range length")

        return self
