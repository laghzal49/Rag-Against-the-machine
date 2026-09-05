"""Adapt BM25 retrieval to the required question and dataset models."""

from tqdm import tqdm
from .bm25 import Bm25, validate_k
from .modele import (
    MinimalSource,
    MinimalSearchResults,
    UnansweredQuestion,
    StudentSearchResults,
    RagDataset,
)


class Retrieval:
    """Retrieve source spans for individual questions."""

    def __init__(self, bm25: Bm25) -> None:
        """Reuse an already built or loaded BM25 index.

        Args:
            bm25: Initialized BM25 search engine.
        """
        self.bm25: Bm25 = bm25

    def one_question(
        self, question: UnansweredQuestion, k: int
    ) -> MinimalSearchResults:
        """Retrieve up to k sources while preserving the question ID.

        Args:
            question: Input question and its identifier.
            k: Maximum number of sources to retrieve.

        Returns:
            Ranked sources with the original question text and ID.

        Raises:
            ValueError: If k is a boolean, non-integer, or negative.
        """
        retrieved: list[MinimalSource] = self.bm25.search(question.question, k)
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=retrieved,
        )

    def multiple_question(
        self, data_set: RagDataset, k: int
    ) -> StudentSearchResults:
        """Search each question in a dataset.

        Args:
            data_set: Input questions with their identifiers.
            k: Maximum sources per question.

        Returns:
            Validated batch results preserving input question order.
        """
        validate_k(k)
        seach_results: list[MinimalSearchResults] = []
        for question in tqdm(data_set.rag_questions):
            seach_results.append(self.one_question(question, k))
        output = StudentSearchResults(search_results=seach_results, k=k)
        return output
