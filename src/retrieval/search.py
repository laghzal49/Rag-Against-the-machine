"""Adapt BM25 retrieval to the required question and dataset models."""

from tqdm import tqdm
from ..retrieval.bm25 import Bm25, validate_k
from ..models import (
    MinimalSource,
    MinimalSearchResults,
    UnansweredQuestion,
    StudentSearchResults,
    RagDataset,
)


def one_question(
    question: UnansweredQuestion, k: int, bm25: Bm25
) -> MinimalSearchResults:
    """Retrieve up to k sources while preserving the question ID.

    Args:
        question: Input question and its identifier.
        k: Maximum number of sources to retrieve.
        bm25: Initialized BM25 search engine.

    Returns:
        Ranked sources with the original question text and ID.

    Raises:
        ValueError: If k is a boolean, non-integer, or negative.
    """
    retrieved: list[MinimalSource] = bm25.search(question.question, k)
    return MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=retrieved,
    )


def multiple_question(
    data_set: RagDataset, k: int, bm25: Bm25
) -> StudentSearchResults:
    """Search each question in a dataset.

    Args:
        data_set: Input questions with their identifiers.
        k: Maximum sources per question.
        bm25: Initialized BM25 search engine.

    Returns:
        Validated batch results preserving input question order.
    """
    validate_k(k)
    search_results: list[MinimalSearchResults] = []
    for question in tqdm(data_set.rag_questions):
        search_results.append(one_question(question, k, bm25))
    output = StudentSearchResults(search_results=search_results, k=k)
    return output
