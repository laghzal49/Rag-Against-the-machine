"""Evaluate source retrieval using character overlap and macro recall."""

from ..models import AnsweredQuestion, MinimalSearchResults, MinimalSource
from ..models import UnansweredQuestion
from ..retrieval.bm25 import validate_k


def calcultion(first: MinimalSource, second: MinimalSource) -> float:
    """Return character intersection over union for same-path spans.

    Args:
        first: Reference source span.
        second: Predicted source span.

    Returns:
        Intersection over union, or zero for distinct paths.
    """
    if first.file_path != second.file_path:
        return 0
    overlap_start = max(
        first.first_character_index,
        second.first_character_index,
    )
    overlap_end = min(first.last_character_index, second.last_character_index)
    overlap = max(0, overlap_end - overlap_start)
    first_len = first.last_character_index - first.first_character_index
    second_len = second.last_character_index - second.first_character_index
    s = first_len + second_len - overlap
    if s != 0:
        return overlap / s
    return 0.0


def evaluation(
    answer: list[MinimalSource],
    unasnwer: list[MinimalSource],
) -> float:
    """Return the fraction of references matched at IoU > 0.05.

    Args:
        answer: Reference source spans for a question.
        unasnwer: Predicted source spans in ranking order.

    Returns:
        Recall between zero and one, counting each reference once.
    """
    if not answer or not unasnwer:
        return 0
    scores = 0
    lent = len(answer)
    for i in range(lent):
        for j in range(len(unasnwer)):
            if calcultion(answer[i], unasnwer[j]) > 0.05:
                scores += 1
                break

    return max(0, scores / lent)


def evaluate_dataset(
    asnswers: list[AnsweredQuestion | UnansweredQuestion],
    unasnwer: list[MinimalSearchResults],
    k: int | None = None,
) -> float:
    """Average labeled question recall, counting missing results as zero.

    Args:
        asnswers: Reference questions, optionally including unlabeled ones.
        unasnwer: Predicted search results keyed by question ID.
        k: Maximum predictions to inspect, or None for all predictions.

    Returns:
        Macro recall over questions with nonempty reference sources.

    Raises:
        ValueError: If k is not a nonnegative integer or None, question
            IDs repeat, or no questions have reference sources.
    """
    if k is not None:
        validate_k(k)
    for questions in (asnswers, unasnwer):
        ids = [question.question_id for question in questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate question IDs in evaluation input")
    labeled = [
        question
        for question in asnswers
        if isinstance(question, AnsweredQuestion) and question.sources
    ]
    if not labeled:
        raise ValueError("Evaluation requires reference sources")
    predictions = {item.question_id: item for item in unasnwer}
    total = 0.0
    for question in labeled:
        prediction = predictions.get(question.question_id)
        if prediction is not None:
            total += evaluation(
                question.sources, prediction.retrieved_sources[:k]
            )
    return total / len(labeled)
