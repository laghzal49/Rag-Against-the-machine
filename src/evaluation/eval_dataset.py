from ..models import AnsweredQuestion, MinimalSearchResults
from ..models import UnansweredQuestion
from ..retrieval.bm25 import validate_k

from .eval import evaluation


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
