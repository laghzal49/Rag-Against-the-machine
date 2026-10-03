from ..modele import MinimalSource
from .metrics import calculate_iou


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
    for answ in answer:
        for unasn in unasnwer:
            if calculate_iou(answ, unasn) > 0.05:
                scores += 1
                break
    len_answers = len(answer)
    return max(0, scores / len_answers)
