"""
Provide character-level overlap utilities for source retrieval evaluation."""

from ..modele import MinimalSource


def _intersection(first: MinimalSource, second: MinimalSource) -> float:
    """Calculate the number of overlapping characters between two spans.

    Args:
        first: First source span.
        second: Second source span.

    Returns:
        The length of the overlapping character range, or 0 if the spans
        do not overlap.
    """
    overlap_start = max(
        first.first_character_index, second.first_character_index
    )
    overlap_end = min(first.last_character_index, second.last_character_index)
    return max(0, overlap_end - overlap_start)


def _union(first: MinimalSource, second: MinimalSource, inter: float) -> float:
    """Calculate the total character coverage of two spans.

    Args:
        first: First source span.
        second: Second source span.
        inter: Length of the intersection between the two spans.

    Returns:
        The number of unique characters covered by either span.
    """
    first_len = first.last_character_index - first.first_character_index
    second_len = second.last_character_index - second.first_character_index

    length = first_len + second_len
    return length - inter


def calculate_iou(first: MinimalSource, second: MinimalSource) -> float:
    """Calculate character-level Intersection over Union (IoU) for two spans.

    The spans are compared only when they belong to the same file.

    Args:
        first: First source span.
        second: Second source span.

    Returns:
        The IoU value in the range [0.0, 1.0]. Returns 0.0 when the spans
        belong to different files or their union has zero length.
    """
    if first.file_path != second.file_path:
        return 0.0

    inter = _intersection(first, second)
    union = _union(first, second, inter)

    if union != 0:
        return inter / union

    return 0.0
