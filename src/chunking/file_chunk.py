"""Chunk Python, Markdown, and text files for indexing."""

from ..models import Chunk
from .utils import DEFAULT_MAX_CHUNK_SIZE, validate_options, make_chunk


def chunk_text(
    text: str,
    file_path: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    offset: int = 0,
) -> list[Chunk]:
    """Split text into bounded, overlapping character windows.

    Adjacent windows overlap by min(200, max_chunk_size // 5) characters.

    Args:
        text: File contents to split.
        file_path: Path stored with each chunk.
        max_chunk_size: Maximum characters per chunk, from 1 to 2,000.
        offset: Original-file character offset of the supplied text.

    Returns:
        Chunks in source order, or an empty list for empty text.

    Raises:
        ValueError: If max_chunk_size is not an integer from 1 to 2,000.
    """
    validate_options(max_chunk_size)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(make_chunk(text, file_path, start, end, offset))

        if end == len(text):
            break

        start = end - min(200, max_chunk_size // 5)

    return chunks
