from ..modele import Chunk

DEFAULT_MAX_CHUNK_SIZE = 2000
MAX_CHUNK_SIZE = 2000


def validate_options(max_chunk_size: int) -> None:
    """Require an integer chunk size between 1 and 2,000 characters.

    Args:
        max_chunk_size: Maximum characters allowed in one chunk.

    Raises:
        ValueError: If the size is a boolean, non-integer, or out of range.
    """
    if (
        isinstance(max_chunk_size, bool)
        or not isinstance(max_chunk_size, int)
        or not 1 <= max_chunk_size <= MAX_CHUNK_SIZE
    ):
        raise ValueError(
            f"max_chunk_size must be an integer between 1 and "
            f"{MAX_CHUNK_SIZE}; got {max_chunk_size!r}"
        )


def make_chunk(
    text: str, file_path: str, start: int, end: int, offset: int = 0
) -> Chunk:
    """Create a chunk from a character range in the original text.

    Args:
        text: Source text or a substring of the original file.
        file_path: Path associated with the source.
        start: Included character offset.
        end: Excluded character offset.
        offset: Original-file character offset of the supplied text.

    Returns:
        A chunk containing ``text[start:end]``.
    """
    return Chunk(
        file_path=file_path,
        first_character_index=start + offset,
        last_character_index=end + offset,
        text=text[start:end],
    )
