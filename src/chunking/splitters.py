"""Chunk Python, Markdown, and text files for indexing."""

import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node

from ..models import Chunk


DEFAULT_MAX_CHUNK_SIZE = 2000
MAX_CHUNK_SIZE = 2000

PYTHON_LANGUAGE = Language(ts_python.language())
PYTHON_PARSER = Parser(PYTHON_LANGUAGE)


def _validate_options(max_chunk_size: int) -> None:
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


def _make_chunk(
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
    _validate_options(max_chunk_size)
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(_make_chunk(text, file_path, start, end, offset))

        if end == len(text):
            break

        start = end - min(200, max_chunk_size // 5)

    return chunks


def give_byte_to_len(raw: bytes, node: Node) -> tuple[int, int]:
    """Convert a Tree-sitter node's byte range to character offsets.

    Args:
        raw: Complete UTF-8 encoded source parsed by Tree-sitter.
        node: Syntax node whose byte offsets refer to raw.

    Returns:
        The inclusive start and exclusive end in Unicode characters.

    Raises:
        UnicodeDecodeError: If a decoded prefix is not valid UTF-8.
    """
    start = raw[: node.start_byte].decode("utf-8")
    end = raw[: node.end_byte].decode("utf-8")
    return len(start), len(end)


def chunk_python_file(
    text: str, file_path: str, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> list[Chunk]:
    """Group top-level syntax nodes and split oversized nodes into windows.

    Whitespace between grouped nodes is retained. Leading, trailing, and
    inter-group whitespace is not necessarily included in the chunks.

    Args:
        text: Python source text to parse with Tree-sitter.
        file_path: Path stored with each chunk.
        max_chunk_size: Maximum characters per chunk, from 1 to 2,000.

    Returns:
        Chunks in syntax-node order with half-open character offsets.
        Text without named syntax nodes produces an empty list.

    Raises:
        ValueError: If max_chunk_size is not an integer from 1 to 2,000.
        UnicodeError: If the source cannot be encoded or decoded as UTF-8.
    """
    _validate_options(max_chunk_size)
    raw = text.encode("utf-8")
    root = PYTHON_PARSER.parse(raw).root_node
    chunks = []
    starts: list[int] = []
    prev_end = 0
    for node in root.named_children:
        start, end = give_byte_to_len(raw, node)
        starts.append(start)
        group_start = starts[0]
        if end - group_start > max_chunk_size and prev_end - group_start > 0:
            chunks.append(_make_chunk(text, file_path, group_start, prev_end))
            starts = []
            starts.append(start)
        if end - start > max_chunk_size:
            chunks.extend(
                chunk_text(text[start:end], file_path, max_chunk_size, start)
            )
            starts = []
        prev_end = end
    if starts and prev_end - starts[0]:
        chunks.append(_make_chunk(text, file_path, starts[0], prev_end))

    return chunks
