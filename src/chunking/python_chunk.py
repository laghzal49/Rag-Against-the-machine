import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node
from .file_chunk import chunk_text
from .utils import Chunk, DEFAULT_MAX_CHUNK_SIZE, make_chunk, validate_options

PYTHON_LANGUAGE = Language(ts_python.language())
PYTHON_PARSER = Parser(PYTHON_LANGUAGE)


def _give_byte_to_len(raw: bytes, node: Node) -> tuple[int, int]:
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
    validate_options(max_chunk_size)
    raw = text.encode("utf-8")
    root = PYTHON_PARSER.parse(raw).root_node
    chunks = []
    starts: list[int] = []
    prev_end = 0
    for node in root.named_children:
        start, end = _give_byte_to_len(raw, node)
        starts.append(start)
        group_start = starts[0]
        if end - group_start > max_chunk_size and prev_end - group_start > 0:
            chunks.append(make_chunk(text, file_path, group_start, prev_end))
            starts = []
            starts.append(start)
        if end - start > max_chunk_size:
            chunks.extend(
                chunk_text(text[start:end], file_path, max_chunk_size, start)
            )
            starts = []
        prev_end = end
    if starts and prev_end - starts[0]:
        chunks.append(make_chunk(text, file_path, starts[0], prev_end))

    return chunks
