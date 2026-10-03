from ..indexing import Indexer
from ..indexing.corpus import ensure_default_corpus

INDEX_PATH = "data/processed/index.pkl"


def index(
    max_chunk_size: int = 2000,
    data_path: str = "data/raw/",
) -> None:
    """Rebuild and save an index from the requested corpus.

    Args:
        max_chunk_size: Maximum characters per chunk, from 1 to 2000.
        data_path: Source file or directory to index.

    Raises:
        ValueError: If the source path or chunk size is invalid.
        OSError: If the index cannot be saved.
    """
    ensure_default_corpus(data_path)
    indexer = Indexer(max_size_chunks=max_chunk_size, path=data_path)
    indexer.save(str(INDEX_PATH))
    print(f"index saved under {INDEX_PATH}")
