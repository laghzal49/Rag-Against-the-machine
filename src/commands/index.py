from ..indexing import Indexer


def index(
    max_chunk_size: int = 2000,
    index_path: str = "data/processed/index.pkl",
    data_path: str = "data/raw/",
) -> None:
    """Rebuild and save an index from the requested corpus.

    Args:
        max_chunk_size: Maximum characters per chunk, from 1 to 2000.
        index_path: Destination for the persisted index.
        data_path: Source file or directory to index.

    Raises:
        ValueError: If the source path or chunk size is invalid.
        OSError: If the index cannot be saved.
    """
    indexer = Indexer(max_size_chunks=max_chunk_size, path=data_path)
    indexer.save(str(index_path))
    print(f"index saved under {index_path}")
