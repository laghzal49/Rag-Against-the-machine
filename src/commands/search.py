from pathlib import Path

from ..modele import StudentSearchResults, UnansweredQuestion
from ..retrieval import Bm25, one_question, validate_k


def search(
    query: str,
    k: int = 5,
    index_path: str = "data/processed/index.pkl",
) -> None:
    """Print the highest-ranked source spans for a query.

    Args:
        query: Natural-language search query.
        k: Maximum number of sources to return.
        index_path: Path to a trusted local BM25 index.

    Raises:
        FileNotFoundError: If the index does not exist.
        ValueError: If k is not a nonnegative integer or the index
            cannot be restored.
    """
    validate_k(k)
    index = Path(index_path)
    if not index.exists():
        raise FileNotFoundError(
            f"Index not found: {index}. Run `index` first."
        )
    bm25 = Bm25.load_index(str(index_path))
    result = one_question(UnansweredQuestion(question=str(query)), k, bm25)
    output = StudentSearchResults(search_results=[result], k=k)
    print(output.model_dump_json(indent=2))
