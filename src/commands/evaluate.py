from ..modele import StudentSearchResults, RagDataset
from ..io import read_json_file
from ..evaluation import evaluate_dataset


def evaluate(
    student_search_results_path: str,
    dataset_path: str,
    k: int | None = None,
) -> None:
    """Print macro recall at k, defaulting to the saved search limit.

    Args:
        student_search_results_path: Predicted retrieval results JSON.
        dataset_path: Dataset containing reference sources.
        k: Source limit, or None to use the saved search limit.

    Raises:
        ValueError: If inputs are invalid or have no labeled questions.
    """
    search = read_json_file(student_search_results_path, StudentSearchResults)
    data = read_json_file(dataset_path, RagDataset)
    result = evaluate_dataset(
        data.rag_questions,
        search.search_results,
        search.k if k is None else k,
    )
    print(
        f"Recall @{search.k if k is None else k} = {round(result * 100, 2)}%"
    )
