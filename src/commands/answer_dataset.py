from ..generation import Answer
from ..io import save_json, read_json_file
from ..modele import StudentSearchResults


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
) -> None:
    """Generate and save answers from previously retrieved sources.

    Args:
        student_search_results_path: Validated retrieval results JSON.
        save_directory: Output directory; keeps the input basename.

    Raises:
        ValueError: If input data, source spans, or budget are invalid.
        OSError: If source files or output cannot be accessed.
    """
    search = read_json_file(student_search_results_path, StudentSearchResults)
    answer = Answer()
    results = answer.generate_dataset(results=search)
    save_json(save_directory, results, student_search_results_path)
