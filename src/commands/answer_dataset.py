from ..generation import Answer
from ..io import save_json, read_json_file
from ..modele import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


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
    if not any(
        item.retrieved_sources and item.question.strip()
        for item in search.search_results
    ):
        results = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer="Insufficient retrieved evidence.",
                )
                for item in search.search_results
            ],
            k=search.k,
        )
        save_json(save_directory, results, student_search_results_path)
        return
    answer = Answer()
    results = answer.generate_dataset(results=search)
    save_json(save_directory, results, student_search_results_path)
