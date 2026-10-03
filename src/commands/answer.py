from ..modele import (
    UnansweredQuestion,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from ..retrieval import one_question, validate_k, Bm25
from ..generation import Answer
from pathlib import Path
from .index import INDEX_PATH


def answer(
    question: str,
    k: int = 5,
) -> None:
    """Retrieve evidence and print a locally generated answer.

    Args:
        question: Question to answer from retrieved evidence.
        k: Maximum number of evidence sources.
    Raises:
        FileNotFoundError: If the index does not exist.
        ValueError: If k, source spans, or token budget are invalid.
    """
    validate_k(k)
    index = Path(INDEX_PATH)
    if not index.exists():
        raise FileNotFoundError(
            f"Index not found: {index}. Run `index` first."
        )
    bm25 = Bm25.load_index(str(INDEX_PATH))
    result = one_question(UnansweredQuestion(question=str(question)), k, bm25)
    if not result.retrieved_sources or not question.strip():
        output = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer="Insufficient retrieved evidence.",
                )
            ],
            k=k,
        )
    else:
        output = Answer().generate_dataset(
            StudentSearchResults(search_results=[result], k=k)
        )
    print(output.model_dump_json(indent=2))
