"""Expose indexing, retrieval, generation, and evaluation through Fire."""

from .models import RagDataset
from .indexing import Indexer
from .retrieval.bm25 import Bm25, validate_k
from pathlib import Path
from .generation import Answer
from .retrieval import one_question, multiple_question
from .models import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
    StudentSearchResults,
)
from .evaluation import evaluate_dataset
from .io import read_json_file, save_json


class Cli:
    """Provide the six command-line operations for the RAG pipeline."""

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed/index.pkl",
    ) -> StudentSearchResults:
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
        return output

    def index(
        self,
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

    def answer(
        self,
        question: str,
        k: int = 5,
        index_path: str = "data/processed/index.pkl",
    ) -> None:
        """Retrieve evidence and print a locally generated answer.

        Args:
            question: Question to answer from retrieved evidence.
            k: Maximum number of evidence sources.
            index_path: Path to a trusted local BM25 index.

        Raises:
            FileNotFoundError: If the index does not exist.
            ValueError: If k, source spans, or token budget are invalid.
        """
        validate_k(k)
        index = Path(index_path)
        if not index.exists():
            raise FileNotFoundError(
                f"Index not found: {index}. Run `index` first."
            )
        bm25 = Bm25.load_index(str(index_path))
        result = one_question(
            UnansweredQuestion(question=str(question)), k, bm25
        )
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

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 5,
    ) -> None:
        """Save ranked sources for every question, preserving input IDs.

        Args:
            dataset_path: Input question dataset JSON file.
            save_directory: Output directory; keeps the input basename.
            k: Maximum sources per question.
            index_path: Path to a trusted local BM25 index.

        Raises:
            FileNotFoundError: If an input file does not exist.
            ValueError: If the dataset, index, or k is invalid, or question
                IDs are duplicated.
            OSError: If input files cannot be read or results cannot be saved.
        """
        validate_k(k)
        index_path: str = "data/processed/index.pkl"
        index = Path(index_path)
        if not index.exists():
            raise FileNotFoundError(
                f"Index not found: {index}. Run `index` first."
            )
        bm25 = Bm25.load_index(str(index_path))
        data = read_json_file(dataset_path, RagDataset)
        ids = [question.question_id for question in data.rag_questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate question IDs in input dataset")
        output = multiple_question(data_set=data, k=k, bm25=bm25)
        save_json(save_directory, output, dataset_path)

    def answer_dataset(
        self,
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
        search = read_json_file(
            student_search_results_path, StudentSearchResults
        )
        answer = Answer()
        results = answer.generate_dataset(results=search)
        save_json(save_directory, results, student_search_results_path)

    def evaluate(
        self,
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
        search = read_json_file(
            student_search_results_path, StudentSearchResults
        )
        data = read_json_file(dataset_path, RagDataset)
        result = evaluate_dataset(
            data.rag_questions,
            search.search_results,
            search.k if k is None else k,
        )
        print(f"Recall @{search.k if k is None else k} = {result * 100}%")
