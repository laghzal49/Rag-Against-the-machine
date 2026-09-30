from ..modele import RagDataset
from ..retrieval import multiple_question, validate_k, Bm25
from ..io import save_json, read_json_file
from pathlib import Path


def search_dataset(
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
