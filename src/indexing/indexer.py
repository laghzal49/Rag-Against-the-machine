"""Discover source files, chunk their exact text, and build a BM25 index."""

import pickle
from pathlib import Path

from tqdm import tqdm

from ..chunking import chunk_python_file, chunk_text, DEFAULT_MAX_CHUNK_SIZE
from ..retrieval.bm25 import Bm25
from ..models import Chunk
from ..io import read_source_file
from ..chunking import _validate_options
from ..retrieval.tokenizer import tokenize


class Indexer:
    """Build a complete index from Python, Markdown, and text files."""

    def __init__(
        self,
        path: str,
        k: float = 1.5,
        b: float = 0.5,
        max_size_chunks: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> None:
        """Validate chunk sizing and index the requested source path.

        Args:
            path: Corpus directory or source file.
            k: Positive BM25 saturation parameter.
            b: BM25 length normalization between zero and one.
            max_size_chunks: Maximum characters per chunk, from 1 to 2000.

        Raises:
            ValueError: If the path, chunk size, or BM25 settings are invalid.
        """
        _validate_options(max_size_chunks)
        self.path: str = path
        self.paths: list[Path] = self.find_files(path)
        self.max_size: int = max_size_chunks
        # TODO: Persist source fingerprints and a versioned index format;
        # carry verification metadata through retrieval to context building.
        self.chunks: list[Chunk] = self.chunk_all_files()
        self.N = len(self.chunks)
        self.corpus = {
            doc_id: tokenize(doc.text) + tokenize(doc.file_path)
            for doc_id, doc in tqdm(
                enumerate(self.chunks), desc="Tokenizing chunks", unit="chunk"
            )
        }
        self.doc_len = {
            doc_id: len(doc) for doc_id, doc in self.corpus.items()
        }
        self.avgdl = sum(self.doc_len.values()) / self.N if self.N else 0
        self.bm25: Bm25 = Bm25(
            self.corpus, self.avgdl, self.N, self.doc_len,
            self.chunks, k=k, b=b,
        )

    def dfs_file(self, path: Path) -> list[Path]:
        """Walk files deterministically, skipping symbolic links.

        Args:
            path: File or directory at which traversal begins.

        Returns:
            Supported source file paths in deterministic traversal order.
        """
        data: list[Path] = []
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_symlink():
                    continue
                if child.is_dir():
                    data.extend(self.dfs_file(child))
                else:
                    if self.is_useful_file(child):
                        data.append(child)
        else:
            if self.is_useful_file(path):
                data.append(path)

        return data

    def is_useful_file(self, path: Path) -> bool:
        """Return whether the file has a supported text extension.

        Args:
            path: Candidate source file path.

        Returns:
            True for Python, Markdown, and plain-text extensions.
        """
        useful = [".py", ".txt", ".md"]
        return path.suffix in useful

    def find_files(self, directory: str) -> list[Path]:
        """Collect supported files or reject a nonexistent input path.

        Args:
            directory: Source directory or individual file.

        Returns:
            Supported source file paths.

        Raises:
            ValueError: If the input path does not exist.
        """
        path = Path(directory).resolve()
        try:
            path = path.relative_to(Path.cwd().resolve())
        except ValueError as error:
            raise ValueError("Corpus must be inside the project") from error
        if not path.exists():
            raise ValueError(f"Corpus path does not exist: {directory}")
        return self.dfs_file(path)

    def chunk_all_files(self) -> list[Chunk]:
        """Chunk exact UTF-8 text, reporting unreadable files as skipped.

        Returns:
            Chunks with offsets into unmodified source text.
        """
        chunks: list[Chunk] = []
        for path in tqdm(self.paths, desc="Chunking the Files:"):
            try:
                text = read_source_file(path)
            except (UnicodeError, OSError) as error:
                tqdm.write(f"Skipping {path}: {error}")
                continue
            if path.suffix == ".py":
                chunks.extend(
                    chunk_python_file(
                        file_path=str(path),
                        text=text,
                        max_chunk_size=self.max_size,
                    )
                )
            else:
                chunks.extend(
                    chunk_text(
                        file_path=str(path),
                        text=text,
                        max_chunk_size=self.max_size,
                    )
                )
        return chunks

    def save(self, path: str) -> None:
        """Save a local pickle directly, creating parent directories.

        Args:
            path: Destination index file.

        Raises:
            OSError: If the destination directory or index cannot be written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Write a temporary sibling file, then atomically replace the
        # target. Test that a failed save preserves the previous index.
        with open(target, "wb") as f:
            pickle.dump(self.bm25.index_data, f)
