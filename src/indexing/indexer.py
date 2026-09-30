"""Discover source files, chunk their exact text, and build a BM25 index."""

import os
import pickle
import tempfile
from pathlib import Path

from tqdm import tqdm

from ..chunking import chunk_python_file, chunk_text, DEFAULT_MAX_CHUNK_SIZE
from ..retrieval.bm25 import Bm25
from ..models import Chunk
from ..io import read_source_file
from ..chunking import validate_options
from .file_index import find_files
from ..tokenizer import tokenize


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
        validate_options(max_size_chunks)
        self.path: str = path
        self.paths: list[Path] = find_files(path)
        self.max_size: int = max_size_chunks
        self.chunks: list[Chunk] = self.chunk_all_files()
        self.N = len(self.chunks)
        self.corpus = {
            doc_id: tokenize(doc.text) + tokenize(doc.file_path)
            for doc_id, doc in tqdm(
                enumerate(self.chunks), desc="Tokenizing chunks"
            )
        }
        self.doc_len = {
            doc_id: len(doc) for doc_id, doc in self.corpus.items()
        }
        self.avgdl = sum(self.doc_len.values()) / self.N if self.N else 0
        self.bm25: Bm25 = Bm25(
            self.corpus,
            self.avgdl,
            self.N,
            self.doc_len,
            self.chunks,
            k=k,
            b=b,
        )

    def chunk_all_files(self) -> list[Chunk]:
        """Chunk exact UTF-8 text, reporting unreadable files as skipped.

        Returns:
            Chunks with offsets into unmodified source text.
        """
        chunks: list[Chunk] = []
        for path in tqdm(self.paths, desc="Chunking the Files"):
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
        """Save an index atomically, creating its parent directory.

        Args:
            path: Destination index file.

        Raises:
            OSError: If the destination directory or index cannot be written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                pickle.dump(self.bm25.index_data, file)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
