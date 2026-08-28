"""Persist and search a lexical BM25 index of source chunks."""

import math
import pickle
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

from tqdm import tqdm

from .modele import Chunk, MinimalSource
from .stop_word import STOP_WORD

PATTERN = r"\b\w+\b"


def validate_k(k: int) -> None:
    """Reject result limits that are not nonnegative integers."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")


class Bm25:
    """Rank chunks using term frequency and document length normalization."""

    def __init__(
        self, chunks: list[Chunk], k: float = 1.5, b: float = 0.5
    ) -> None:
        """Build postings from chunks with valid BM25 parameters.

        Args:
            chunks: Source passages to index.
            k: Positive term-frequency saturation parameter.
            b: Length normalization parameter between zero and one.
        """
        if k <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k > 0 and 0 <= b <= 1")
        self.chunks: list[Chunk] = chunks
        self.k1: float = k
        self.b: float = b
        self.N: int = len(self.chunks)
        self.corpus: list[list[str]] = [
            self.tokenize(doc.text) + self.tokenize(doc.file_path)
            for doc in tqdm(
                self.chunks, desc="Tokenizing chunks", unit="chunk"
            )
        ]
        self.avgdl: float = (
            sum(len(chunk) for chunk in self.corpus) / self.N if self.N else 0
        )
        self.doc_len: dict[int, int] = {
            i: len(doc) for i, doc in enumerate(self.corpus)
        }
        self.inverted: dict[str, dict[int, int]] = self.build_inverted_index()

    def tokenize(self, text: str) -> list[str]:
        """Extract lowercase words, excluding common English stopwords.

        Args:
            text: Source text, path, or query to analyze.

        Returns:
            Tokens in occurrence order, including repeated words.
        """
        tokens = []
        for token in re.findall(PATTERN, text):
            tokens.append(token.lower())
            expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
            parts = expanded.replace("_", " ").lower().split()
            if len(parts) > 1:
                tokens.extend(parts)
        new = []
        for token in tokens:
            if token in STOP_WORD:
                continue
            new.append(token)
        return new

    @property
    def index_data(self) -> dict[str, Any]:
        """Return the state needed to restore retrieval without rebuilding.

        Returns:
            Chunk, posting, length, and scoring parameter dictionary.
        """
        return {
            "chunks": self.chunks,
            "inverted": self.inverted,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "N": self.N,
        }

    def save_index(self, path: str) -> None:
        """Save a local pickle directly, creating parent directories.

        Args:
            path: Destination index file.

        Raises:
            OSError: If the destination directory or index cannot be written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self.index_data, f)

    def build_inverted_index(self) -> dict[str, dict[int, int]]:
        """Map each term to its frequency in each containing chunk.

        Returns:
            Term-to-postings mapping with document IDs and frequencies.
        """
        invert: defaultdict[str, dict[int, int]] = defaultdict(dict)
        for doc_id, doc in enumerate(
            tqdm(self.corpus, desc="Building index", unit="chunk")
        ):
            frequencies = Counter(doc)
            for word, f in frequencies.items():
                invert[word][doc_id] = f

        return invert

    def idf(self, term: str) -> float:
        """Compute smoothed inverse document frequency for a term.

        Args:
            term: An analyzed query token.

        Returns:
            Smoothed inverse document frequency.
        """
        df = len(self.inverted.get(term, {}))
        return math.log(1 + ((self.N - df + 0.5) / (df + 0.5)))

    def score(self, terms: list[str]) -> dict[int, float]:
        """Score matching chunks.

        Args:
            terms: Tokenized query terms.

        Returns:
            Scores keyed by chunk ID.
        """
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            idf = self.idf(term)
            inverted = self.inverted.get(term, {})
            for doc_id, frequency in inverted.items():
                normalization = self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl
                )
                scores[doc_id] += (
                    (frequency * (self.k1 + 1)) / (frequency + normalization)
                ) * idf
        return scores

    def search(self, query: str, top_k: int) -> list[MinimalSource]:
        """Return at most top_k spans, ranked by descending BM25 score.

        Args:
            query: Text to search for.
            top_k: Maximum number of results, including zero.

        Returns:
            Source spans ordered by descending relevance.

        Raises:
            ValueError: If top_k is a boolean, non-integer, or negative.
        """
        validate_k(top_k)
        if top_k == 0:
            return []
        terms = self.tokenize(query)
        scores = self.score(terms)
        ranked = sorted(
            scores.items(), key=lambda score: score[1], reverse=True
        )[:top_k]
        sources: list[MinimalSource] = []
        for i, _ in ranked:
            chunk = self.chunks[i]
            sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
        return sources

    @classmethod
    def load_index(cls, path: str) -> "Bm25":
        """Restore a trusted local pickle; never load untrusted indexes.

        Args:
            path: Index file previously written by this implementation.

        Returns:
            Restored BM25 search engine.

        Raises:
            OSError: If the index cannot be read.
            ValueError: If unpickling fails or required state keys are missing.
        """
        try:
            with open(path, "rb") as f:
                info = pickle.load(f)
        except OSError as error:
            raise OSError(f"Cannot read index {path}: {error}") from error
        except (
            pickle.UnpicklingError,
            ImportError,
            AttributeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"Cannot load index {path}: corrupt or incompatible file. "
                "Run `index` to rebuild it."
            ) from error
        required = {"chunks", "N", "avgdl", "inverted", "doc_len", "k1", "b"}
        if not isinstance(info, dict) or not required.issubset(info):
            raise ValueError(
                f"Invalid index structure in {path}. "
                "Run `index` to rebuild it."
            )
        obj = cls.__new__(cls)
        obj.chunks = info["chunks"]
        obj.N = info["N"]
        obj.avgdl = info["avgdl"]
        obj.inverted = info["inverted"]
        obj.doc_len = info["doc_len"]
        obj.k1 = info["k1"]
        obj.b = info["b"]
        return obj
