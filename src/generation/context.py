"""Build validated source context independently of model generation."""

from pathlib import Path

from ..io import read_source_file
from ..models import MinimalSource, UnansweredQuestion
from ..retrieval import Bm25, one_question


class ContextBuilder:
    """Retrieve and format exact source passages for a language model."""

    def context(
        self, question: UnansweredQuestion, k: int, bm25: Bm25
    ) -> str:
        """Retrieve sources and format their exact passages as context.

        Args:
            question: Question used for source retrieval.
            k: Maximum number of sources.
            bm25: Initialized BM25 search engine.

        Returns:
            Numbered source passages, or an empty string if none match.
        """
        result = one_question(question, k=k, bm25=bm25)
        return self.context_from_sources(result.retrieved_sources)

    def context_from_sources(self, sources: list[MinimalSource]) -> str:
        """Build model context from an existing ranked source list.

        Args:
            sources: Ranked source paths and half-open character spans.

        Returns:
            Numbered passages preserving the original source text.

        Raises:
            ValueError: If a path escapes the project or a span is stale.
            OSError: If a source file cannot be read.
        """
        # TODO: Keep passages paired with source metadata so generation can
        # select complete passages within its token budget.
        text = ""
        for contx in sources:
            path = Path(contx.file_path).resolve()
            if not path.is_relative_to(Path.cwd().resolve()):
                raise ValueError("Source path is outside the project")
            # TODO: Compare a persisted source fingerprint before slicing;
            # bounds alone cannot detect edits that keep the same file length.
            file = read_source_file(path)
            if contx.last_character_index > len(file):
                raise ValueError("Source span exceeds current file length")
            text += (
                f"\n[{contx.file_path}]\n"
                + file[
                    contx.first_character_index : contx.last_character_index
                ]
                + "\n"
            )
        return text
