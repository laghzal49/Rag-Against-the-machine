"""Generate grounded answers with a local language model."""

from typing import Any
from pathlib import Path

from tqdm import tqdm
from transformers import pipeline

from .file_reader import read_source_file
from .modele import (
    MinimalSource,
    UnansweredQuestion,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from .retrieval import Retrieval


class Answer:
    """Generate grounded answers from retrieved source passages."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the text generation pipeline.

        Args:
            model_name: Hugging Face model ID or local model directory.
        """
        self.pipe: Any = pipeline("text-generation", model=model_name)

    def generate(self, question: UnansweredQuestion, context: str) -> str:
        """Answer from cited context or abstain when evidence is empty.

        Args:
            question: Input question, including its preserved identifier.
            context: Numbered evidence passages to supply to the model.

        Returns:
            Generated answer text, or an insufficient-evidence message.
        """
        if not context.strip() or not question.question.strip():
            return "Insufficient retrieved evidence."

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer concisely using only the provided context. "
                    "If evidence is insufficient, state so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\nQuestion: {question.question}"
                ),
            },
        ]
        text = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = self.pipe(text, max_new_tokens=150, return_full_text=False)
        return str(out[0]["generated_text"]).strip()

    def context(
        self, question: UnansweredQuestion, k: int, re: Retrieval
    ) -> str:
        """Retrieve sources and format their exact passages as context.

        Args:
            question: Question used for source retrieval.
            k: Maximum number of sources.
            re: Retrieval service backed by a loaded index.

        Returns:
            Numbered source passages, or an empty string if none match.
        """
        result = re.one_question(question, k=k)
        return self.context_from_sources(result.retrieved_sources)

    def generate_dataset(
        self, results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Answer each search result while preserving IDs and sources.

        Args:
            results: Previously retrieved questions and ranked sources.

        Returns:
            Dataset of generated answers with the original search limit.
        """
        anwers: list[MinimalAnswer] = []
        for question in tqdm(results.search_results):
            qt = UnansweredQuestion(
                question_id=question.question_id, question=question.question
            )
            contx = self.context_from_sources(question.retrieved_sources)
            respond = self.generate(qt, contx)
            anwers.append(
                MinimalAnswer(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=question.retrieved_sources,
                    answer=respond,
                )
            )
        retu = StudentSearchResultsAndAnswer(
            search_results=anwers, k=results.k
        )
        return retu

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
        text = ""
        for contx in sources:
            path = Path(contx.file_path).resolve()
            if not path.is_relative_to(Path.cwd().resolve()):
                raise ValueError("Source path is outside the project")
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
