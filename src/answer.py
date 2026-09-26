"""Generate grounded answers with a local language model."""

from pathlib import Path

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
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

    def __init__(
        self, model_name: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 150
    ) -> None:
        """Initialize the text generation pipeline.

        Args:
            model_name: Hugging Face model ID or local model directory.
            max_new_tokens: Maximum number of tokens to generate.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer.truncation_side = "left"

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
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings
            - self.max_new_tokens,
        )
        output = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens
        )
        inputs_len = inputs["input_ids"].shape[1]
        generated = output[0][inputs_len:]
        dec = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
        return dec

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
