"""Generate grounded answers with a local language model."""

from tqdm import tqdm
from ..models import (
    UnansweredQuestion,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from .context import ContextBuilder


class Answer(ContextBuilder):
    """Generate grounded answers from retrieved source passages."""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 150
    ) -> None:
        """Initialize the text generation pipeline.

        Args:
            model_name: Hugging Face model ID or local model directory.
            max_new_tokens: Maximum number of tokens to generate.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if max_new_tokens >= self.model.config.max_position_embeddings:
            raise ValueError("max_new_tokens exceeds the model context budget")
        self.tokenizer.truncation_side = "right"

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
                    f"Question: {question.question}\n\nContext:\n{context}"
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
        dec: str = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
        return dec

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
