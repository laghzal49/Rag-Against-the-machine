"""Generate answers from retrieved code with a local text pipeline."""

from tqdm import tqdm

from ..models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from .context import ContextBuilder

NO_EVIDENCE = "Insufficient retrieved evidence."
SYSTEM_PROMPT = (
    "Answer concisely using only the provided context. "
    "If evidence is insufficient, state so."
)


class Answer(ContextBuilder):
    """Answer questions with Qwen and a bounded context window."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 150,
        max_input_tokens: int = 2048,
    ) -> None:
        """Load the local text-generation pipeline and set token limits.

        Args:
            model_name: Hugging Face model ID or local model directory.
            max_new_tokens: Maximum number of tokens in each answer.
            max_input_tokens: Maximum number of tokens in the prompt.

        Raises:
            ValueError: If a token limit is invalid, the model has no
                tokenizer, or no room remains for the prompt.
        """
        from transformers import pipeline

        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("max_input_tokens", max_input_tokens),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")

        self.generator = pipeline("text-generation", model=model_name)
        tokenizer = self.generator.tokenizer
        if tokenizer is None:
            raise ValueError("The model has no tokenizer")
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        max_prompt_tokens = (
            self.generator.model.config.max_position_embeddings
            - max_new_tokens
        )
        if max_prompt_tokens < 1:
            raise ValueError("No model context remains for the prompt")
        self.max_input_tokens = min(max_input_tokens, max_prompt_tokens)

    def generate(self, question: UnansweredQuestion, context: str) -> str:
        """Answer from the leading context within the model token limit.

        Args:
            question: Question and identifier to answer.
            context: Retrieved source passages, ordered by relevance.

        Returns:
            Generated answer, or an insufficient-evidence message if the
            question or usable context is empty.

        Raises:
            ValueError: If the pipeline does not return answer text.
        """
        if not question.question.strip() or not context.strip():
            return NO_EVIDENCE

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question.question}\n\nContext:\n",
            },
        ]
        overhead = len(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        margin_safety = 32
        context_budget = self.max_input_tokens - overhead - margin_safety
        if context_budget < 1:
            return NO_EVIDENCE

        context_tokens = self.tokenizer.encode(
            context, add_special_tokens=False
        )
        if len(context_tokens) > context_budget:
            context = self.tokenizer.decode(
                context_tokens[:context_budget], skip_special_tokens=True
            )
        if not context.strip():
            return NO_EVIDENCE

        messages[1]["content"] += context
        output = self.generator(
            messages,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        answer = output[0]["generated_text"]
        if not isinstance(answer, str):
            raise ValueError("The pipeline returned an invalid answer")
        return answer.strip() or NO_EVIDENCE

    def generate_dataset(
        self, results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Answer each retrieved question while preserving its metadata.

        Args:
            results: Ranked sources and questions to answer.

        Returns:
            Answers with the original question IDs, sources, and k value.
        """
        answers: list[MinimalAnswer] = []
        for item in tqdm(results.search_results):
            question = UnansweredQuestion(
                question_id=item.question_id, question=item.question
            )
            context = self.context_from_sources(item.retrieved_sources)
            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=self.generate(question, context),
                )
            )
        return StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k
        )
