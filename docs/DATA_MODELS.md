# Data models and JSON contracts

The authoritative definitions live in `src/models.py`. Every external envelope
uses Pydantic v2 validation before processing or saving.

## Models required by the subject

| Model | Fields | Purpose |
| --- | --- | --- |
| `MinimalSource` | `file_path: str`, `first_character_index: int`, `last_character_index: int` | Exact source location |
| `UnansweredQuestion` | `question_id: str` with UUID default, `question: str` | One input question |
| `AnsweredQuestion` | Unanswered fields plus `sources: list[MinimalSource]`, `answer: str` | Reference question |
| `RagDataset` | `rag_questions: list[AnsweredQuestion \| UnansweredQuestion]` | Input dataset envelope |
| `MinimalSearchResults` | `question_id`, `question`, `retrieved_sources: list[MinimalSource]` | One search result |
| `MinimalAnswer` | Search-result fields plus `answer: str` | One generated answer |
| `StudentSearchResults` | `search_results: list[MinimalSearchResults]`, `k: int` | Search output envelope |
| `StudentSearchResultsAndAnswer` | `search_results: list[MinimalAnswer]`, `k: int` | Answer output envelope |

Do not interchange the reference field `sources` and output field
`retrieved_sources`. `question_id` joins predictions to references: preserving it
matters more than dataset order. Single questions receive fresh UUIDs. If an
unanswered dataset has no IDs, generating IDs independently in two separate
files cannot provide a reliable evaluation join; official references should
supply shared IDs.

## Unanswered and answered input examples

```json
{
  "rag_questions": [
    {"question_id": "q1", "question": "How is caching enabled?"}
  ]
}
```

```json
{
  "rag_questions": [
    {
      "question_id": "q1",
      "question": "How is caching enabled?",
      "sources": [
        {
          "file_path": "data/raw/demo/cache.md",
          "first_character_index": 0,
          "last_character_index": 164
        }
      ],
      "answer": "Set enable_cache=True."
    }
  ]
}
```

These illustrate the small original smoke corpus, not official vLLM ground truth.
See [CLI.md](CLI.md) for the corresponding output envelope.

## Internal models

| Model | Additional data | Invariant |
| --- | --- | --- |
| `Chunk` | Exact text, ranking-only context | Width equals text length and is 1..2000 |
| `IndexedFile` | SHA-256 digest, chunk list | Hash identifies the original bytes |
| `IndexData` | Versions, prefix, settings, files, lengths, postings | Stable chunk order matches posting IDs |
| `QuestionRecall` | Expected count, matched count, recall | One reference can be counted once |
| `EvaluationReport` | k, IoU threshold, macro recall, scored/skipped counts, question rows | Unscorable datasets are rejected |
| HTTP `QueryRequest` | Strict query string and bounded integer k | Query <=16000 characters, 0<=k<=100 |

The base source validates nonnegative, ordered endpoints but permits a zero-
width reference. Such a reference never overlaps. Indexed/retrieved evidence
must be nonempty and at most 2000 characters; generation additionally checks
bounds against the actual source file.

## Example local metric report

```json
{
  "k": 5,
  "iou_threshold": 0.05,
  "recall": 0.5,
  "evaluated_questions": 1,
  "skipped_without_sources": 0,
  "questions": [
    {"question_id": "q1", "expected": 2, "matched": 1, "recall": 0.5}
  ]
}
```

This extended report is for local iteration, not the official output format of
the school's executable. The implementation never imports that executable.
