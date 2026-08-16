# Benchmarking with the official attachments

## What is still required

The school's exact vLLM corpus, answered/unanswered question datasets and target
machine are needed to establish the mandatory thresholds. The supplied PDF
contains requirements and illustrative examples, not measurable ground truth.

Before comparing scores, confirm corpus path prefixes match references exactly.
Never copy official reference answers into retrieval rules, tune on private test
answers, or make the application import/call the official evaluator.

## Baseline commands

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5 --output_path data/output/recall_docs.json
```

Repeat with the exact code dataset filename provided by the school. The local
metric aids debugging. Run the school's evaluation tool yourself using the
instructions in the subject; that independent result is authoritative.

## Chunk-size experiment

Use independent directories so you can compare identical datasets against
several indices. On bash:

```bash
for size in 500 1000 1500 2000; do
  uv run python -m src index \
    --max_chunk_size "$size" \
    --index_directory "data/processed/chunks_$size" \
    --incremental=False
  uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --index_directory "data/processed/chunks_$size" \
    --k 5 --save_directory "data/output/chunks_$size/UnansweredQuestions"
  uv run python -m src evaluate \
    --student_search_results_path "data/output/chunks_$size/UnansweredQuestions/dataset_docs_public.json" \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 5 --output_path "data/output/chunks_$size/recall_docs.json"
done
```

Repeat separately for code. Keep the default 10% overlap during this first
experiment; change one factor at a time. Next investigate explicit overlap, k1,
b and retrieval method if baseline recall is insufficient.

## Record real results

| Chunk chars | Chunk count | Index seconds | 200-query seconds | Docs recall@5 | Code recall@5 |
| --- | --- | --- | --- | --- | --- |
| 500 | Not measured | Not measured | Not measured | Not measured | Not measured |
| 1000 | Not measured | Not measured | Not measured | Not measured | Not measured |
| 1500 | Not measured | Not measured | Not measured | Not measured | Not measured |
| 2000 | Not measured | Not measured | Not measured | Not measured | Not measured |

Required thresholds: index <=300 s, 200 queries <=90 s, docs >=0.80, code >=0.50.
Record CPU, Python version, memory, corpus revision, dependency lock, query count,
method, chunk settings and whether the run started cold. Model generation is
not part of the lexical retrieval throughput number.

## Investigate a missed source

1. Locate the question in the saved local evaluation report.
2. Compare reference and retrieved paths verbatim.
3. Open the expected source interval and inspect its indexed chunk boundaries.
4. Compare query tokens with the correct chunk's tokens.
5. Search an exact identifier from the reference region.
6. Decide whether the problem is ingestion, segmentation, vocabulary or ranking.

If the correct file was never indexed, ranking changes cannot fix it. If only
paraphrases fail, try semantic/hybrid retrieval. If very short chunks fail IoU,
inspect coverage and source widths. Evaluate changes on public data and keep
separate held-out questions where available.
