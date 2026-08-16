# Command reference

Every command runs as `uv run python -m src COMMAND [options]`.
Use `--option=value` when passing a negative value to Fire during an error test.
All relative paths are relative to the process working directory; run from the
project root as the subject expects. `project_root` only controls the corpus
source-path base; use absolute paths for other options when running elsewhere.

| Command | Main options | Output |
| --- | --- | --- |
| `index` | `max_chunk_size=2000`, `corpus_path=data/raw`, `index_directory=data/processed`, `project_root=.`, `overlap=None`, `incremental=True` | `index.json`, `index_report.json` |
| `search QUERY` | `k=5`, `index_directory`, `output_path`, `method=bm25`, `k1=1.5`, `b=0.75`, `local_files_only=False` | Search envelope; default `data/output/search_results/single.json` |
| `search_dataset` | required `dataset_path`; `k=5`, `save_directory`, `index_directory`, `method`, `k1`, `b`, `local_files_only` | Search envelope named after input dataset |
| `answer QUERY` | `k=5`, `index_directory`, `project_root`, `output_path`, `method`, `model_name`, `context_tokens=8192`, `max_new_tokens=384`, `local_files_only`, `device=cpu` | Answer envelope; default `data/output/search_results_and_answer/single.json` |
| `answer_dataset` | required `student_search_results_path`; `save_directory`, `index_directory`, `project_root`, `model_name`, `context_tokens`, `max_new_tokens`, `local_files_only`, `device` | Answer envelope named after search file |
| `evaluate` | required `student_search_results_path`, `dataset_path`; `k=None`, `iou_threshold=0.05`, `output_path=data/output/evaluation.json` | Local recall report |
| `semantic_index` | `index_directory`, `model_name=sentence-transformers/all-MiniLM-L6-v2`, `local_files_only` | `vectors.npy`, `semantic.json` |
| `serve` | `host=127.0.0.1`, `port=8000`, `index_directory`, `project_root`, `method`, `model_name`, `local_files_only` | Foreground local HTTP service |

Generation defaults to `Qwen/Qwen3-0.6B`. You can pass a local model directory
with `--model_name=/absolute/path --local_files_only=True`. No remote inference
API is used. `--device=cuda` or `--device=mps` requires a compatible environment;
the supplied Linux lock configuration intentionally uses CPU PyTorch.

## Search JSON

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How is caching enabled?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/demo/cache.md",
          "first_character_index": 0,
          "last_character_index": 120
        }
      ]
    }
  ],
  "k": 5
}
```

The example illustrates the schema, not a source in the supplied official
corpus. Answer output adds `answer` inside each question entry. Source ranges
are half-open: `[first_character_index, last_character_index)`.

## Degenerate inputs

| Input | Behavior |
| --- | --- |
| Empty/whitespace query | Empty retrieved list; answer explicitly abstains |
| Query with no known BM25 terms | Empty retrieved list |
| Nonsense with a known identifier | May return matching sources; semantics are not guaranteed |
| `k=0` | Valid empty retrieved list |
| Negative/invalid `k` | Nonzero error without traceback |
| `k` greater than corpus size | Up to the available matching chunks |
| Missing/corrupt index or JSON | Nonzero error without fabricated output |
| Empty corpus | Error; keeps previous index |
| Empty question dataset | Valid empty batch output |
| Duplicate question IDs | Rejected in batch/evaluation commands |
| No ground-truth sources for evaluation | Error; no meaningless recall value |
| Changed corpus during generation | Error asking to re-index and search again |
| Missing weights in offline mode | Clear failure; no substitute model or fake answer |

Evaluation ignores unanswered/no-source references with an explicit count,
counts missing predictions as zero, and rejects unknown/duplicate prediction IDs.
A reference span may exceed 2000 characters; a retrieved span may not.
