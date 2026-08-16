# Optional features and how to demonstrate them

The subject grades bonuses only after all mandatory requirements pass. These
features are implemented, but bonus eligibility cannot be established without
the official retrieval/performance evaluation.

## Semantic embeddings

```bash
uv sync --extra semantic
uv run --extra semantic python -m src index
uv run --extra semantic python -m src semantic_index
uv run --extra semantic python -m src search \
  "How are pending requests selected?" --method semantic --k 5
```

The default encoder is `sentence-transformers/all-MiniLM-L6-v2`, running on CPU.
It represents each chunk plus path/heading metadata as a normalized vector.
The saved files are `vectors.npy` and `semantic.json`. Querying takes a normalized
query vector, computes dot products, and returns top positions.

The model download is separate from mandatory Qwen. Its maximum input length
can truncate long chunk text, so a relevant sentence near the end might not
influence a vector. No claim is made that embeddings alone outperform BM25.
Use `--local_files_only=True` once the encoder cache is populated.

## Hybrid retrieval

```bash
uv run --extra semantic python -m src search \
  "How are pending requests selected?" --method hybrid --k 5
```

The candidate pool is up to `max(4*k,20)`, capped by corpus size, per method.
Reciprocal-rank fusion combines the lists with a constant of 60. The final list
contains unique chunk IDs. Inspect separate BM25, semantic, and hybrid results
for the same query. Measure docs/code recall independently: exact identifiers
often favor lexical scoring, while paraphrases may favor vectors.

## Incremental indexing

```bash
uv run python -m src index
uv run python -m src index
```

On an unchanged corpus the second report should show reused files and zero
changed files. Edit one source in a temporary copy, then re-index that copy:
only its chunks should be recomputed. Deleted files disappear from the index.
Changing the character limit, overlap, prefix or tokenizer/schema version forces
a rebuild of cached chunks.

This saves parsing/chunking work, but still hashes every file and rebuilds global
postings. Do not describe it as an incremental database update. Use
`--incremental=False` to rebuild when recovering from a corrupt index.

## Caching

Within one process, `Retriever` keeps the loaded index and 256 LRU query results.
`Generator` keeps one loaded model. `search_dataset` and the HTTP service benefit
from this reuse. Two separate `uv run ... search` calls do not share Python
memory; compare repeated requests to the same service instead.

After re-indexing, restart the service. A running process intentionally does not
poll the index file on every query. Stale source hashes are rejected on generation.

## Local HTTP API

```bash
uv run python -m src serve --host 127.0.0.1 --port 8000
```

Endpoints:

| Method/path | Request | Response |
| --- | --- | --- |
| `GET /health` | No body | Status and retrieval method |
| `POST /search` | `{"query":"cache","k":5}` | `StudentSearchResults` |
| `POST /answer` | Same shape | `StudentSearchResultsAndAnswer` |

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do I enable LoRA?","k":5}'
```

The server binds to loopback only, runs serially, and validates request limits.
It has no authentication or production concurrency layer and is intended as a
local project demonstration. Model failures return an error rather than invented
answers. The first answer may take longer while the model loads. Ctrl-C closes
the listening socket.
