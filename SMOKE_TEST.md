# Portfolio smoke test

Run this checklist from a clean clone at the repository root.

## 1. Reproducible retrieval demo

```bash
uv sync --locked

uv run python -m src index \
  --data_path demo/corpus \
  --index_path demo/index.pkl

uv run python -m src search \
  "When is an expired cache entry removed?" \
  --index_path demo/index.pkl \
  --k 3
```

Pass conditions:

- the commands exit successfully;
- at least one retrieved source is returned;
- the first result references `demo/corpus/cache.py`;
- every source span satisfies `0 <= start < end <= len(file_text)`;
- `file_text[start:end]` contains the evidence described by that result.

The generated `demo/index.pkl` is local output and should not be committed.

## 2. CLI failure behavior

```bash
uv run python -m src search "cache" \
  --index_path demo/missing.pkl

uv run python -m src search "cache" \
  --index_path demo/index.pkl \
  --k=-1

uv run python -m src index \
  --data_path demo/missing-corpus \
  --index_path demo/index.pkl
```

Pass condition: every command fails with a concise error and non-zero exit code,
without a Python traceback.

## 3. Grounded generation

```bash
uv run python -m src answer \
  "When is an expired cache entry removed?" \
  --index_path demo/index.pkl \
  --k 3
```

Pass conditions:

- the first run clearly handles the Qwen model download/load;
- output is valid JSON;
- retrieved sources are preserved in the answer object;
- the answer says that `get` removes an expired entry when that key is read,
  or that `purge_expired` removes all expired entries;
- the answer does not introduce behavior absent from the retrieved source.

## 4. Static checks

```bash
make lint
uv run python -m src --help
```

Pass condition: both commands exit successfully.

## 5. Official dataset check

With the supplied corpus and public question datasets installed:

```bash
uv run python -m src index --max_chunk_size 2000

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results/UnansweredQuestions \
  --k 5

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5
```

Record the commit SHA, Python version, operating system, CPU, corpus version,
question count, indexing time, retrieval time, and recall@5. Never present a
measurement without this context.

## Release gate

The project is portfolio-ready when all of these are true:

- [ ] A clean clone completes the retrieval demo.
- [ ] Every returned span is an exact source slice.
- [ ] Invalid inputs produce controlled failures.
- [ ] Lint and type checks pass.
- [ ] One grounded answer has been manually inspected.
- [ ] Official recall numbers have been rerun or clearly labeled as historical.
- [ ] The README contains no command or feature absent from the current code.
