*This project was created as part of the 42 curriculum by tlaghzal.*

# RAG Against the Machine

A local codebase RAG pipeline that indexes Python and documentation files,
retrieves exact character spans with BM25, and answers questions with
`Qwen/Qwen3-0.6B` using the retrieved evidence.

Unlike a framework-only RAG demo, the core retrieval pipeline is implemented in
this repository: syntax-aware chunking, tokenization, inverted-index
construction, BM25 scoring, source-span validation, evaluation, and the CLI.

## What it demonstrates

- Tree-sitter chunking for Python and overlapping windows for Markdown/text
- UTF-8 byte-to-character offset conversion for exact `text[start:end]` spans
- BM25 with an inverted index, stop-word filtering, and identifier splitting
- Pydantic contracts for questions, sources, retrieval results, and answers
- Persisted indexes and single-query or dataset CLI workflows
- Local answer generation with a small Qwen model

## Quick demo

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and network access
for the first dependency/model download.

```bash
git clone https://github.com/laghzal49/rag-v2.git
cd rag-v2
uv sync --locked

uv run python -m src index \
  --data_path demo/corpus \
  --index_path demo/index.pkl

uv run python -m src search \
  "When is an expired cache entry removed?" \
  --index_path demo/index.pkl \
  --k 3
```

The first result should reference `demo/corpus/cache.py`. Inspect the returned
half-open offsets against `file_text[start:end]` to verify that the evidence is
an exact slice of the indexed source.

Generate a grounded answer with the same index:

```bash
uv run python -m src answer \
  "When is an expired cache entry removed?" \
  --index_path demo/index.pkl \
  --k 3
```

The answer command loads `Qwen/Qwen3-0.6B`; the first run downloads the model
unless it is already cached. Retrieval itself does not require the model.

## Pipeline

```text
.py / .md / .txt
       |
       v
syntax-aware chunking
       |
       v
BM25 inverted index ----> index.pkl
       |
       v
ranked source spans
       |
       v
bounded source context
       |
       v
Qwen3-0.6B answer
```

## Key engineering decisions

### Syntax-aware, bounded chunks

Python files are parsed with Tree-sitter so chunks follow top-level syntax
boundaries where possible. Oversized nodes and prose files fall back to
overlapping windows. Every chunk is bounded by `max_chunk_size <= 2000`.

Tree-sitter returns byte offsets while Python slices use Unicode character
offsets. The chunker converts between them so every retrieved span can be
verified directly against the original UTF-8 text.

### Code-aware BM25

The tokenizer lowercases terms, removes common English stop words, and expands
snake_case and CamelCase identifiers while retaining the full identifier.
An inverted index limits scoring to chunks containing query terms.

The default BM25 parameters are `k1=1.5` and `b=0.5`.

### Explicit data contracts

Pydantic models validate source spans, questions, retrieval results, and
generated answers between pipeline stages. Saved search results can be evaluated
independently from answer generation.

## Main commands

```bash
# Index the official corpus (default data path and index path)
uv run python -m src index --max_chunk_size 2000

# Search one question
uv run python -m src search "What is PagedAttention?" --k 5

# Answer one question
uv run python -m src answer "What is PagedAttention?" --k 5

# Search a dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results/UnansweredQuestions \
  --k 10

# Evaluate saved retrieval results
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

Run `uv run python -m src --help` or append `--help` to a command for its
arguments.

## Project structure

| Path | Responsibility |
| --- | --- |
| `src/chunking/splitters.py` | Python AST and text-window chunking |
| `src/retrieval/bm25.py` | Tokenization, postings, ranking, persistence |
| `src/indexing/indexer.py` | Corpus discovery and index construction |
| `src/retrieval/search.py` | Single and batch retrieval |
| `src/generation/` | Validated source context and local generation |
| `src/io/` | Exact source reads and dataset JSON input/output |
| `src/evaluation/metrics.py` | Character-span recall evaluation |
| `src/models/schemas.py` | Pydantic data contracts |
| `src/cli.py` | Fire command-line interface |
| `demo/corpus/` | Small reproducible corpus |
| `SMOKE_TEST.md` | Final verification checklist |

## Validation

Local development runs on the supplied vLLM 0.10.1 corpus previously reported
approximately 85% documentation recall@5 and 82.8% code recall@5. These are
machine- and dataset-dependent measurements, not guarantees. Use the commands
in `SMOKE_TEST.md` to reproduce the demo and rerun official evaluation.

## Limitations

- Retrieval is lexical: synonyms absent from the corpus/query can be missed.
- The small generator can still produce incorrect wording; retrieved spans are
  preserved so answers can be inspected.
- Pickle indexes must only be loaded from trusted sources and rebuilt after
  corpus changes.
- The official 42 corpus, datasets, model weights, and generated outputs are not
  committed.

## Development

Search for `TODO:` in `src/`, `tests/`, and `Makefile` for improvements
listed beside the code that needs changing.

```bash
make install
make lint
make clean
```

See `SMOKE_TEST.md` for the release checklist.

## Resources

- [vLLM](https://github.com/vllm-project/vllm)
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [Okapi BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic](https://docs.pydantic.dev/)

## AI use

AI assisted with documentation, type annotations, edge-case review, and style
checks. The retrieval design and implementation remain inspectable in the
repository.
