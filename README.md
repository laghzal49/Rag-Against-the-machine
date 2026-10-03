*This project has been created as part of the 42 curriculum by tlaghzal.*

# RAG Against the Machine

## Description

This project answers questions about the supplied vLLM 0.10.1 codebase. It reads
Python, Markdown, and text files, stores searchable source chunks, retrieves
ranked passages with BM25, and gives those passages to a local
`Qwen/Qwen3-0.6B` model. Retrieval results and answers use the JSON structures
specified in the subject. The default index covers `data/raw/` and is saved at
`data/processed/index.pkl`.

## Instructions

Use Python 3.10 or later and [uv](https://docs.astral.sh/uv/). The first `uv sync`
may download dependencies; the first answer may download the Qwen model. Place
the supplied vLLM tree at `data/raw/vllm-0.10.1/` and the public question JSON
files under `data/datasets/{UnansweredQuestions,AnsweredQuestions}/`. The raw
corpus, datasets, generated index, search results, and model weights are not part
of the Git submission.

For a fresh checkout, download the supplied vLLM archive before indexing:

```bash
mkdir -p data/raw
wget -q https://cdn.intra.42.fr/document/document/54840/vllm-0.10.1.zip \
  -O /tmp/vllm-0.10.1.zip
unzip -q /tmp/vllm-0.10.1.zip -d data/raw/
```

If the default corpus is absent, `index` downloads the same archive
automatically. The explicit commands above are useful when preparing an
offline evaluation workspace.

```bash
uv sync
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure OpenAI server?" --k 5
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

Run a public dataset from indexing through local recall measurement:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results/UnansweredQuestions \
  --k 5

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Replace `docs` with `code` in the dataset filenames to measure code retrieval.
Use separate output directories for datasets that share a filename. The answer
dataset command runs the local model for every result and may take much longer
than retrieval. `search` and `answer` print JSON to standard output; the dataset
commands write a JSON file named after their input file. All commands run as
`uv run python -m src <command>`. `search` accepts an optional `--index_path`,
and `index` accepts `--data_path`. Dataset input and output paths
are supplied as CLI arguments.

The Makefile provides `install`, `run`, `debug`, `clean`, and `lint`. For example,
`make lint` runs flake8 and mypy with the subject's required flags. The generated
index uses pickle and should only be loaded from a trusted local build.

## System architecture

```text
data/raw/ Python, Markdown, text
             |
             v
   file discovery and chunking
             |
             v
   tokenization and BM25 index ----> data/processed/index.pkl
             |
      single or dataset search
             |
             v
   ranked source file paths and character spans
             |
       source text loading
             |
             v
   Qwen/Qwen3-0.6B generation ----> answer JSON
```

`src/indexing/` discovers files and builds the index. `src/chunking/` splits
them, `src/tokenizer/` prepares terms, and `src/retrieval/` ranks chunks.
`src/generation/` reads the returned source spans and generates answers.
`src/models/` contains the Pydantic contracts for questions, sources, search
results, and answers. `src/commands/` exposes the six Python Fire commands;
`src/evaluation/` implements a local recall calculation.

## Chunking strategy

Python files use Tree-sitter top-level syntax nodes. Adjacent nodes are grouped
until the configured limit; a node larger than the limit is split into windows.
Markdown and plain text use character windows with up to 200 characters of
overlap. The default maximum is 2,000 characters, and validation rejects larger
values. Source offsets are half-open Unicode character positions, so
`file_text[first_character_index:last_character_index]` reproduces a chunk.
Tree-sitter's byte offsets are converted before storing them. Source paths retain
the `data/raw/...` prefix expected by the evaluator.

## Retrieval method and design decisions

The tokenizer lowercases text, filters common English stop words, and keeps
identifiers both whole and split at underscores or CamelCase boundaries. It
indexes passage text together with its source path. An inverted index finds
candidate chunks; BM25 scores term matches using frequency, inverse document
frequency, and document length normalization (`k1=1.5`, `b=0.5`). Results are
sorted by descending score and limited to `k` source spans.

Keeping exact paths and character ranges lets the evaluator check overlap and
lets generation read the original text. Lexical retrieval is fast and
inspectable, but a paraphrase with no shared terms may miss a relevant passage.
Overlapping prose windows help avoid losing context at a boundary, while smaller
windows increase the number of candidate chunks. The local generator defaults
to `Qwen/Qwen3-0.6B`; its code also accepts a model name when constructed in
Python, while the CLI uses the default.

## Performance analysis

The following measurements used the supplied vLLM 0.10.1 corpus and public
question datasets on this machine. Each row used a freshly built index, `k=5`,
and the supplied evaluator with `--max_context_length 2000`. The public docs
set has 100 questions; the public code set has 99.

| Maximum chunk size | Index time | Docs recall@5 | Code recall@5 |
| ---: | ---: | ---: | ---: |
| 2,000 characters | 7.56 s | 85.0% | 82.8% |
| 1,000 characters | 10.35 s | 84.0% | 71.7% |

At 2,000 characters the index contained 13,717 chunks; it scanned 1,969 files,
of which 1,847 contributed nonempty chunks. The
two dataset search commands took about 2.3 and 2.2 seconds respectively,
including separate process startup and index loading. These local measurements
are below the subject's 300-second indexing and 90-second retrieval limits.
Smaller chunks reduced public recall, especially for code, and roughly doubled
the number of indexed chunks. The correction sheet grades private datasets,
so these public scores do not establish private recall or answer quality. Its
thresholds are docs recall@5 of at least 80% and code recall@5 of at least 50%.

To reproduce the public scores, build an index with the chosen chunk size, run
`search_dataset` for both public unanswered datasets with `--k 5`, then run
`evaluate` against the corresponding answered datasets. The evaluator used for
the table was the separate supplied `moulinette` executable; the implementation
does not import or call it. The local `evaluate` command reports the same
recall@5 values for these public results.

## Challenges faced and limitations

Tree-sitter identifies syntax locations in UTF-8 bytes, while the required
output uses character indices. Converting each node boundary before storing a
span keeps the returned source slice exact. The code and documentation datasets
also use different vocabulary; indexing full identifiers and their component
words helps match both. Lexical retrieval still depends on term overlap.

The answer model can produce incomplete wording despite relevant evidence.
Source text is trimmed before chat formatting, and the Transformers
text-generation pipeline limits the prompt to 2,048 tokens and the answer to
150 tokens by default; long context can be truncated, so
lower-ranked sources may not reach the model. Verify grounding
against the returned source spans. Empty queries and zero-result requests
return empty sources or an insufficient-evidence answer; bad paths and malformed
JSON are reported as errors without an unhandled traceback.

## Resources and AI use

- [42 project subject](https://projects.intra.42.fr/projects/rag-against-the-machine)
- [vLLM repository](https://github.com/vllm-project/vllm)
- [Tree-sitter documentation](https://tree-sitter.github.io/tree-sitter/)
- [BM25 explanation](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)

AI assisted with documentation, type annotations, edge-case review, and style
checks. The implementation and reported measurements remain available for
inspection and independent verification.
